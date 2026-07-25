# -*- coding: utf-8 -*-
"""YH_ultra v2 每日信号 + K线图 (MA200趋势 + TOP3动量)"""
import sys, io, os, json, ssl, time, base64, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import akshare as ak, pandas as pd, numpy as np
import urllib.request as ur
import matplotlib; matplotlib.use('Agg')
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

_fonts = [f.name for f in fm.fontManager.ttflist]
CN = 'WenQuanYi Zen Hei' if 'WenQuanYi Zen Hei' in _fonts else ('SimHei' if 'SimHei' in _fonts else 'DejaVu Sans')
plt.rcParams['font.sans-serif'] = [CN]; plt.rcParams['axes.unicode_minus'] = False

MAIN_SYM = 'sh512890'; MAIN_NAME = '红利低波'
GOLD_SYM = 'sh518880'; GOLD_NAME = '黄金ETF'
GROWTH = {'创业板':'sz159915', '科创50':'sh588000', '人工智能':'sh515070'}
ALL_NAMES = [MAIN_NAME, GOLD_NAME] + list(GROWTH.keys())
ALL_SYMS = {MAIN_NAME: MAIN_SYM, GOLD_NAME: GOLD_SYM, **GROWTH}

BB_P = 45; BB_S = 2.0; RSI_P = 14; RSI_L = 30; RSI_H = 70; ERS = 65
MA_TREND = 200; MOM_P = 20; TOP_N = 3
BARK_KEYS = []  # 已关闭推送
REPO = 'sunran1996/my_candle'
W_DEFENSE = {MAIN_NAME: 0.50, GOLD_NAME: 0.50}

def fetch():
    dfs = {}
    for n, s in ALL_SYMS.items():
        df = ak.fund_etf_hist_sina(symbol=s); df['date'] = pd.to_datetime(df['date'])
        dfs[n] = df[['date','open','high','low','close','volume']].sort_values('date').reset_index(drop=True)
    return dfs

def add_main(df):
    df = df.copy()
    r = df['close'].pct_change().fillna(0); r[abs(r) > 0.1] = 0
    df['adj'] = (1 + r).cumprod()
    df['ma200'] = df['close'].rolling(MA_TREND).mean()
    df['bb_ma'] = df['close'].rolling(BB_P).mean()
    df['bb_std'] = df['close'].rolling(BB_P).std()
    df['up'] = df['bb_ma'] + BB_S * df['bb_std']
    df['lo'] = df['bb_ma'] - BB_S * df['bb_std']
    d = df['adj'].diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    df['rsi'] = 100 - 100 / (1 + g.ewm(alpha=1/RSI_P, adjust=False).mean() /
                   l.ewm(alpha=1/RSI_P, adjust=False).mean().replace(0, np.nan))
    return df

def add_growth(df):
    df = df.copy()
    df['mom'] = df['close'] / df['close'].shift(MOM_P) - 1
    e10 = df['close'].ewm(span=10, adjust=False).mean()
    e20 = df['close'].ewm(span=20, adjust=False).mean()
    df['macd_line'] = e10 - e20
    df['macd_h'] = df['macd_line'] - df['macd_line'].ewm(span=7, adjust=False).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    return df

def add_all_mom(dfs):
    out = {}
    for n, df in dfs.items():
        d = df.copy(); d['mom'] = d['close'] / d['close'].shift(MOM_P) - 1; out[n] = d
    return out

def send_bark(title, body, url=''):
    if not BARK_KEYS: return
    for bk in BARK_KEYS:
        try:
            data = json.dumps({'title':title,'body':body,'url':url}).encode()
            ur.urlopen(ur.Request(f'https://api.day.app/{bk}',data=data,
                       headers={'Content-Type':'application/json'}),timeout=10)
        except: pass

def main():
    try:
        print("获取数据...")
        raw = fetch(); df_main = add_main(raw[MAIN_NAME])
        dfs_g = {n: add_growth(d) for n, d in raw.items() if n in GROWTH}
        dfs_all_mom = add_all_mom(raw)

        is_weekend = pd.Timestamp.now().dayofweek >= 5
        if not is_weekend:
            try:
                spot = ak.fund_etf_spot_em()
                code_map = {'512890': MAIN_NAME, '518880': GOLD_NAME,
                            '159915': '创业板', '588000': '科创50', '515070': '人工智能'}
                for code, name in code_map.items():
                    s = spot[spot['代码'] == code]
                    if len(s) > 0:
                        rt = float(s['最新价'].iloc[0])
                        raw[name].loc[raw[name].index[-1], 'close'] = rt
                        raw[name].loc[raw[name].index[-1], 'date'] = pd.Timestamp.now()
                        print(f'  {name} → 实时 {rt:.4f}')
                df_main = add_main(raw[MAIN_NAME])
                dfs_g = {n: add_growth(raw[n]) for n in GROWTH}
                dfs_all_mom = add_all_mom(raw)
            except Exception as e: print(f'  实时行情失败: {e}')

        idx = -1; row = df_main.iloc[idx]; date = row['date']
        price = row['close']; rsi = row['rsi']; lo = row['lo']; up = row['up']
        ma200 = row['ma200']
        bb_pos = (price - lo) / (up - lo) * 100 if up > lo else 50

        # 趋势
        above_trend = not pd.isna(ma200) and price >= ma200
        trend_pct = (price / ma200 - 1) * 100 if not pd.isna(ma200) else 0

        # 动量排名
        ranking = []
        for n in ALL_NAMES:
            v = dfs_all_mom[n]['mom'].iloc[-1]
            if not pd.isna(v): ranking.append((n, v))
        ranking.sort(key=lambda x: x[1], reverse=True)

        # 状态
        if not above_trend:
            state = 'CASH'; emoji = '💤'
            action = f'{emoji} 空仓观望'
            detail = f'{MAIN_NAME} {price:.3f} < MA{MA_TREND} {ma200:.3f} ({trend_pct:+.1f}%)'
            color = '#888888'
        else:
            bb_buy = price <= lo; bb_sell = price >= up; rsi_buy = rsi <= RSI_L
            buy_ok = bb_buy or rsi_buy; sell_ok = (bb_sell and rsi >= ERS)
            best_n, best_v = None, -99
            for n in GROWTH:
                v = dfs_g[n]['macd_h'].iloc[-1]
                if not pd.isna(v) and v > best_v: best_v, best_n = v, n
            if buy_ok:
                state = 'DEFENSE'; emoji = '🛡'
                action = f'{emoji} 防御模式'
                detail = f'{MAIN_NAME}50% + {GOLD_NAME}50%'
                color = '#3498DB'
            elif sell_ok and best_n and best_v > 0:
                state = 'ATTACK'; emoji = '🚀'
                action = f'{emoji} 进攻模式'
                detail = f'100% {best_n}  MACD{best_v:+.3f}'
                color = '#E74C3C'
            else:
                state = 'NEUTRAL'; emoji = '⚖'
                top3 = [n for n, _ in ranking[:TOP_N]]
                action = f'{emoji} TOP{TOP_N}动量'
                detail = ' + '.join(top3)
                color = '#2ECC71'

        warn = ''
        if not above_trend:
            warn = ''
        else:
            near_buy = (bb_pos < 35 or rsi < 45) and not bb_buy and not (rsi <= RSI_L)
            near_sell = (bb_pos > 65 or rsi > 60) and not (bb_sell and rsi >= ERS)
            if near_sell: warn = ' ⚠ 接近卖出'
            elif near_buy: warn = ' ⚠ 接近买入'

        rank_str = ' > '.join(f'{n}({v:+.1%})' for n, v in ranking)
        chg_str = ' | '.join(
            f'{n} {raw[n]["close"].pct_change().iloc[-1]*100:+.2f}%'
            for n in ALL_NAMES)

        # ===== 120日迷你回测 =====
        lookback = 120
        main_close = raw[MAIN_NAME]['close'].tail(lookback).reset_index(drop=True)
        gold_close = raw[GOLD_NAME]['close'].tail(lookback).reset_index(drop=True)
        growth_close = {n: raw[n]['close'].tail(lookback).reset_index(drop=True) for n in GROWTH}
        m_sub = df_main.tail(lookback).reset_index(drop=True)
        g_sub = {n: dfs_g[n].tail(lookback).reset_index(drop=True) for n in GROWTH}
        mom_sub = {n: dfs_all_mom[n].tail(lookback).reset_index(drop=True) for n in ALL_NAMES}

        def px_at(i, n):
            if n in growth_close: return growth_close[n].iloc[i]
            elif n == MAIN_NAME: return main_close.iloc[i]
            elif n == GOLD_NAME: return gold_close.iloc[i]
            return 0

        INIT = 1_000_000; cash = INIT
        shares = {n: 0.0 for n in ALL_NAMES}
        back_state = 'CASH'; peak = INIT; last_trade_cd = 0
        hse = 0; ep = {}; top3_held = []
        navs = []

        for i in range(lookback):
            r2 = m_sub.iloc[i]
            close_p, rsi2, lo2, up2, ma200_2 = r2['close'], r2['rsi'], r2['lo'], r2['up'], r2['ma200']
            above = not pd.isna(ma200_2) and close_p >= ma200_2

            nav = cash + sum(shares[n] * px_at(i, n) for n in ALL_NAMES)
            if nav > peak: peak = nav

            # NAV止损
            dd = (nav - peak) / peak if peak > 0 else 0
            if dd < -0.10 and any(shares[n] > 0 for n in ALL_NAMES):
                for n in ALL_NAMES:
                    if shares[n] > 0: cash += shares[n] * px_at(i, n) * 0.9997; shares[n] = 0.0
                peak = nav; back_state = 'CASH'; top3_held = []; ep = {}
                navs.append(nav); continue

            cd_ok = last_trade_cd <= 0

            if not above:
                if back_state != 'CASH':
                    if cd_ok:
                        for n in ALL_NAMES:
                            if shares[n] > 0: cash += shares[n] * px_at(i, n) * 0.9997; shares[n] = 0.0
                        back_state = 'CASH'; top3_held = []; last_trade_cd = 3
                        navs.append(cash); continue
                    # else: cooldown blocked, fall through to normal nav calc
                else:
                    navs.append(cash); continue

            # 信号
            if pd.isna(lo2) or pd.isna(rsi2):
                navs.append(nav); continue

            bb_b = close_p <= lo2; bb_s = close_p >= up2; rsi_b = rsi2 <= RSI_L
            buy_ok2 = bb_b or rsi_b; sell_ok2 = (bb_s and rsi2 >= ERS)

            best_n2, best_v2 = None, -99
            if i < max(len(g_sub.get(n, pd.DataFrame())) for n in GROWTH):
                for n in GROWTH:
                    if i < len(g_sub[n]):
                        v = g_sub[n]['macd_h'].iloc[i]
                        if not pd.isna(v) and v > best_v2: best_v2, best_n2 = v, n

            if buy_ok2:
                new_state = 'DEFENSE'
            elif sell_ok2 and best_n2 and best_v2 > 0:
                new_state = 'ATTACK'
            elif sell_ok2:
                new_state = 'DEFENSE'
            else:
                new_state = 'NEUTRAL'

            if new_state != back_state and cd_ok:
                for n in ALL_NAMES:
                    if shares[n] > 0: cash += shares[n] * px_at(i, n) * 0.9997; shares[n] = 0.0
                if new_state == 'ATTACK' and best_n2:
                    cp = px_at(i, best_n2)
                    if cp > 0: shares[best_n2] = nav / cp * 0.9997; cash = 0; hse = cp; ep[best_n2] = cp
                    top3_held = []
                elif new_state == 'DEFENSE':
                    for n, w in W_DEFENSE.items():
                        cp = px_at(i, n)
                        if cp > 0: shares[n] = nav * w / cp * 0.9997; cash -= nav * w; ep[n] = cp
                    top3_held = []
                else:
                    # NEUTRAL: TOP3动量
                    mom_rank = []
                    for n in ALL_NAMES:
                        if i < len(mom_sub[n]):
                            v = mom_sub[n]['mom'].iloc[i]
                            if not pd.isna(v): mom_rank.append((n, v))
                    mom_rank.sort(key=lambda x: x[1], reverse=True)
                    top3 = [n for n, _ in mom_rank[:TOP_N]]
                    w_each = 1.0 / TOP_N
                    for n in top3:
                        cp = px_at(i, n)
                        if cp > 0: shares[n] = nav * w_each / cp * 0.9997; cash -= nav * w_each; ep[n] = cp
                    top3_held = top3
                back_state = new_state; last_trade_cd = 3

            if last_trade_cd > 0: last_trade_cd -= 1

            # 同状态
            if back_state == 'ATTACK':
                tgt = None
                for n in GROWTH:
                    if shares[n] > 0: tgt = n; break
                if tgt:
                    cp = px_at(i, tgt)
                    if cp > hse: hse = cp
                    if cp < hse * 0.94:  # -6% trailing → NEUTRAL
                        cash += shares[tgt] * cp * 0.9997; shares[tgt] = 0.0
                        mom_rank = []
                        for n in ALL_NAMES:
                            if i < len(mom_sub[n]):
                                v = mom_sub[n]['mom'].iloc[i]
                                if not pd.isna(v): mom_rank.append((n, v))
                        mom_rank.sort(key=lambda x: x[1], reverse=True)
                        top3 = [n for n, _ in mom_rank[:TOP_N]]
                        w_each = 1.0 / TOP_N
                        nav2 = cash
                        for n in top3:
                            cp2 = px_at(i, n)
                            if cp2 > 0: shares[n] = nav2 * w_each / cp2 * 0.9997; cash -= nav2 * w_each
                        back_state = 'NEUTRAL'; top3_held = top3; last_trade_cd = 3
            elif back_state == 'NEUTRAL' and cd_ok:
                mom_rank = []
                for n in ALL_NAMES:
                    if i < len(mom_sub[n]):
                        v = mom_sub[n]['mom'].iloc[i]
                        if not pd.isna(v): mom_rank.append((n, v))
                mom_rank.sort(key=lambda x: x[1], reverse=True)
                new_top3 = [n for n, _ in mom_rank[:TOP_N]]
                if set(new_top3) != set(top3_held):
                    for n in ALL_NAMES:
                        if shares[n] > 0: cash += shares[n] * px_at(i, n) * 0.9997; shares[n] = 0.0
                    w_each = 1.0 / TOP_N
                    nav2 = cash
                    for n in new_top3:
                        cp = px_at(i, n)
                        if cp > 0: shares[n] = nav2 * w_each / cp * 0.9997; cash -= nav2 * w_each
                    top3_held = new_top3; last_trade_cd = 3

            # 硬止损
            for n in list(ALL_NAMES):
                if shares[n] > 0 and n in ep and ep.get(n, 0) > 0:
                    cp = px_at(i, n)
                    if cp > 0 and cp < ep[n] * 0.92:  # -8%
                        cash += shares[n] * cp * 0.9997; shares[n] = 0.0

            nav = cash + sum(shares[n] * px_at(i, n) for n in ALL_NAMES)
            navs.append(nav)

        navs = np.array(navs); navs = navs / navs[0] if navs[0] > 0 else navs

        # ===== 图表 =====
        fig = plt.figure(figsize=(6, 10), facecolor='#FAFAFA')
        gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 2.5, 1.2],
                              hspace=0.2, left=0.06, right=0.94, top=0.96, bottom=0.03)

        ax0 = fig.add_subplot(gs[0]); ax0.axis('off'); ax0.set_ylim(0, 8)
        ax0.text(0, 7.2, f'YH_ultra v2  {date.strftime("%Y-%m-%d")}',
                 fontsize=14, fontweight='bold', color='#1A1A1A')
        ax0.text(0, 5.2, f'{action}{warn}',
                 fontsize=18, fontweight='bold', color=color)
        ax0.text(0, 3.7, f'{detail}', fontsize=10, color='#555')
        ax0.text(0, 2.5, f'MA{MA_TREND}:{ma200:.3f} ({trend_pct:+.1f}%)  RSI{rsi:.0f}  BB{bb_pos:.0f}%',
                 fontsize=10, color='#888')
        ax0.text(0, 1.3, f'动量: {rank_str}', fontsize=8, color='#AAA')
        ax0.text(0, 0.3, f'{chg_str}', fontsize=7.5, color='#BBB')

        cn_c = mpf.make_marketcolors(up='#CC0000', down='#008800', edge='inherit',
                                      wick='inherit', volume='inherit')
        cn_s = mpf.make_mpf_style(marketcolors=cn_c, gridstyle='',
                                   rc={'font.sans-serif':[CN], 'axes.unicode_minus':False})

        ohlc = raw[MAIN_NAME].tail(lookback).copy()
        ohlc = ohlc.rename(columns={'open':'Open','high':'High','low':'Low',
                                     'close':'Close','volume':'Volume'})
        ohlc = ohlc.set_index('date')[['Open','High','Low','Close','Volume']]
        bb_df = df_main.tail(lookback)

        ax1 = fig.add_subplot(gs[1])
        ap_ma = mpf.make_addplot(bb_df['bb_ma'].values, color='#888', width=0.8, linestyle='--', ax=ax1)
        ap_up = mpf.make_addplot(bb_df['up'].values, color='#9B59B6', width=0.6, linestyle='--', ax=ax1)
        ap_lo = mpf.make_addplot(bb_df['lo'].values, color='#9B59B6', width=0.6, linestyle='--', ax=ax1)
        ap_200 = mpf.make_addplot(bb_df['ma200'].values, color='#333', width=1.2, linestyle='-', ax=ax1)
        mpf.plot(ohlc, type='candle', ax=ax1, volume=False, style=cn_s,
                 addplot=[ap_ma, ap_up, ap_lo, ap_200])
        ax1.set_title(f'{MAIN_NAME}  MA{MA_TREND}  RSI{rsi:.0f}  BB{bb_pos:.0f}%  |  {state}',
                      fontsize=10, loc='left', color=color)
        ax1.tick_params(labelsize=7); ax1.grid(True, alpha=0.12)

        ax2 = fig.add_subplot(gs[2]); ax2.set_facecolor('#FFFFFF')
        lc = '#CC0000' if navs[-1] >= 1 else '#008800'
        ax2.fill_between(range(len(navs)), 1, navs, alpha=0.08, color=lc)
        ax2.plot(range(len(navs)), navs, color=lc, lw=2.0)
        ax2.axhline(y=1, color='#AAA', lw=0.8, ls='--')
        dd_min = ((navs/navs.cummax())-1).min()*100 if len(navs) > 0 else 0
        ax2.set_title(f'120日净值 {(navs[-1]-1)*100:+.1f}%  DD:{dd_min:+.1f}%',
                      fontsize=10, loc='left', color=lc)
        ax2.tick_params(labelsize=7); ax2.grid(True, alpha=0.12)

        buf = io.BytesIO()
        fig.savefig(buf, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
        plt.close(fig)
        img_bytes = buf.getvalue()

        token = os.environ.get('GH_TOKEN', '')
        if not token:
            for p in ['../github_token.txt', 'github_token.txt', 'd:/策略/github_token.txt']:
                try: token = open(p).read().strip(); break
                except: pass
        chart_url = ''
        if token:
            ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            ctx2 = ssl._create_unverified_context()
            h2 = {'Authorization': 'Bearer ' + token, 'User-Agent': 'YH_ultra',
                  'Content-Type': 'application/json'}
            api2 = f'https://api.github.com/repos/{REPO}/contents/YH_ultra/chart_{ts}.png'
            body2 = json.dumps({'message': 'YH_ultra chart',
                                'content': base64.b64encode(img_bytes).decode('ascii'),
                                'branch': 'main'}).encode()
            ur.urlopen(ur.Request(api2, data=body2, headers=h2, method='PUT'),
                       timeout=15, context=ctx2)
            chart_url = f'https://cdn.jsdelivr.net/gh/{REPO}@main/YH_ultra/chart_{ts}.png'

        body = f'{action}{warn}\n{detail}\n动量: {rank_str}\n{chg_str}'
        send_bark(f'YH_ultra {action}{warn}', body, chart_url)
        with open('_preview.png', 'wb') as f: f.write(img_bytes)
        print(f"完成! 图表: _preview.png")

    except Exception as e:
        print(f"失败: {e}"); import traceback; traceback.print_exc()
        send_bark('YH_ultra信号失败', str(e)[:200])

if __name__ == '__main__':
    main()