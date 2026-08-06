# -*- coding: utf-8 -*-
"""
个股交易策略 v9: YH02趋势延迟卖出
标的: 山东高速(600350) + 渝农商行(601077)

买入: RSI<42 或 距BB下轨<25%
卖出: YH02扩张延迟逻辑 + 移动止损5% + 硬止损10%
      BB扩张(趋势加速)→延迟卖, BB收缩→正常卖
"""
import sys, io, os, json, ssl, base64, warnings
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

STOCKS = {'山东高速': 'sh600350', '渝农商行': 'sh601077', '皖通高速': 'sh600012', '江苏银行': 'sh600919'}
INIT = 1_000_000; COMM = 0.0003; SLIP = 0.0001; MAX_POS = 0.25
BARK_KEY = 'eoq8G58fJtDDFxHjhNueGH'
REPO = 'sunran1996/my_candle'

# 买入 — 正常模式
RSI_BUY = 42; BB_BUY = 0.25
# 买入 — 严格模式(上笔亏损后, 阈值收紧但不要求同时满足)
RSI_STRICT = 35; BB_STRICT = 0.18

# 卖出 — YH02风格
RSI_H = 70; ERS = 75      # RSI高位 / 扩张模式RSI阈值(提高)
TRAIL_STOP = 0.08          # 移动止损8%
HARD_STOP  = 0.10          # 硬止损10%

SCRIPT = os.path.dirname(os.path.abspath(__file__))

# =====================================================
def fetch():
    dfs = {}
    for name, sym in STOCKS.items():
        df = ak.stock_zh_a_daily(symbol=sym, adjust='qfq')
        df['date'] = pd.to_datetime(df['date'])
        dfs[name] = df[['date','open','high','low','close','volume']].sort_values('date').reset_index(drop=True)
    return dfs

def add_indicators(df):
    df = df.copy(); c = df['close']
    df['ma20'] = c.rolling(20).mean(); df['ma60'] = c.rolling(60).mean()
    df['bb_ma'] = c.rolling(20).mean(); df['bb_std'] = c.rolling(20).std()
    df['bb_up'] = df['bb_ma'] + 2*df['bb_std']; df['bb_lo'] = df['bb_ma'] - 2*df['bb_std']
    df['bb_bw'] = (df['bb_up'] - df['bb_lo']) / df['bb_ma']  # 带宽
    d = c.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    df['rsi'] = 100 - 100/(1 + g.ewm(alpha=1/14,adjust=False).mean() /
                l.ewm(alpha=1/14,adjust=False).mean().replace(0, np.nan))
    return df

def check_buy(row, strict=False):
    if pd.isna(row['bb_lo']) or pd.isna(row['rsi']): return False, 0
    rsi = row['rsi']; c = row['close']; lo = row['bb_lo']; up = row['bb_up']
    if up <= lo: return False, 0
    dist = (c - lo) / (up - lo)
    rsi_th = RSI_STRICT if strict else RSI_BUY
    bb_th  = BB_STRICT if strict else BB_BUY
    rsi_ok = rsi <= rsi_th
    bb_ok = dist <= bb_th
    if strict:
        # 严格模式: 阈值收紧, 至少满足一个
        score = (1 if rsi_ok else 0) + (1 if bb_ok else 0)
        if rsi <= 25: score += 1
        return score >= 1, score
    else:
        score = (1 if rsi_ok else 0) + (1 if bb_ok else 0)
        if rsi <= 30: score += 1
        return score >= 1, score

# =====================================================
def run_backtest(start_str=None):
    start = pd.Timestamp(start_str) if start_str else None
    print("获取数据...")
    raw = fetch(); dfs = {n: add_indicators(d) for n, d in raw.items()}
    dates = sorted(set.intersection(*[set(d['date']) for d in dfs.values()]))
    if start: dates = [d for d in dates if d >= start]
    if len(dates) < 60: print("数据不足"); return

    cash = INIT
    shares = {n: 0.0 for n in STOCKS}
    entry = {n: 0.0 for n in STOCKS}
    high = {n: 0.0 for n in STOCKS}
    pbw = {n: None for n in STOCKS}  # 上一期BB带宽
    loss = {n: False for n in STOCKS}  # 上笔是否亏损→触发缩放模式
    scale_step = {n: 0 for n in STOCKS}  # 缩放模式: 0=正常, 1/2/3=第几次买入
    scale_entry = {n: 0.0 for n in STOCKS}  # 缩放模式累计成本(算加权均价)
    scale_high = {n: 0.0 for n in STOCKS}  # 缩放模式最高价
    navs = []; trades = []
    SCALE_PCTS = [0.30, 0.30, 0.40]  # 分三次: 30%+30%+40%=100%仓位

    for date in dates:
        px = {n: raw[n][raw[n]['date']==date]['close'].iloc[0]
              for n in STOCKS if len(raw[n][raw[n]['date']==date])>0}

        # ── 卖出 ──
        for n in STOCKS:
            if shares[n] <= 0: continue
            cp = px.get(n, 0); r = dfs[n][dfs[n]['date']==date]
            if cp <= 0 or len(r)==0: continue
            if cp > high[n]: high[n] = cp
            pnl = cp / entry[n] - 1; dd = cp / high[n] - 1

            # YH02风格卖出: BB扩张延迟, BB收缩正常
            bw = r.iloc[0]['bb_bw'] if not pd.isna(r.iloc[0].get('bb_bw')) else None
            expanding = (pbw[n] is not None and bw is not None and bw > pbw[n])
            at_bb_up = cp >= r.iloc[0]['bb_up'] if not pd.isna(r.iloc[0].get('bb_up')) else False
            rsi_v = r.iloc[0]['rsi']

            yh_sell = False; yh_why = ''
            if expanding:
                # 扩张=趋势加速, 延迟卖出: 必须同时满足BB上轨+RSI>=65
                if at_bb_up and not pd.isna(rsi_v) and rsi_v >= ERS:
                    yh_sell = True; yh_why = f'扩张触顶 RSI{rsi_v:.0f}'
            else:
                # 收缩=正常卖出: BB上轨 或 RSI>=70
                if at_bb_up: yh_sell = True; yh_why = 'BB上轨'
                elif not pd.isna(rsi_v) and rsi_v >= RSI_H:
                    yh_sell = True; yh_why = f'RSI超买{rsi_v:.0f}'

            do = False; why = ''
            if pnl <= -HARD_STOP: do = True; why = f'硬止损{pnl*100:+.1f}%'
            elif dd <= -TRAIL_STOP: do = True; why = f'移动止损{dd*100:+.1f}%'
            elif yh_sell: do = True; why = f'{yh_why} 盈{pnl*100:+.1f}%'

            if bw is not None: pbw[n] = bw  # 更新带宽记忆

            if do:
                cash += shares[n] * cp * (1-COMM-SLIP)
                trades.append({'date':date,'name':n,'dir':'SELL','price':cp,
                               'pnl':pnl*100,'reason':why})
                if pnl >= 0:
                    # 盈利→重置缩放模式
                    scale_step[n] = 0; loss[n] = False
                else:
                    # 亏损→触发缩放模式
                    loss[n] = True; scale_step[n] = 0
                shares[n] = 0; entry[n] = 0; high[n] = 0

        nav = cash + sum(shares[n]*px.get(n,0) for n in STOCKS)

        # ── 买入 ──
        for n in STOCKS:
            cp = px.get(n, 0); r = dfs[n][dfs[n]['date']==date]
            if cp <= 0 or len(r)==0: continue

            in_scale = loss[n] and scale_step[n] < 3  # 缩放模式中
            if shares[n] > 0 and not in_scale: continue  # 正常模式满仓, 不补
            # 缩放模式中已有仓位也可以继续补

            ok, sc = check_buy(r.iloc[0], strict=loss[n])
            if not ok: continue

            if in_scale:
                # 缩放模式: 按步买入
                step = scale_step[n] + 1  # 第1/2/3次
                pct = SCALE_PCTS[step - 1]
                val = min(cash, nav * MAX_POS * pct)
                if val > 5000:
                    qty = val/cp*(1-COMM-SLIP)
                    # 更新加权均价
                    old_cost = entry[n] * shares[n] if shares[n] > 0 else 0
                    shares[n] += qty; cash -= val
                    entry[n] = (old_cost + cp * qty) / shares[n] if shares[n] > 0 else cp
                    high[n] = max(high[n], cp) if shares[n] - qty > 0 else cp
                    scale_step[n] = step
                    trades.append({'date':date,'name':n,'dir':'BUY',
                                   'price':cp,'pnl':0,
                                   'reason':f'严补{step}/3 RSI{r.iloc[0]["rsi"]:.0f}'})
                    if scale_step[n] >= 3:
                        scale_step[n] = 3  # 满仓, 不再补
            else:
                # 正常模式: 直接满仓
                if shares[n] > 0: continue
                val = min(cash, nav*MAX_POS)
                if val > 5000:
                    qty = val/cp*(1-COMM-SLIP)
                    shares[n] = qty; cash -= val
                    entry[n] = cp; high[n] = cp
                    trades.append({'date':date,'name':n,'dir':'BUY','price':cp,
                                   'pnl':0,'reason':f'RSI{r.iloc[0]["rsi"]:.0f} 评{sc}'})

        nav = cash + sum(shares[n]*px.get(n,0) for n in STOCKS)
        holding = [n for n in STOCKS if shares[n]>0]
        navs.append({'date':date,'nav':nav,'hold':','.join(holding) if holding else 'CASH'})

    # ── 统计 ──
    ndf = pd.DataFrame(navs); final = ndf['nav'].iloc[-1]
    ret = (final/INIT-1)*100; ann = ((1+ret/100)**(252/len(ndf))-1)*100
    dr = ndf['nav'].pct_change().dropna(); vol = dr.std()*np.sqrt(252)*100
    sr = (ann-2)/vol if vol>0 else 0
    mdd = ((ndf['nav']-ndf['nav'].cummax())/ndf['nav'].cummax()).min()*100

    td = pd.DataFrame(trades)
    buys = td[td['dir']=='BUY']; sells = td[td['dir']=='SELL']
    wr = (sells['pnl']>0).sum()/len(sells)*100 if len(sells)>0 else 0
    aw = sells[sells['pnl']>0]['pnl'].mean() if len(sells[sells['pnl']>0])>0 else 0
    al = sells[sells['pnl']<0]['pnl'].mean() if len(sells[sells['pnl']<0])>0 else 0
    cpct = (ndf['hold']=='CASH').sum()/len(ndf)*100

    print(f"\n{'='*60}")
    print(f"  策略v9: YH02趋势延迟卖出")
    print(f"  {'─'*40}")
    print(f"  累计: {ret:+.1f}%  年化: {ann:+.1f}%  夏普: {sr:.2f}  回撤: {mdd:+.1f}%")
    print(f"  交易: BUY{len(buys)} SELL{len(sells)}  胜率{wr:.0f}%  均盈{aw:+.1f}%  均亏{al:+.1f}%  空仓{cpct:.0f}%")

    ndf['year'] = ndf['date'].dt.year
    print(f"\n  {'年份':<6} {'收益':>8} {'MaxDD':>8}")
    for yr, grp in ndf.groupby('year'):
        if len(grp)<10: continue
        yr_ret = (grp['nav'].iloc[-1]/grp['nav'].iloc[0]-1)*100
        yr_mdd = ((grp['nav']-grp['nav'].cummax())/grp['nav'].cummax()).min()*100
        print(f"  {yr:<6} {yr_ret:>+7.1f}% {yr_mdd:>+7.1f}%")

    # ── 图表 ──
    RED = '#CC0000'; GREEN = '#008800'; PURPLE = '#9B59B6'; BLUE = '#3498DB'
    ORANGE = '#E67E22'; GRAY = '#888888'; CYAN = '#2ECC71'; DBLUE = '#2980B9'
    colors4 = [RED, ORANGE, CYAN, DBLUE]
    plot_start = ndf['date'].iloc[-1] - pd.DateOffset(years=3)

    fig = plt.figure(figsize=(20, 22), facecolor='white')
    gs = fig.add_gridspec(6, 1, height_ratios=[1.2, 2.2, 2.2, 2.2, 2.2, 1.0],
                          hspace=0.22, top=0.97, bottom=0.03, left=0.05, right=0.97)

    nav_s = ndf['nav']/INIT; dd_s = (nav_s-nav_s.cummax())/nav_s.cummax()

    # P1: 净值+回撤
    ax = fig.add_subplot(gs[0])
    ax.plot(ndf['date'], nav_s, color=RED, lw=2.0, label='策略净值', zorder=3)
    ax.fill_between(ndf['date'], 1, nav_s, alpha=0.06, color=RED)
    ax.axhline(y=1.0, color=GRAY, lw=0.8, ls='--')
    in_dd, dd_srt = False, None
    for i, (d, dv) in enumerate(zip(ndf['date'], dd_s)):
        if dv<-0.05 and not in_dd: dd_srt=d; in_dd=True
        elif dv>-0.02 and in_dd and dd_srt:
            ax.axvspan(dd_srt, d, alpha=0.06, color='red'); in_dd=False; dd_srt=None
    if in_dd and dd_srt: ax.axvspan(dd_srt, ndf['date'].iloc[-1], alpha=0.06, color='red')
    for _, t in td.iterrows():
        m = ndf['date']==t['date']
        if not m.any(): continue
        yv = nav_s[m].iloc[0]
        ci = list(STOCKS.keys()).index(t['name']) if t['name'] in STOCKS else 0
        c = colors4[ci % 4]
        if t['dir']=='BUY': ax.scatter(t['date'],yv,color=c,s=50,marker='^',zorder=6,edgecolors='white',lw=1)
        else: ax.scatter(t['date'],yv,color=GREEN,s=50,marker='v',zorder=6,edgecolors='white',lw=1)
    ax.set_ylabel('净值',fontsize=10); ax.legend(fontsize=8,loc='upper left'); ax.grid(True,alpha=0.12)
    ax.set_title(f'v9 YH02趋势延迟卖出 | +{ret:.1f}% | 年化{ann:.1f}% | 夏普{sr:.2f} | 回撤{mdd:.1f}% | {len(td)}笔 | 胜率{wr:.0f}%',
                 fontsize=13, fontweight='bold')

    cn_c = mpf.make_marketcolors(up=RED, down=GREEN, edge='inherit', wick='inherit', volume='inherit')
    cn_s = mpf.make_mpf_style(marketcolors=cn_c, gridstyle='',
                               rc={'font.sans-serif':[CN],'axes.unicode_minus':False})

    for idx, (name, color) in enumerate(zip(STOCKS.keys(), colors4)):
        ax = fig.add_subplot(gs[idx+1])
        ohlc = raw[name][raw[name]['date']>=plot_start].copy()
        if len(ohlc)<20: ohlc = raw[name].tail(500).copy()
        ohlc = ohlc.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
        ohlc = ohlc.set_index('date')[['Open','High','Low','Close','Volume']]
        mpf.plot(ohlc, type='candle', ax=ax, volume=False, style=cn_s)
        bb = dfs[name][dfs[name]['date']>=plot_start]
        if len(bb)<20: bb = dfs[name].tail(500)
        x = range(len(ohlc))
        ax.plot(x, bb['bb_up'].values[-len(ohlc):], color=PURPLE, lw=0.6, ls='--', alpha=0.5)
        ax.plot(x, bb['bb_lo'].values[-len(ohlc):], color=PURPLE, lw=0.6, ls='--', alpha=0.5)
        ax.plot(x, bb['bb_ma'].values[-len(ohlc):], color=GRAY, lw=0.7, ls='--', alpha=0.4)
        ax.plot(x, bb['ma60'].values[-len(ohlc):], color=BLUE, lw=1.0, alpha=0.6, label='MA60')
        ohlc_dates = ohlc.index
        for _, t in td.iterrows():
            if t['name']!=name: continue
            td_d = pd.Timestamp(t['date'])
            for j, od in enumerate(ohlc_dates):
                if pd.Timestamp(od).date()==td_d.date():
                    if t['dir']=='BUY':
                        ax.scatter(j, ohlc['Low'].iloc[j], color='#FF0000', s=120, marker='^',
                                  zorder=10, edgecolors='white', lw=2.0)
                        ax.annotate(f"买\n{t['price']:.2f}", (j, ohlc['Low'].iloc[j]),
                                   textcoords='offset points', xytext=(0,-25),
                                   fontsize=7, color='#CC0000', fontweight='bold', ha='center')
                    else:
                        ax.scatter(j, ohlc['High'].iloc[j], color='#008800', s=120, marker='v',
                                  zorder=10, edgecolors='white', lw=2.0)
                        ax.annotate(f"卖\n{t['pnl']:+.1f}%", (j, ohlc['High'].iloc[j]),
                                   textcoords='offset points', xytext=(0,15),
                                   fontsize=7, color='#008800', fontweight='bold', ha='center')
                    break
        lp = ohlc['Close'].iloc[-1]; lr = dfs[name]['rsi'].iloc[-1]
        ax.set_title(f'{name}  {lp:.2f}  {ohlc_dates[-1].strftime("%Y-%m-%d")}  RSI{lr:.0f}',
                    fontsize=12, fontweight='bold', color=color)
        ax.legend(fontsize=7, loc='upper left'); ax.tick_params(labelsize=7); ax.grid(True, alpha=0.1)

    ax = fig.add_subplot(gs[5])
    ax.fill_between(ndf['date'], 0, dd_s*100, color='#E74C3C', alpha=0.35, step='post')
    ax.plot(ndf['date'], dd_s*100, color='#C0392B', lw=0.8)
    ax.axhline(y=-5, color=GRAY, lw=0.5, ls='--', alpha=0.5)
    ax.axhline(y=-10, color=GRAY, lw=0.5, ls='--', alpha=0.5)
    ax.set_ylabel('回撤 %', fontsize=10); ax.set_ylim(None, 2)
    ax.tick_params(labelsize=8); ax.grid(True, alpha=0.12)
    ax.set_title('组合回撤', fontsize=11, fontweight='bold')

    plt.savefig(os.path.join(SCRIPT,'backtest_chart.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\n  图表: {SCRIPT}\\backtest_chart.png')

    # 交易明细
    if len(td)>0:
        print(f"\n  ── 交易明细 ──")
        print(f"  {'日期':<12} {'标的':<8} {'操作':<5} {'价格':>7} {'盈亏':>7}  {'说明'}")
        print(f"  {'─'*60}")
        for _, t in td.iterrows():
            d = t['date'].strftime('%Y-%m-%d')
            pnl_s = f"{t['pnl']:+.1f}%" if t['dir']=='SELL' else '—'
            print(f"  {d:<12} {t['name']:<8} {t['dir']:<5} {t['price']:>7.2f} {pnl_s:>7}  {t['reason']}")

    return ndf, td

def send_bark(title, body, url=''):
    if not BARK_KEY: return
    try:
        data = json.dumps({'title':title,'body':body,'url':url}).encode()
        ur.urlopen(ur.Request(f'https://api.day.app/{BARK_KEY}', data=data,
                   headers={'Content-Type':'application/json'}), timeout=10)
    except: pass

def upload_chart(token, img_bytes):
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    fn = f'chart_{ts}.png'
    ctx = ssl._create_unverified_context()
    h = {'Authorization':'Bearer '+token, 'User-Agent':'YH_ultra'}
    api = f'https://api.github.com/repos/{REPO}/contents/YH_ultra/{fn}'
    try:
        r = json.loads(ur.urlopen(ur.Request(api, headers=h), timeout=10, context=ctx).read())
        sha = r.get('sha')
    except: sha = None
    body = json.dumps({'message':'YH_ultra chart','content':base64.b64encode(img_bytes).decode('ascii'),
                       'branch':'main', **({'sha':sha} if sha else {})}).encode()
    ur.urlopen(ur.Request(api, data=body, headers={**h, 'Content-Type':'application/json'}, method='PUT'),
               timeout=15, context=ctx)
    return f'https://cdn.jsdelivr.net/gh/{REPO}@main/YH_ultra/{fn}'

def live_signal():
    print("获取数据...")
    raw = fetch(); dfs = {n: add_indicators(d) for n, d in raw.items()}

    # 实时行情
    try:
        spot = ak.stock_zh_a_spot_em()
        code_map = {'600350':'山东高速','601077':'渝农商行','600012':'皖通高速','600919':'江苏银行'}
        for code, name in code_map.items():
            s = spot[spot['代码']==code]
            if len(s)>0:
                rt = float(s['最新价'].iloc[0])
                raw[name].loc[raw[name].index[-1],'close'] = rt
                raw[name].loc[raw[name].index[-1],'date'] = pd.Timestamp.now()
        dfs = {n: add_indicators(d) for n, d in raw.items()}
        print("  实时行情已更新")
    except: print("  实时行情失败,用日线收盘价")

    lines = []
    buy_list = []
    for name in STOCKS:
        row = dfs[name].iloc[-1]
        close = row['close']; rsi = row['rsi']
        bb_up = row['bb_up']; bb_lo = row['bb_lo']
        bb_range = bb_up - bb_lo
        bb_pos = (close-bb_lo)/(bb_up-bb_lo)*100 if bb_range>0 else 50
        buy_ok, sc = check_buy(row)
        if buy_ok: buy_list.append((name, sc))
        sig = f'★ 买入(评{sc})' if buy_ok else '—'
        lines.append(f'{name} {close:.2f} RSI{rsi:.0f} BB{bb_pos:.0f}% {sig}')

    print(f"\n{'='*50}")
    print(f"  {' '.join(lines)}")
    print(f"{'='*50}")

    # 简版K线图
    fig, axes = plt.subplots(len(STOCKS), 1, figsize=(8, 2.5*len(STOCKS)), facecolor='white')
    if len(STOCKS)==1: axes = [axes]
    cn_c = mpf.make_marketcolors(up='#CC0000', down='#008800', edge='inherit', wick='inherit', volume='inherit')
    cn_s = mpf.make_mpf_style(marketcolors=cn_c, gridstyle='',
                               rc={'font.sans-serif':[CN],'axes.unicode_minus':False})
    for idx, name in enumerate(STOCKS):
        ax = axes[idx]
        ohlc = raw[name].tail(90).copy()
        ohlc = ohlc.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
        ohlc = ohlc.set_index('date')[['Open','High','Low','Close','Volume']]
        mpf.plot(ohlc, type='candle', ax=ax, volume=False, style=cn_s)
        bb = dfs[name].tail(90)
        x = range(len(ohlc))
        ax.plot(x, bb['bb_up'].values[-len(ohlc):], color='#9B59B6', lw=0.5, ls='--', alpha=0.5)
        ax.plot(x, bb['bb_lo'].values[-len(ohlc):], color='#9B59B6', lw=0.5, ls='--', alpha=0.5)
        row = dfs[name].iloc[-1]; px = row['close']; rsi = row['rsi']
        chg = (raw[name]['close'].iloc[-1]/raw[name]['close'].iloc[-2]-1)*100 if len(raw[name])>1 else 0
        ax.set_title(f'{name} {px:.2f} {chg:+.2f}% RSI{rsi:.0f}', fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.1)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    img_bytes = buf.getvalue()

    # 上传GitHub
    token = os.environ.get('GH_TOKEN','')
    if not token:
        for p in ['../github_token.txt','github_token.txt','d:/策略/github_token.txt']:
            try: token = open(p).read().strip(); break
            except: pass
    chart_url = ''
    if token: chart_url = upload_chart(token, img_bytes)

    # 推送
    title = '个股 ' + ' '.join(f'{n}{"★" if any(n==b for b,_ in buy_list) else ""}' for n in STOCKS)
    body = '\n'.join(lines)
    send_bark(title, body, chart_url)
    print("已推送")

    with open('_preview.png','wb') as f: f.write(img_bytes)
    print(f"完成! _preview.png")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--from', dest='fr', type=str, default=None)
    p.add_argument('--live', action='store_true', default=False)
    a = p.parse_args()
    if a.live: live_signal()
    else: run_backtest(a.fr or '2016-01-01')

if __name__ == '__main__':
    main()
