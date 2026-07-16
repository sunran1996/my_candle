# -*- coding: utf-8 -*-
"""
YH01 红利低波 BB+RSI 策略
BB(52,2.0)+RSI(20,38,68) OR/OR + DCA 2w + 10w备用金(回撤>8%)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import akshare as ak, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json, os, warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

INITIAL_CAPITAL = 1_000_000; RESERVE = 100_000; COMMISSION = 0.0001; SLIPPAGE = 0.0001
ETF_SYMBOL = 'sh512890'; ETF_NAME = '红利低波'
BB_PERIOD = 52; BB_STD = 2.0
RSI_PERIOD = 20; RSI_OVERSOLD = 38; RSI_OVERBOUGHT = 68
EXPAND_THRESHOLD = 1.01   # BB宽度增长>1%视为扩张
EXPAND_SELL_AND = True     # 扩张时卖点需AND双确认
EXPAND_RSI_BOOST = 4       # 扩张时RSI卖点提高4点
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSITION_FILE = os.path.join(SCRIPT_DIR, 'positions.json')

def compute_indicators(df):
    ret = df['close'].pct_change().fillna(0); ret[abs(ret) > 0.1] = 0
    df['adj_close'] = (1 + ret).cumprod()
    df['ma'] = df['adj_close'].rolling(BB_PERIOD).mean()
    df['std'] = df['adj_close'].rolling(BB_PERIOD).std()
    df['upper'] = df['ma'] + BB_STD * df['std']
    df['lower'] = df['ma'] - BB_STD * df['std']
    delta = df['adj_close'].diff(); gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
    df['rsi'] = 100 - 100 / (1 + gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean() /
                             loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean().replace(0, np.nan))

def run_backtest(start_date_str):
    start_date = pd.Timestamp(start_date_str)
    df = ak.fund_etf_hist_sina(symbol=ETF_SYMBOL)
    df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date')
    compute_indicators(df)
    df = df[df['date'] >= start_date].reset_index(drop=True)

    cash = 0.0; total_invested = INITIAL_CAPITAL; reserve_pool = RESERVE; reserve_active = False; reserve_shares = 0.0
    shares = INITIAL_CAPITAL / df['close'].iloc[0] * (1 - COMMISSION - SLIPPAGE)
    peak_nav = INITIAL_CAPITAL; nav = INITIAL_CAPITAL
    trades = []; missed = []; nav_log = []; prev_price = None; last_month = None; dca = 20000
    signal_count = 0; prev_bb_width = None

    for i, row in df.iterrows():
        price = row['close']; d = row['date']; adj = row['adj_close']
        if prev_price and prev_price > 0 and abs(price/prev_price - 1) > 0.1: shares *= (prev_price / price)
        prev_price = price
        ym = (d.year, d.month)
        if last_month and ym != last_month: cash += dca; total_invested += dca
        last_month = ym
        lower = row['lower']; upper = row['upper']; rsi = row['rsi']
        if np.isnan(lower) or np.isnan(rsi): continue
        nav = cash + (shares + reserve_shares) * price
        if nav > peak_nav: peak_nav = nav
        dd = (nav - peak_nav) / peak_nav if peak_nav > 0 else 0
        if dd < -0.08 and not reserve_active and reserve_pool > 0:
            cash += reserve_pool; reserve_pool = 0; reserve_active = True

        bb_buy = (adj <= lower); bb_sell = (adj >= upper)
        rsi_buy = (rsi <= RSI_OVERSOLD)
        # 动态卖点: BB扩张时卖点提高(BB开口=趋势启动, 多拿)
        bb_width = (upper - lower) / row['ma'] if row['ma'] > 0 else 0.1
        expanding = (prev_bb_width is not None and bb_width > prev_bb_width * EXPAND_THRESHOLD)
        rsi_sell_threshold = RSI_OVERBOUGHT + EXPAND_RSI_BOOST if expanding else RSI_OVERBOUGHT
        rsi_sell = (rsi >= rsi_sell_threshold)
        buy_sig = (bb_buy or rsi_buy)
        if expanding and EXPAND_SELL_AND: sell_sig = (bb_sell and rsi_sell)
        else:                             sell_sig = (bb_sell or rsi_sell)
        prev_bb_width = bb_width

        action = None; buy_type = ''
        if buy_sig or sell_sig: signal_count += 1
        if signal_count < 2:  # 跳过第一个信号
            nav_log.append({'date': d, 'nav': nav}); continue
        if buy_sig:
            if cash > nav * 0.01:
                amt = cash
                buy_shares = amt / price * (1 - COMMISSION - SLIPPAGE)
                if reserve_active:
                    reserve_shares += buy_shares * (RESERVE / (INITIAL_CAPITAL + RESERVE))
                    shares += buy_shares - buy_shares * (RESERVE / (INITIAL_CAPITAL + RESERVE))
                    buy_type = 'RESERVE'
                elif cash < nav * 0.1:
                    shares += buy_shares; buy_type = 'DCA'
                else:
                    shares += buy_shares; buy_type = 'BUY'
                cash = 0; action = 'BUY'
            else:
                missed.append({'date': d, 'type': 'buy'})
        elif sell_sig:
            if shares > 0 or reserve_shares > 0:
                cash += (shares + reserve_shares) * price * (1 - COMMISSION - SLIPPAGE)
                if reserve_active:
                    reserve_pool = reserve_shares * price * (1 - COMMISSION - SLIPPAGE)
                    cash -= reserve_pool; reserve_shares = 0; reserve_active = False
                shares = 0; action = 'SELL'
            else:
                missed.append({'date': d, 'type': 'sell'})
        if action:
            nav = cash + (shares + reserve_shares) * price
            trades.append({'date': d.strftime('%Y-%m-%d'), 'dir': action, 'price': round(price, 4),
                          'nav': round(nav, 0), 'ret': round((nav/total_invested-1)*100, 2),
                          'type': buy_type if action == 'BUY' else ''})
        nav_log.append({'date': d, 'nav': nav})

    last_price = df['close'].iloc[-1]; final_nav = cash + shares * last_price
    total_ret = (final_nav / total_invested - 1) * 100
    n_days = len(nav_log)
    ann_ret = ((1 + total_ret/100) ** (252 / n_days) - 1) * 100 if n_days > 0 else 0
    nav_df = pd.DataFrame(nav_log)
    daily_rets = nav_df['nav'].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) * 100
    sharpe = (ann_ret/100 - 0.02) / (ann_vol/100) if ann_vol > 0 else 0
    cum = nav_df['nav'] / INITIAL_CAPITAL
    mdd = ((cum - cum.cummax()) / cum.cummax()).min() * 100

    trade_df = pd.DataFrame(trades)
    print(f"\n  {ETF_NAME} BB({BB_PERIOD},{BB_STD})+RSI({RSI_PERIOD},{RSI_OVERSOLD},{RSI_OVERBOUGHT})")
    print(f"  Return: {total_ret:+.2f}%  Annual: {ann_ret:+.2f}%  Sharpe: {sharpe:.3f}")
    print(f"  MaxDD: {mdd:+.2f}%  Trades: {len(trade_df)}  Final: {final_nav:,.0f}")

    with open(os.path.join(SCRIPT_DIR, 'trades.txt'), 'w', encoding='utf-8') as tf:
        tf.write(f"{'date':<12} {'dir':<6} {'price':<10} {'nav':<12}\n")
        for t in trades:
            tf.write(f"{t['date']:<12} {t['dir']:<6} {t['price']:<10} {t['nav']:<12,.0f}\n")
        tf.write(f"\n{len(trade_df)} trades\n")

    if len(nav_df) > 1:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]}, facecolor='white')
        fig.subplots_adjust(hspace=0.25, top=0.95)

        # P1: 价格+BB+买卖点
        ax = axes[0]
        ax.set_title(f'{ETF_NAME}  BB+RSI  |  {total_ret:+.1f}%  |  {ann_ret:+.1f}%pa  |  夏普 {sharpe:.3f}  |  回撤 {mdd:+.1f}%  |  {len(trade_df)}笔',
                     fontsize=13, fontweight='bold', color='black', loc='left', pad=8)
        ax.plot(df['date'], df['adj_close'], color='black', lw=0.8)
        ax.fill_between(df['date'], df['upper'], df['lower'], alpha=0.05, color='#3498db')
        ax.plot(df['date'], df['upper'], color='#e74c3c', lw=0.5, ls='--', alpha=0.6)
        ax.plot(df['date'], df['ma'], color='#bdc3c7', lw=0.5, alpha=0.4)
        ax.plot(df['date'], df['lower'], color='#27ae60', lw=0.5, ls='--', alpha=0.6)
        pm = dict(zip(df['date'].dt.strftime('%Y-%m-%d'), df['adj_close']))
        if len(trade_df) > 0:
            sd = trade_df[trade_df['dir']=='SELL'].copy(); sd['adj'] = sd['date'].map(pm)
            if len(sd) > 0: ax.scatter(pd.to_datetime(sd['date']), sd['adj'], color='#27ae60', s=55, marker='v', zorder=5, edgecolors='white', linewidth=1.2, label='卖出')
            for c, t, lbl in [('#c0392b','BUY','买入'),('#e67e22','DCA','定投'),('#8e44ad','RESERVE','备用')]:
                bd = trade_df[(trade_df['dir']=='BUY')&(trade_df['type']==t)]
                if len(bd) > 0: bd = bd.copy(); bd['adj'] = bd['date'].map(pm); ax.scatter(pd.to_datetime(bd['date']), bd['adj'], color=c, s=55, marker='^', zorder=5, edgecolors='white', linewidth=1.2, label=lbl)
        ax.legend(fontsize=8, loc='upper left', ncol=3, framealpha=0.9)
        ax.tick_params(labelsize=9, colors='black'); ax.grid(True, alpha=0.2, color='#ecf0f1')

        # P2: 净值对比
        ax = axes[1]
        ax.plot(nav_df['date'], nav_df['nav']/INITIAL_CAPITAL, color='#c0392b', lw=1.8, label='策略')
        ax.plot(df['date'], df['adj_close']/df['adj_close'].iloc[0], color='#bdc3c7', lw=1, ls='--', label='买入持有')
        ax.axhline(1.0, color='#ecf0f1', lw=0.8, ls='--')
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
        ax.tick_params(labelsize=9, colors='black'); ax.grid(True, alpha=0.2, color='#ecf0f1')

        plt.savefig(os.path.join(SCRIPT_DIR, 'backtest_chart.png'), dpi=150, bbox_inches='tight', facecolor='white'); plt.close()

def main():
    import argparse, urllib.request, json
    parser = argparse.ArgumentParser()
    parser.add_argument('--from', dest='from_date', type=str, default=None)
    parser.add_argument('--push', type=int, default=0, metavar='N', help='回测N天并推送图表')
    args = parser.parse_args()
    if args.from_date: run_backtest(args.from_date); return
    if args.push:
        start = (datetime.now() - timedelta(days=args.push)).strftime('%Y-%m-%d')
        run_backtest(start)
        # 推图表
        try:
            import urllib.request as ur, ssl, base64
            chart_file = os.path.join(SCRIPT_DIR, 'backtest_chart.png')
            if os.path.exists(chart_file):
                token = os.environ.get('GITHUB_TOKEN', '') or open(os.path.join(SCRIPT_DIR, '..', 'github_token.txt')).read().strip()
                with open(chart_file, 'rb') as f: img = base64.b64encode(f.read()).decode('ascii')
                ctx = ssl._create_unverified_context()
                api = 'https://api.github.com/repos/sunran1996/my_candle/contents/YH01/backtest_chart.png'
                sha = None
                try:
                    r = json.loads(ur.urlopen(ur.Request(api, headers={'Authorization': 'Bearer '+token, 'User-Agent': 'gh'}), timeout=10, context=ctx).read())
                    sha = r.get('sha')
                except: pass
                body = {'message': 'chart', 'content': img}
                if sha: body['sha'] = sha
                ur.urlopen(ur.Request(api, data=json.dumps(body, ensure_ascii=True).encode('ascii'),
                    headers={'Authorization': 'Bearer '+token, 'User-Agent': 'gh', 'Content-Type': 'application/json'},
                    method='PUT'), timeout=15, context=ctx)
                import time
                chart_url = f'https://cdn.jsdelivr.net/gh/sunran1996/my_candle@main/YH01/backtest_chart.png?t={int(time.time())}'
                payload = json.dumps({'title': f'{ETF_NAME} {args.push}天回顾', 'body': f'{start} -> 今天\n点击查看收益图', 'url': chart_url}).encode()
                ur.urlopen(ur.Request('https://api.day.app/eoq8G58fJtDDFxHjhNueGH', data=payload,
                    headers={'Content-Type': 'application/json'}), timeout=10)
                print("  图表已推送")
        except Exception as e: print(f"  推送失败: {e}")
        return
    df = ak.fund_etf_hist_sina(symbol=ETF_SYMBOL); df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date')
    compute_indicators(df)
    r = df.iloc[-1]; prev_r = df.iloc[-2]
    price = r['close']; rsi = r['rsi']; bb_pos = (r['adj_close']-r['lower'])/(r['upper']-r['lower'])
    bb_w = (r['upper']-r['lower'])/r['ma']; prev_bb_w = (prev_r['upper']-prev_r['lower'])/prev_r['ma']
    expanding = bb_w > prev_bb_w * EXPAND_THRESHOLD
    sig_cn = '买入' if (bb_pos<=0 or rsi<=RSI_OVERSOLD) else ('卖出' if (bb_pos>=1 or rsi>=RSI_OVERBOUGHT) else '持有')
    exp_cn = '扩张' if expanding else '收缩'
    print(f"\n  {ETF_NAME}  {r['date'].strftime('%Y-%m-%d')}  价格 {price:.4f}  RSI {rsi:.1f}  BB {bb_pos*100:.0f}%  {exp_cn}")
    print(f"  信号: {sig_cn}")
    # 记录NAV+推送
    pos = {'cash': INITIAL_CAPITAL, 'shares': 0, 'total_invested': INITIAL_CAPITAL, 'nav_history': []}
    if os.path.exists(POSITION_FILE):
        try:
            with open(POSITION_FILE, 'r', encoding='utf-8') as f: pos = json.load(f)
        except: pass
    nav = pos['cash'] + pos.get('shares', 0) * price
    ret = (nav / pos['total_invested'] - 1) * 100
    # 记录净值历史
    nav_hist = pos.get('nav_history', [])
    today_str = r['date'].strftime('%Y-%m-%d')
    if not nav_hist or nav_hist[-1]['date'] != today_str:
        nav_hist.append({'date': today_str, 'nav': round(nav, 0), 'ret': round(ret, 2)})
        if len(nav_hist) > 500: nav_hist = nav_hist[-500:]
        pos['nav_history'] = nav_hist
        with open(POSITION_FILE, 'w', encoding='utf-8') as f: json.dump(pos, f, ensure_ascii=False, indent=2)

    # 实盘收益图
    try:
        dates = [h['date'] for h in nav_hist]
        nvs = [h['nav']/pos['total_invested'] for h in nav_hist]
        # 上证指数对比
        sh_df = ak.stock_zh_index_daily(symbol='sh000001')
        sh_df['date'] = pd.to_datetime(sh_df['date']); sh_df = sh_df.sort_values('date')
        sh_start = sh_df[sh_df['date'] >= dates[0]]
        sh_vals = (sh_start['close'] / sh_start['close'].iloc[0]).tolist() if len(sh_start) > 0 else []

        fig = plt.figure(figsize=(12, 7), facecolor='white')
        gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.05)
        fig.subplots_adjust(left=0.06, right=0.94, top=0.88, bottom=0.12)

        # 左: 净值曲线
        ax = fig.add_subplot(gs[0])
        if len(nvs) >= 2:
            ax.fill_between(range(len(nvs)), 1.0, nvs, alpha=0.08, color='#c0392b')
            ax.plot(range(len(nvs)), nvs, color='#c0392b', lw=1.5, label='策略')
            # 上证指数
            if sh_vals:
                ax.plot(range(len(sh_vals)), sh_vals, color='#7f8c8d', lw=1, ls='--', alpha=0.6, label='上证指数')
            ax.legend(fontsize=8, loc='upper left', framealpha=0.8)
            ax.scatter(len(nvs)-1, nvs[-1], color='#c0392b', s=40, zorder=5)
            ax.scatter(len(nvs)-1, nvs[-1], color='#c0392b', s=120, zorder=4, alpha=0.15)
        else:
            ax.text(0.5, 0.5, f'{nav:,.0f}', ha='center', va='center', fontsize=40,
                    color='#c0392b', fontweight='bold', transform=ax.transAxes)
        ax.axhline(1.0, color='#2c3e50', lw=1.2, ls='-')
        ax.set_xlim(-0.5, max(1, len(nvs)-0.5))
        if len(nvs) >= 2:
            ax.set_xticks([0, len(nvs)-1])
            ax.set_xticklabels([dates[0][5:], dates[-1][5:]], fontsize=9, color='black')
        ax.tick_params(labelsize=8, colors='#2c3e50', width=1.2)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
        for spine in ax.spines.values(): spine.set_visible(True); spine.set_color('#2c3e50'); spine.set_linewidth(1.2)
        ax.grid(True, alpha=0.12, color='#ecf0f1')

        # 右: 信息卡
        ax = fig.add_subplot(gs[1])
        ax.axis('off')
        ax.set_xlim(0, 10); ax.set_ylim(0, 12)
        ax.text(0, 11, ETF_NAME, fontsize=16, fontweight='bold', color='black')
        ax.text(0, 10, '实盘账户', fontsize=10, color='#95a5a6')

        kpis = [
            (0, 8.5, f'{nav:,.0f}', '净值 (元)', '#c0392b'),
            (0, 6.5, f'{ret:+.2f}%', '累计收益', '#c0392b' if ret >= 0 else '#27ae60'),
            (0, 4.5, sig_cn, '今日信号', '#c0392b' if sig_cn == '买入' else ('#27ae60' if sig_cn == '卖出' else 'black')),
        ]
        for x, y, val, lbl, c in kpis:
            ax.text(x, y+0.5, val, fontsize=22, color=c, fontweight='bold')
            ax.text(x, y-0.2, lbl, fontsize=9, color='#95a5a6')

        # 底部信息
        ax.text(0, 1.5, f'价格 {price:.4f}', fontsize=13, color='black')
        ax.text(0, 0.3, f'RSI {rsi:.1f}  |  BB {bb_pos*100:.0f}%  |  {exp_cn}', fontsize=12, color='black')
        # 跑赢/跑输大盘
        if sh_vals and len(nvs) >= 2:
            vs_market = nvs[-1] - sh_vals[-1] if len(sh_vals) > len(nvs)-1 else nvs[-1] - sh_vals[min(len(sh_vals)-1, len(nvs)-1)]
            beat_str = f'跑赢大盘 {vs_market*100:+.1f}%' if vs_market > 0 else f'跑输大盘 {vs_market*100:+.1f}%'
            beat_color = '#c0392b' if vs_market > 0 else '#27ae60'
            ax.text(0, -0.8, beat_str, fontsize=12, color=beat_color, fontweight='bold')

        start_d = pd.to_datetime(dates[0]); end_d = pd.to_datetime(dates[-1])
        date_title = f'{start_d.year}年{start_d.month}月{start_d.day}日 — {end_d.year}年{end_d.month}月{end_d.day}日'
        fig.suptitle(date_title, fontsize=12, color='black', fontweight='bold', x=0.5, y=0.98, ha='center')

        live_chart = os.path.join(SCRIPT_DIR, 'live_chart.png')
        plt.savefig(live_chart, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()

        # 推送
        import urllib.request as ur, ssl, base64, time
        token = open('d:\\策略\\github_token.txt').read().strip()
        with open(live_chart, 'rb') as f: img = base64.b64encode(f.read()).decode('ascii')
        ctx = ssl._create_unverified_context()
        api = 'https://api.github.com/repos/sunran1996/my_candle/contents/YH01/live_chart.png'
        sha = None
        try:
            r2 = json.loads(ur.urlopen(ur.Request(api, headers={'Authorization': 'Bearer '+token, 'User-Agent': 'gh'}), timeout=10, context=ctx).read())
            sha = r2.get('sha')
        except: pass
        body_gh = {'message': 'live', 'content': img}
        if sha: body_gh['sha'] = sha
        ur.urlopen(ur.Request(api, data=json.dumps(body_gh, ensure_ascii=True).encode('ascii'),
            headers={'Authorization': 'Bearer '+token, 'User-Agent': 'gh', 'Content-Type': 'application/json'}, method='PUT'), timeout=15, context=ctx)
        chart_url = f'https://cdn.jsdelivr.net/gh/sunran1996/my_candle@main/YH01/live_chart.png?t={int(time.time())}'
        body = f'价格 {price:.4f}  RSI {rsi:.1f}  BB {bb_pos*100:.0f}%  {exp_cn}\n净值 {nav:,.0f}  收益 {ret:+.1f}%'
        payload = json.dumps({'title': f'{ETF_NAME} {sig_cn}', 'body': body, 'url': chart_url}).encode()
        ur.urlopen(ur.Request('https://api.day.app/eoq8G58fJtDDFxHjhNueGH', data=payload,
            headers={'Content-Type': 'application/json'}), timeout=10)
        print("  实盘图已推送")
    except Exception as e: print(f"  推送失败: {e}")

if __name__ == '__main__': main()
