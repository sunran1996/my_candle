# -*- coding: utf-8 -*-
"""
YH_ultra v6  红利+黄金核心卫星 — 穿越牛熊

核心理念:
  底仓: 黄金70% (长期通胀对冲+危机保护)
  卫星: 红利低波30% (A股防御红利) — MA200趋势过滤, 趋势坏了红利→现金
  月调仓, 简单稳健

用法: python monitor.py                     → 实时信号
      python monitor.py --from 2020-01-01   → 回测
      python monitor.py --from 2020-01-01 --dca 2  → 回测+DCA月投2万
"""
import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import akshare as ak, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

_fonts = [f.name for f in fm.fontManager.ttflist]
CN = 'WenQuanYi Zen Hei' if 'WenQuanYi Zen Hei' in _fonts else ('SimHei' if 'SimHei' in _fonts else 'DejaVu Sans')
plt.rcParams['font.sans-serif'] = [CN]; plt.rcParams['axes.unicode_minus'] = False

# ======================== 参数 ========================
INIT  = 1_000_000; COMM = 0.0003; SLIP = 0.0001; DCA = 0
MA_TREND = 200
W_GOLD   = 0.75      # 黄金底仓
W_MAIN   = 0.25      # 红利卫星(趋势坏了→现金)
SCRIPT = os.path.dirname(os.path.abspath(__file__))

MAIN_SYM='sh512890'; MAIN_NAME='红利低波'
GOLD_SYM='sh518880'; GOLD_NAME='黄金ETF'

def fetch():
    dfs = {}
    for n, s in [(MAIN_NAME, MAIN_SYM), (GOLD_NAME, GOLD_SYM)]:
        df = ak.fund_etf_hist_sina(symbol=s); df['date'] = pd.to_datetime(df['date'])
        dfs[n] = df[['date', 'close']].sort_values('date').reset_index(drop=True)
    return dfs

def add_ma(df):
    df = df.copy()
    df['ma200'] = df['close'].rolling(MA_TREND).mean()
    return df

def run_backtest(start_str):
    start = pd.Timestamp(start_str)
    print("获取数据...")
    raw = fetch()
    for n in raw: raw[n] = add_ma(raw[n])

    dates = sorted(set.intersection(*[set(d['date']) for d in raw.values()]))
    dates = [d for d in dates if d >= start]
    if len(dates) < MA_TREND + 60: return

    cash = 0.0; total_inv = INIT
    shares_main = 0.0; shares_gold = 0.0
    peak = INIT; navs = []

    # 初始建仓
    p0m = raw[MAIN_NAME][raw[MAIN_NAME]['date'] >= start]['close'].iloc[0]
    p0g = raw[GOLD_NAME][raw[GOLD_NAME]['date'] >= start]['close'].iloc[0]
    above0 = (not pd.isna(raw[MAIN_NAME][raw[MAIN_NAME]['date'] >= start]['ma200'].iloc[0])
              and p0m > raw[MAIN_NAME][raw[MAIN_NAME]['date'] >= start]['ma200'].iloc[0])
    wm = W_MAIN if above0 else 0
    if p0m > 0: shares_main = INIT * wm / p0m * (1 - COMM - SLIP)
    if p0g > 0: shares_gold = INIT * W_GOLD / p0g * (1 - COMM - SLIP)
    cash = INIT - (INIT * wm + INIT * W_GOLD)
    last_month = None

    for date in dates:
        ym = (date.year, date.month)
        if DCA > 0 and last_month and ym != last_month:
            cash += DCA; total_inv += DCA
        last_month = ym

        pm = raw[MAIN_NAME][raw[MAIN_NAME]['date'] == date]
        pg = raw[GOLD_NAME][raw[GOLD_NAME]['date'] == date]
        if len(pm) == 0 or len(pg) == 0: continue
        px_m = pm['close'].iloc[0]; px_g = pg['close'].iloc[0]

        nav = cash + shares_main * px_m + shares_gold * px_g
        if nav > peak: peak = nav
        navs.append({'date': date, 'nav': nav})

        # 月底调仓
        if date.month != (dates[dates.index(date) - 1].month if dates.index(date) > 0 else date.month):
            # 判断红利趋势
            above = (not pd.isna(pm['ma200'].iloc[0]) and px_m > pm['ma200'].iloc[0])
            wm = W_MAIN if above else 0

            # 全清→重建
            cash += shares_main * px_m * (1 - COMM - SLIP)
            cash += shares_gold * px_g * (1 - COMM - SLIP)
            shares_main = 0.0; shares_gold = 0.0

            nav2 = cash
            if px_m > 0 and wm > 0:
                shares_main = nav2 * wm / px_m * (1 - COMM - SLIP)
                cash -= nav2 * wm
            if px_g > 0:
                shares_gold = nav2 * W_GOLD / px_g * (1 - COMM - SLIP)
                cash -= nav2 * W_GOLD

    # 统计
    ndf = pd.DataFrame(navs); final = ndf['nav'].iloc[-1]
    ret  = (final / (total_inv if DCA > 0 else INIT) - 1) * 100
    ann  = ((1 + ret/100) ** (252 / len(ndf)) - 1) * 100
    dr   = ndf['nav'].pct_change().dropna()
    sr   = (ann/100 - 0.02) / (dr.std() * np.sqrt(252)) if dr.std() > 0 else 0
    mdd  = ((ndf['nav']/INIT - (ndf['nav']/INIT).cummax())/(ndf['nav']/INIT).cummax()).min() * 100

    dca_label = f' DCA月投{DCA/1e4:.0f}万' if DCA > 0 else ''
    print(f"\n  YH_ultra v6  黄金{W_GOLD*100:.0f}%底仓+红利{W_MAIN*100:.0f}%卫星(MA{MA_TREND}过滤){dca_label}")
    print(f"  Return: {ret:+.2f}%  Annual: {ann:+.2f}%  Sharpe: {sr:.3f}  MaxDD: {mdd:+.2f}%")
    if DCA > 0:
        print(f"  投入{total_inv/1e4:.0f}万  终值{final/1e4:.1f}万  净赚{final-total_inv:,.0f}")
    else:
        print(f"  终值{final:,.0f}")

    ndf['year'] = ndf['date'].dt.year
    ys, yrets = [], []
    print(f"\n  {'年份':<6} {'收益':>8} {'MaxDD':>8}")
    for yr, grp in ndf.groupby('year'):
        if len(grp) < 10: continue
        yr_ret = (grp['nav'].iloc[-1]/grp['nav'].iloc[0] - 1) * 100
        yr_mdd = ((grp['nav']/grp['nav'].iloc[0] - (grp['nav']/grp['nav'].iloc[0]).cummax())/
                  (grp['nav']/grp['nav'].iloc[0]).cummax()).min() * 100
        print(f"  {yr:<6} {yr_ret:>+7.1f}% {yr_mdd:>+7.1f}%")
        ys.append(yr); yrets.append(yr_ret)

    # 图表
    if len(ndf) > 1:
        fig = plt.figure(figsize=(18, 12), facecolor='white')
        gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.2, 0.8],
                              hspace=0.2, left=0.06, right=0.94, top=0.96, bottom=0.04)

        ax1 = fig.add_subplot(gs[0])
        m_df = raw[MAIN_NAME][raw[MAIN_NAME]['date'] >= start]
        g_df = raw[GOLD_NAME][raw[GOLD_NAME]['date'] >= start]
        ax1.plot(m_df['date'], m_df['close']/m_df['close'].iloc[0], color='#9B59B6', lw=1.5, label=MAIN_NAME)
        ax1.plot(m_df['date'], m_df['ma200']/m_df['close'].iloc[0], color='#9B59B6', lw=0.8, ls='--', alpha=0.5, label=f'{MAIN_NAME}MA{MA_TREND}')
        ax1.plot(g_df['date'], g_df['close']/g_df['close'].iloc[0], color='#F39C12', lw=1.5, label=GOLD_NAME)
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.12)
        ax1.set_title('红利低波 + 黄金ETF (归一化)', fontsize=12, fontweight='bold'); ax1.tick_params(labelsize=8)

        ax2 = fig.add_subplot(gs[1]); ax2.set_facecolor('#FAFAFA')
        nc = '#CC2222' if ret >= 0 else '#228B22'
        ax2.fill_between(ndf['date'], 1, ndf['nav']/INIT, alpha=0.08, color=nc)
        ax2.plot(ndf['date'], ndf['nav']/INIT, color=nc, lw=2.0)
        ax2.axhline(y=1, color='#AAA', lw=0.8, ls='--')
        ax2.set_title(f'策略净值 {ret:+.1f}%  年化{ann:+.1f}%  夏普{sr:.3f}  回撤{mdd:.1f}%', fontsize=12, fontweight='bold', color=nc)
        ax2.tick_params(labelsize=8); ax2.grid(True, alpha=0.12)

        ax3 = fig.add_subplot(gs[2])
        colors2 = ['#CC2222' if r >= 0 else '#228B22' for r in yrets]
        bars = ax3.bar(range(len(ys)), yrets, color=colors2, alpha=0.85, edgecolor='white', lw=1)
        for bar, val in zip(bars, yrets):
            off = 1.5 if val >= 0 else -3.5
            ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+off,
                    f'{val:+.1f}%', ha='center', fontsize=13, fontweight='bold',
                    color='#CC2222' if val >= 0 else '#228B22')
        ax3.axhline(y=0, color='black', lw=1)
        ax3.set_xticks(range(len(ys))); ax3.set_xticklabels([str(y) for y in ys], fontsize=13, fontweight='bold')
        ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%+.0f%%'))
        ax3.set_title('年度收益', fontsize=12, fontweight='bold'); ax3.grid(True, alpha=0.12, axis='y')

        fig.suptitle(f'YH_ultra v6  黄金{W_GOLD*100:.0f}%底仓+红利{W_MAIN*100:.0f}%卫星(MA{MA_TREND}过滤)  月调仓{dca_label}',
                     fontsize=14, fontweight='bold', y=0.99)
        plt.savefig(os.path.join(SCRIPT, 'backtest_chart.png'), dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
        print(f'  图表: {SCRIPT}\\backtest_chart.png')
        ndf[['date', 'nav']].to_csv(os.path.join(SCRIPT, '_nav_ultra.csv'), index=False)


def live_signal():
    print("获取数据...")
    raw = fetch()
    for n in raw: raw[n] = add_ma(raw[n])

    is_weekend = pd.Timestamp.now().dayofweek >= 5
    if not is_weekend:
        try:
            spot = ak.fund_etf_spot_em()
            for code, name in [('512890', MAIN_NAME), ('518880', GOLD_NAME)]:
                s = spot[spot['代码'] == code]
                if len(s) > 0:
                    rt = float(s['最新价'].iloc[0])
                    raw[name].loc[raw[name].index[-1], 'close'] = rt
                    raw[name].loc[raw[name].index[-1], 'date'] = pd.Timestamp.now()
                    print(f'  {name} → {rt:.4f}')
            for n in raw: raw[n] = add_ma(raw[n])
        except Exception as e: print(f'  实时行情失败: {e}')

    idx = -1
    m = raw[MAIN_NAME].iloc[idx]; g = raw[GOLD_NAME].iloc[idx]
    date = m['date']; px_m = m['close']; px_g = g['close']
    ma200 = m['ma200']
    above = not pd.isna(ma200) and px_m > ma200
    wm = W_MAIN if above else 0

    print(f"\n{'='*55}")
    print(f"  YH_ultra v6  {date.strftime('%Y-%m-%d')}")
    print(f"  {MAIN_NAME} {px_m:.3f}  MA{MA_TREND}:{ma200:.3f}  {'↑趋势上' if above else '↓趋势下'}")
    print(f"  {GOLD_NAME} {px_g:.3f}")
    print(f"  配置: 黄金{W_GOLD*100:.0f}% + 红利{wm*100:.0f}% + 现金{(1-W_GOLD-wm)*100:.0f}%")
    print(f"{'='*55}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--from', dest='fr', type=str, default=None)
    p.add_argument('--dca', dest='dca', type=float, default=0, help='月定投额(万)')
    a = p.parse_args()
    global DCA; DCA = a.dca * 10000
    if a.fr:
        run_backtest(a.fr)
    else:
        live_signal()

if __name__ == '__main__':
    main()
