# -*- coding: utf-8 -*-
"""YH08 滚动回测 (Walk-Forward): 从2016开始, 3年一个窗口, 每年滚动推进
固定参数(当前已调优: BASE_POS分档 + MA50牛熊 + TS8%/HS12%/CD40), 不逐窗重优化,
检验策略在不同3年周期上的稳健性。含每月定投2w。
"""
import sys, io, os, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

import daily_signal as d
import scan_params as sp

_fonts = [f.name for f in fm.fontManager.ttflist]
CN = 'SimHei' if 'SimHei' in _fonts else 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = [CN]; plt.rcParams['axes.unicode_minus'] = False

START_YR = 2016      # 滚动起点
WIN_YRS = 3          # 窗口宽度(年)
STEP_YRS = 1         # 步进(年)

CACHE = os.path.join(d.SCRIPT, '_scan_cache.pkl')


def load_full():
    """加载完整历史数据(缓存里已有 1997~2026 全量), 日期从 START_YR 起."""
    if not os.path.exists(CACHE):
        raw = d.fetch()
        dfs = {n: d.add_indicators(x) for n, x in raw.items()}
    else:
        with open(CACHE, 'rb') as f:
            raw, dfs = pickle.load(f)
    all_dates = sorted(set.intersection(*[set(x['date']) for x in dfs.values()]))
    all_dates = [x for x in all_dates if x >= pd.Timestamp(f'{START_YR}-01-01')]
    return raw, dfs, all_dates


def run_window(raw, dfs, dates, bp):
    """在当前参数下跑一个窗口, 返回 (metrics, ndf)."""
    ndf, td, total_injected, stock_pnl = d._simulate(raw, dfs, dates, True)
    c, ann, mdd, ret, wr, nsell = sp.metrics(ndf, total_injected, td)
    return {'calmar': c, 'ann': ann, 'mdd': mdd, 'ret': ret, 'wr': wr, 'nsell': nsell}, ndf


raw, dfs, all_dates = load_full()
print(f'[数据] {len(all_dates)} 天 ({all_dates[0].date()} ~ {all_dates[-1].date()})')
print(f'[参数] 固定当前调优参数  TS{d.TRAIL_STOP:.0%}/HS{d.HARD_STOP:.0%}/CD{d.COOLDOWN}  '
      f'REGIME_MA={d.REGIME_MA}  BASE_POS={d.BASE_POS}  月投{d.MONTHLY_INJECT/10000:.0f}w\n')

# 生成滚动窗口
windows = []
y = START_YR
while True:
    ws = pd.Timestamp(f'{y}-01-01')
    we = pd.Timestamp(f'{y + WIN_YRS}-01-01') - pd.Timedelta(days=1)
    if ws > all_dates[-1]:
        break
    wdates = [x for x in all_dates if ws <= x <= we]
    if len(wdates) >= 240:   # 至少约1年数据, 避免年化失真
        partial = we > all_dates[-1]
        windows.append((f'{y}-{y + WIN_YRS - 1}', ws, we, wdates, partial))
    y += STEP_YRS

print(f'  滚动窗口 {len(windows)} 个 ({WIN_YRS}年宽 / {STEP_YRS}年步进):')
for label, ws, we, wdates, partial in windows:
    mark = ' (数据未满)' if partial else ''
    print(f'    {label}: {ws.date()} ~ {wdates[-1].date()}{mark}')
print()

bp = sp.base_buy()          # 快照当前牛熊参数
d.BUY_PARAMS = bp
d.TRAIL_STOP, d.HARD_STOP, d.COOLDOWN = d.TRAIL_STOP, d.HARD_STOP, d.COOLDOWN

print(f'  {"窗口":<12} {"区间":<22} {"Calmar":>7} {"年化":>8} {"回撤":>8} {"累计":>8} {"胜率":>6} {"卖笔":>5}')
print('  ' + '─' * 76)

results = []
for label, ws, we, wdates, partial in windows:
    # 子集到窗口范围, 指标列(MA/RSI/MACD/BB)已按全历史预计算, 无未来函数
    raw_w = {n: raw[n][(raw[n]['date'] >= ws) & (raw[n]['date'] <= we)].reset_index(drop=True)
             for n in d.ALL_STOCKS}
    dfs_w = {n: dfs[n][(dfs[n]['date'] >= ws) & (dfs[n]['date'] <= we)].reset_index(drop=True)
             for n in d.ALL_STOCKS}
    m, ndf = run_window(raw_w, dfs_w, wdates, bp)
    mark = '*' if partial else ''
    print(f'  {label:<12} {ws.strftime("%Y-%m-%d")}~{wdates[-1].strftime("%Y-%m-%d"):<10} '
          f'{m["calmar"]:>7.2f} {m["ann"]:>+7.1f}% {m["mdd"]:>+7.1f}% {m["ret"]:>+7.1f}% '
          f'{m["wr"]:>5.0f}% {m["nsell"]:>5} {mark}')
    results.append({'label': label, 'partial': partial, **m, 'nav': ndf['nav']})

# 汇总(排除未满窗口, 避免年化失真)
full = [r for r in results if not r['partial']]
print('  ' + '─' * 76)
print(f'  {"均值":<12} {"":<22} {np.mean([r["calmar"] for r in full]):>7.2f} '
      f'{np.mean([r["ann"] for r in full]):>+7.1f}% {np.mean([r["mdd"] for r in full]):>+7.1f}% '
      f'{np.mean([r["ret"] for r in full]):>+7.1f}% {np.mean([r["wr"] for r in full]):>5.0f}%')
print(f'  {"最差":<12} {"":<22} {np.min([r["calmar"] for r in full]):>7.2f} '
      f'{np.min([r["ann"] for r in full]):>+7.1f}% {np.max([r["mdd"] for r in full]):>+7.1f}% '
      f'{np.min([r["ret"] for r in full]):>+7.1f}% {np.min([r["wr"] for r in full]):>5.0f}%')
pos_wins = sum(1 for r in full if r['ret'] > 0)
print(f'\n  稳健性: {pos_wins}/{len(full)} 个完整窗口盈利 '
      f'| 年化均值 {np.mean([r["ann"] for r in full]):+.1f}% '
      f'| Calmar均值 {np.mean([r["calmar"] for r in full]):.2f} '
      f'| 最差窗口 {min(full, key=lambda r: r["calmar"])["label"]}')

# 图表
if results:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor='white')
    ax = axes[0]
    cols = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for i, r in enumerate(results):
        n2 = r['nav'] / r['nav'].iloc[0]
        tag = '*' if r['partial'] else ''
        ax.plot(range(len(n2)), n2, color=cols[i], lw=1.8,
                label=f'{r["label"]}{tag} ({r["ret"]:+.0f}%)')
    ax.axhline(y=1, color='#888', lw=0.8, ls='--'); ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.12)
    ax.set_title(f'YH08 滚动回测净值 (3年窗口/年步进, 归一化起点=1)', fontsize=13, fontweight='bold')
    ax = axes[1]
    rets = [r['ret'] for r in results]
    colors2 = ['#CC2222' if v >= 0 else '#228B22' for v in rets]
    bars = ax.bar(range(len(results)), rets, color=colors2, alpha=0.85, edgecolor='white', lw=1)
    for bar, val in zip(bars, rets):
        off = 1.5 if val >= 0 else -3.5
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + off, f'{val:+.0f}%',
                ha='center', fontsize=11, fontweight='bold',
                color='#CC2222' if val >= 0 else '#228B22')
    ax.axhline(y=0, color='black', lw=1)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([r['label'] for r in results], fontsize=9)
    ax.set_title('各窗口累计收益', fontsize=13, fontweight='bold'); ax.grid(True, alpha=0.12, axis='y')
    out = os.path.join(d.SCRIPT, 'walkforward_chart.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print(f'\n  图表: {out}')
