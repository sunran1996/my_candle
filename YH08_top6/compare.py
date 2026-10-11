# -*- coding: utf-8 -*-
"""YH08_top6 (6股) vs YH08基线(4股) 三窗口对比."""
import sys, io, warnings, os
import pandas as pd
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import daily_signal as d   # 已自行包 utf-8 stdout

raw = d.fetch()
dfs = {n: d.add_indicators(x) for n, x in raw.items()}
dates = sorted(set.intersection(*[set(x['date']) for x in dfs.values()]))
dates = [x for x in dates if x >= pd.Timestamp('2020-01-01')]
in_dates = [x for x in dates if x < pd.Timestamp('2024-01-01')]
oos_dates = [x for x in dates if x >= pd.Timestamp('2024-01-01')]

def met(ndf, ti, td):
    final = ndf['nav'].iloc[-1]
    ret = final / ti - 1
    ann = (1 + ret) ** (252 / len(ndf)) - 1
    mdd = ((ndf['nav'] - ndf['nav'].cummax()) / ndf['nav'].cummax()).min()
    calmar = ann / abs(mdd) if mdd < 0 else 0
    sells = td[td['dir'] == 'SELL']
    wr = (sells['pnl'] > 0).sum() / len(sells) * 100 if len(sells) > 0 else 0
    return calmar, ann*100, mdd*100, ret*100, wr, len(sells)

print(f'\n数据 {dates[0].date()} ~ {dates[-1].date()}  共{len(dates)}交易日')
print(f'{"组合":<16} {"窗口":<3} {"Calmar":>7} {"年化":>7} {"回撤":>7} {"累计":>8} {"胜率":>5} {"卖":>4}')
base = {'全': (9.93, 84.0, -8.5), '内': (10.18, 86.1, -8.5), '外': (10.10, 74.7, -7.4)}
for ds, w in [(dates, '全'), (in_dates, '内'), (oos_dates, '外')]:
    ndf, td, ti, _ = d._simulate(raw, dfs, ds, True)
    c, a, m, r, wr, ns = met(ndf, ti, td)
    bc, ba, bm = base[w]
    print(f'  {"YH08基线(4股)":<14} {w:<3} {bc:>7.2f} {ba:>+6.1f}% {bm:>+6.1f}% {"—":>8} {"—":>5} {"—":>4}')
    print(f'  {"YH08_top6(6股)":<14} {w:<3} {c:>7.2f} {a:>+6.1f}% {m:>+6.1f}% {r:>+7.1f}% {wr:>4.0f}% {ns:>4}')
