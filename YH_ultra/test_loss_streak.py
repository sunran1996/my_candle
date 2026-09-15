# -*- coding: utf-8 -*-
"""测试 YH_ultra loss_streak 机制"""
import sys,io,os
import importlib.util
S=os.path.dirname(os.path.abspath(__file__))
spec=importlib.util.spec_from_file_location("x",os.path.join(S,"daily_signal.py"))
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')

print("=== YH_ultra 回测 (含 loss_streak) ===")
ndf,td=m.run_backtest()

sells=td[td['dir']=='SELL']
buys=td[td['dir']=='BUY']
print(f'总交易: {len(td)}笔 (买入{len(buys)}, 卖出{len(sells)})')

print('\n--- 连续亏损统计 ---')
for name in m.STOCKS:
    sub=sells[sells['name']==name].sort_values('date')
    streaks=[];run=0
    for _,t in sub.iterrows():
        if t['pnl']<0:run+=1
        else:
            if run>0:streaks.append(run)
            run=0
    if run>0:streaks.append(run)
    if streaks:
        print(f'{name}: 连续亏损组{len(streaks)}, 明细{streaks}, 最大{max(streaks)}')

worst=sells.loc[sells['pnl'].idxmin()]
best=sells.loc[sells['pnl'].idxmax()]
print(f'\n最佳: {best["date"].strftime("%Y-%m-%d")} {best["name"]} {best["pnl"]:.1f}%')
print(f'最差: {worst["date"].strftime("%Y-%m-%d")} {worst["name"]} {worst["pnl"]:.1f}%')

wins=len(sells[sells['pnl']>0])
print(f'胜率: {wins}/{len(sells)} = {wins/len(sells)*100:.1f}%')
print(f'均盈: {sells[sells["pnl"]>0]["pnl"].mean():.1f}%')
print(f'均亏: {sells[sells["pnl"]<0]["pnl"].mean():.1f}%')

import numpy as np
final_nav=ndf['nav'].iloc[-1]
total_return=(final_nav/m.INIT-1)*100
print(f'\n累计收益: {total_return:.1f}%')
print(f'最终净值: {final_nav:,.0f}')

days=(ndf['date'].iloc[-1]-ndf['date'].iloc[0]).days
years=days/365.25
ann=(final_nav/m.INIT)**(1/years)-1
print(f'年化: {ann*100:.1f}%')

nav_peak=ndf['nav'].cummax()
dd=(ndf['nav']-nav_peak)/nav_peak
print(f'最大回撤: {dd.min()*100:.1f}%')

ret=ndf['nav'].pct_change().dropna()
if len(ret)>0 and ret.std()>0:
    sharpe=(ret.mean()/ret.std())*np.sqrt(252)
    print(f'夏普: {sharpe:.2f}')

print('\n--- _quick_positions 验证 ---')
raw=m.fetch();dfs={n:m.add_indicators(d)for n,d in raw.items()}
positions,holdings,cash_end,recent=m._quick_positions(raw,dfs)
print(f'现金: {cash_end:,.0f}')
for n in m.STOCKS:
    if holdings[n]>0:
        print(f'{n}: {holdings[n]:.0f}股, 成本{positions[n]:.2f}')
print(f'最近交易: {len(recent)}笔')
