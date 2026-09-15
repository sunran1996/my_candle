# -*- coding: utf-8 -*-
"""用调优结果做最终整体回测验证 (加DCA, 2020起), 并与基线对比"""
import os, json, pickle
import pandas as pd
import numpy as np
import daily_signal as d

SCRIPT = d.SCRIPT
CACHE = os.path.join(SCRIPT, '_scan_cache.pkl')
BEST = os.path.join(SCRIPT, '_scan_best.json')

with open(CACHE, 'rb') as f:
    raw, dfs = pickle.load(f)
dates = sorted(set.intersection(*[set(x['date']) for x in dfs.values()]))
dates = [x for x in dates if x >= pd.Timestamp('2020-01-01')]

with open(BEST, 'r', encoding='utf-8') as f:
    b = json.load(f)
g = b['global']
final = b['final']
# 兼容旧版平铺格式 {name: params} → 视为牛市参数(熊市保持默认)
if final and all(k in d.ALL_STOCKS for k in final):
    final = {'bull': final}


def run(buy_params, trail, hard, cd):
    d.BUY_PARAMS = buy_params
    d.TRAIL_STOP = trail
    d.HARD_STOP = hard
    d.COOLDOWN = cd
    ndf, td, total_injected, sp = d._simulate(raw, dfs, dates, True)
    fv = ndf['nav'].iloc[-1]
    ret = fv / total_injected - 1
    ann = (1 + ret) ** (252 / len(ndf)) - 1
    mdd = ((ndf['nav'] - ndf['nav'].cummax()) / ndf['nav'].cummax()).min()
    calmar = ann / abs(mdd)
    vol = ndf['nav'].pct_change().dropna().std() * np.sqrt(252)
    sr = (ann - 0.02) / vol if vol > 0 else 0
    sells = td[td['dir'] == 'SELL']
    wr = (sells['pnl'] > 0).mean() * 100 if len(sells) else 0
    return ret * 100, ann * 100, mdd * 100, calmar, sr, wr, len(sells), len(td), sp


def show(tag, r):
    ret, ann, mdd, calmar, sr, wr, nsell, ntd, sp = r
    print(f'  {tag:<14} 累计{ret:>+7.1f}%  年化{ann:>+6.1f}%  回撤{mdd:>+6.1f}%  '
          f'Calmar{calmar:>5.2f}  夏普{sr:>4.2f}  胜率{wr:>4.0f}%  卖{nsell:>3}笔')
    for n in d.ALL_STOCKS:
        print(f'      {n:<8} {sp[n]/10000:>+7.1f}w')


# 基线 (当前默认)
base_buy = {reg: {n: dict(d.BUY_PARAMS[reg][n]) for n in d.ALL_STOCKS} for reg in ('bull', 'bear')}
print('=' * 70)
print('  YH08 调优验证 (加DCA, 2020-01-01 起)')
print('=' * 70)
show('基线', run(base_buy, 0.08, 0.10, 20))

# 最优参数
opt_buy = {reg: {n: dict(d.BUY_PARAMS[reg][n]) for n in d.ALL_STOCKS} for reg in ('bull', 'bear')}
for reg, stocks in final.items():
    for n, p in stocks.items():
        opt_buy[reg][n].update(p)
print('\n  最优参数:')
print(f'    TRAIL_STOP={g["trail"]}  HARD_STOP={g["hard"]}  COOLDOWN={g["cooldown"]}')
for reg in ('bull', 'bear'):
    print(f'    [{reg}]')
    for n in d.CORE_STOCKS:
        print(f'      {n}: {opt_buy[reg][n]}')
print()
show('调优后', run(opt_buy, g['trail'], g['hard'], g['cooldown']))
