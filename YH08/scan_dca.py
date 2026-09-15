# -*- coding: utf-8 -*-
"""YH08 定投(DCA)金额扫描: 对比不同月投金额的 Calmar/年化/回撤/累计/终值
口径: 归一化(每投入1元的回报), 全区间 2020 起, 用当前 BASE_POS(神华国电0.35) + global 卖出参数
"""
import daily_signal as d
import scan_params as sp

AMOUNTS = [0, 10000, 20000, 30000, 50000, 100000]

raw, dfs, dates = sp.load()
bp = sp.base_buy()
g = sp.load_best().get('global', {'trail': d.TRAIL_STOP, 'hard': d.HARD_STOP, 'cooldown': d.COOLDOWN})
d.TRAIL_STOP, d.HARD_STOP, d.COOLDOWN = g['trail'], g['hard'], g['cooldown']


def run(amt):
    d.MONTHLY_INJECT = amt
    d.BUY_PARAMS = bp
    ndf, td, total_injected, stock_pnl = d._simulate(raw, dfs, dates, True)
    final = ndf['nav'].iloc[-1]
    ret = final / total_injected - 1
    ann = (1 + ret) ** (252 / len(ndf)) - 1
    mdd = ((ndf['nav'] - ndf['nav'].cummax()) / ndf['nav'].cummax()).min()
    calmar = ann / abs(mdd)
    sells = td[td['dir'] == 'SELL']
    wr = (sells['pnl'] > 0).mean() * 100 if len(sells) else 0
    return calmar, ann * 100, mdd * 100, ret * 100, wr, len(sells), final, total_injected


print(f'[数据] {len(dates)} 天 ({dates[0].date()} ~ {dates[-1].date()})')
print(f'[参数] TS{d.TRAIL_STOP:.0%}/HS{d.HARD_STOP:.0%}/CD{d.COOLDOWN}  BASE_POS={d.BASE_POS}')
print()
print(f'  {"月投":<7} {"本金":>8} {"终值":>8} {"累计":>8} {"年化":>8} {"回撤":>7} {"Calmar":>7} {"胜率":>6} {"卖笔":>5}')
print('  ' + '─' * 70)
for amt in AMOUNTS:
    c, ann, mdd, ret, wr, nsell, final, tot = run(amt)
    tag = '无定投' if amt == 0 else f'{amt/10000:.0f}w'
    print(f'  {tag:<7} {tot/10000:>6.0f}w  {final/10000:>6.0f}w  {ret:>+7.1f}%  {ann:>+7.1f}%  {mdd:>+6.1f}%  {c:>7.2f}  {wr:>5.0f}%  {nsell:>5}')

d.MONTHLY_INJECT = 20000  # 恢复默认
