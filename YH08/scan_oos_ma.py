# -*- coding: utf-8 -*-
"""YH08 牛熊分界 MA 周期 —— 样本外(2024-2026)全周期扫描, 确认稳健最优"""
import daily_signal as d
import scan_params as sp
import sample_validation as sv

PERIODS = [30, 40, 50, 60, 90, 120, 150, 180, 250]

raw, dfs, in_dates, oos_dates = sv.load_split()
bp = sp.base_buy()
trail, hard, cd = d.TRAIL_STOP, d.HARD_STOP, d.COOLDOWN

print(f'[样本外 2024-2026] {len(oos_dates)} 天 (TS{trail:.0%}/HS{hard:.0%}/CD{cd})')
print(f'  {"MA":<6} {"Calmar":>7} {"年化":>8} {"回撤":>8} {"累计":>9} {"胜率":>6} {"卖笔":>5}')
print('  ' + '─' * 56)
rows = []
for p in PERIODS:
    d.REGIME_MA = p
    c, ann, mdd, ret, wr, nsell, td, spd = sp.sim(raw, dfs, oos_dates, bp, trail, hard, cd)
    rows.append((c, ann, mdd, ret, wr, nsell, p))
    print(f'  MA{p:<5} {c:>7.2f} {ann:>+7.1f}% {mdd:>+7.1f}% {ret:>+8.1f}% {wr:>5.0f}% {nsell:>5}')
rows.sort(key=lambda r: -r[0])
print(f'\n>>> 样本外最优: MA{rows[0][6]}  Calmar {rows[0][0]:.2f}')
d.REGIME_MA = 120
