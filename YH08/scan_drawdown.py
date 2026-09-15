# -*- coding: utf-8 -*-
"""YH08 回撤优化扫描: 目标降回撤、保/升 Calmar (年化/|回撤|)
维度1: 组合回撤止损 NAV_STOP × 全局冷却 NAV_COOL (单笔止损保持 8%/12%)
维度2: 收紧单笔止损 TRAIL_STOP × HARD_STOP (无组合止损)
维度3: 降低单笔仓位上限 BASE_POS (神华/国电从35%下调)
"""
import sys, itertools
import daily_signal as d   # 内部已把 stdout 包装为 UTF-8
import scan_params as sp


def run(nav_stop, nav_cool, trail, hard, cd, bp):
    d.NAV_STOP = nav_stop; d.NAV_COOL = nav_cool
    d.TRAIL_STOP = trail; d.HARD_STOP = hard; d.COOLDOWN = cd
    ndf, td, total_injected, stock_pnl = d._simulate(raw, dfs, dates, True)
    c, ann, mdd, ret, wr, nsell = sp.metrics(ndf, total_injected, td)
    nstop = len(td[(td['dir'] == 'SELL') & (td['reason'].str.contains('组合止损', na=False))])
    return c, ann, mdd, ret, wr, nsell, nstop


def line(tag, c, ann, mdd, ret, wr, nsell, nstop=''):
    return (f'  {tag:<16} {c:>7.2f} {ann:>+7.1f}% {mdd:>+7.1f}% '
            f'{ret:>+7.1f}% {wr:>5.0f}% {nsell:>5} {nstop:>6}')


raw, dfs, dates = sp.load()
bp = sp.base_buy()
print(f'[数据] {len(dates)} 天 ({dates[0].date()} ~ {dates[-1].date()})')
print(f'[参数] BASE_POS={d.BASE_POS}  REGIME_MA={d.REGIME_MA}  月投{d.MONTHLY_INJECT/10000:.0f}w\n')

# 基线
d.NAV_STOP = None; d.NAV_COOL = 0
c, ann, mdd, ret, wr, nsell, nstop = run(None, 0, 0.08, 0.12, 40, bp)
print(f'  {"基线":<16} Calmar{ c:>6.2f}  年化{ann:>+7.1f}%  回撤{mdd:>+7.1f}%  累计{ret:>+7.1f}%  胜率{wr:>5.0f}%  卖{nsell}笔\n')

# 维度1: NAV_STOP
print('[维度1] 组合回撤止损 NAV_STOP × 冷却 (单笔止损 8%/12% 不变):')
print(f'  {"参数":<16} {"Calmar":>7} {"年化":>8} {"回撤":>8} {"累计":>8} {"胜率":>6} {"卖":>5} {"组合止损":>6}')
rows1 = []
for ns, cool in itertools.product([0.06, 0.08, 0.10, 0.12], [10, 20, 40]):
    c, ann, mdd, ret, wr, nsell, nstop = run(ns, cool, 0.08, 0.12, 40, bp)
    rows1.append((c, ann, mdd, ret, wr, ns, cool, nstop))
    print(line(f'NS{ns:.0%}/冷{cool}d', c, ann, mdd, ret, wr, nsell, nstop))
rows1.sort(key=lambda r: -r[0])
best1 = rows1[0]
print(f'  >>> 维度1最优: NAV_STOP={best1[6]:.0%} 冷却{best1[7]}d  Calmar {best1[0]:.2f} 回撤 {best1[2]:+.1f}%\n')

# 维度2: 收紧单笔止损
print('[维度2] 收紧单笔止损 (无组合止损):')
print(f'  {"参数":<16} {"Calmar":>7} {"年化":>8} {"回撤":>8} {"累计":>8} {"胜率":>6} {"卖":>5} {"组合止损":>6}')
rows2 = []
for trail, hard in itertools.product([0.05, 0.06, 0.08], [0.08, 0.10, 0.12]):
    if hard <= trail:
        continue
    c, ann, mdd, ret, wr, nsell, nstop = run(None, 0, trail, hard, 40, bp)
    rows2.append((c, ann, mdd, ret, wr, trail, hard))
    print(line(f'TS{trail:.0%}/HS{hard:.0%}', c, ann, mdd, ret, wr, nsell))
rows2.sort(key=lambda r: -r[0])
best2 = rows2[0]
print(f'  >>> 维度2最优: TS{best2[5]:.0%}/HS{best2[6]:.0%}  Calmar {best2[0]:.2f} 回撤 {best2[2]:+.1f}%\n')

# 维度3: 降仓位
print('[维度3] 降低单笔仓位上限 (无组合止损, 止损8%/12%):')
print(f'  {"参数":<16} {"Calmar":>7} {"年化":>8} {"回撤":>8} {"累计":>8} {"胜率":>6} {"卖":>5} {"组合止损":>6}')
rows3 = []
pos_cfgs = [
    ('神华国电25%', {'长江电力': 0.25, '招商银行': 0.25, '国电电力': 0.25, '中国神华': 0.25}),
    ('神华国电30%', {'长江电力': 0.25, '招商银行': 0.25, '国电电力': 0.30, '中国神华': 0.30}),
    ('当前35%',     {'长江电力': 0.25, '招商银行': 0.25, '国电电力': 0.35, '中国神华': 0.35}),
    ('全部20%',     {'长江电力': 0.20, '招商银行': 0.20, '国电电力': 0.20, '中国神华': 0.20}),
]
for tag, cfg in pos_cfgs:
    d.BASE_POS = cfg
    c, ann, mdd, ret, wr, nsell, nstop = run(None, 0, 0.08, 0.12, 40, bp)
    rows3.append((c, ann, mdd, ret, wr, tag))
    print(line(tag, c, ann, mdd, ret, wr, nsell))
rows3.sort(key=lambda r: -r[0])
best3 = rows3[0]
print(f'  >>> 维度3最优: {best3[5]}  Calmar {best3[0]:.2f} 回撤 {best3[2]:+.1f}%')

d.NAV_STOP = None; d.NAV_COOL = 0
d.TRAIL_STOP, d.HARD_STOP, d.COOLDOWN = 0.08, 0.12, 40
d.BASE_POS = {'长江电力': 0.25, '招商银行': 0.25, '国电电力': 0.35, '中国神华': 0.35}
