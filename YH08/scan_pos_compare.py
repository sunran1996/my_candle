# -*- coding: utf-8 -*-
"""YH08 默认仓位调整对比: 神华/国电 0.25→0.35 (全区间 + 样本外验证)"""
import daily_signal as d
import scan_params as sp
import sample_validation as sv

BASE0 = {'长江电力': 0.25, '招商银行': 0.25, '国电电力': 0.25, '中国神华': 0.25}
BASE1 = {'长江电力': 0.25, '招商银行': 0.25, '国电电力': 0.35, '中国神华': 0.35}

raw, dfs, dates = sp.load()


def run(tag, base_pos, ds, trail, hard, cd):
    d.BASE_POS = base_pos
    bp = sp.base_buy()
    c, ann, mdd, ret, wr, nsell, td, spd = sp.sim(raw, dfs, ds, bp, trail, hard, cd)
    print(f'  {tag:<16} Calmar{c:>5.2f}  年化{ann:>+6.1f}%  回撤{mdd:>+6.1f}%  累计{ret:>+7.1f}%  胜率{wr:>4.0f}%  卖{nsell:>3}笔')
    for n in d.ALL_STOCKS:
        print(f'      {n:<8} {spd[n]/10000:>+7.1f}w')
    return c


g = sp.load_best().get('global', {'trail': d.TRAIL_STOP, 'hard': d.HARD_STOP, 'cooldown': d.COOLDOWN})
print(f'[全区间 2020-2026] {len(dates)} 天 (TS{g["trail"]:.0%}/HS{g["hard"]:.0%}/CD{g["cooldown"]})')
run('基线 全0.25', BASE0, dates, g['trail'], g['hard'], g['cooldown'])
run('神华国电0.35', BASE1, dates, g['trail'], g['hard'], g['cooldown'])

_, _, in_dates, oos_dates = sv.load_split()
g_in = sv.load_in().get('global', {'trail': 0.10, 'hard': 0.12, 'cooldown': 40})
print(f'\n[样本外 2024-2026] {len(oos_dates)} 天 (TS{g_in["trail"]:.0%}/HS{g_in["hard"]:.0%}/CD{g_in["cooldown"]})')
run('基线 全0.25', BASE0, oos_dates, g_in['trail'], g_in['hard'], g_in['cooldown'])
run('神华国电0.35', BASE1, oos_dates, g_in['trail'], g_in['hard'], g_in['cooldown'])
