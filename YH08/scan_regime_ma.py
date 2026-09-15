# -*- coding: utf-8 -*-
"""YH08 牛熊分界 MA 周期扫描: 全区间 + 样本外验证
market_regime 用 现价 vs MA{REGIME_MA} 判牛熊, 扫不同 MA 周期看 Calmar 最优.
"""
import daily_signal as d
import scan_params as sp
import sample_validation as sv

PERIODS = [30, 40, 50, 60, 90, 120, 150, 180, 250]


def scan(raw, dfs, dates, tag):
    bp = sp.base_buy()
    trail, hard, cd = d.TRAIL_STOP, d.HARD_STOP, d.COOLDOWN
    print(f'\n[{tag}] {len(dates)} 天 (TS{trail:.0%}/HS{hard:.0%}/CD{cd})')
    print(f'  {"MA周期":<8} {"Calmar":>7} {"年化":>8} {"回撤":>8} {"累计":>9} {"胜率":>6} {"卖笔":>5} {"牛买":>5} {"熊买":>5}')
    print('  ' + '─' * 68)
    rows = []
    for p in PERIODS:
        d.REGIME_MA = p
        c, ann, mdd, ret, wr, nsell, td, spd = sp.sim(raw, dfs, dates, bp, trail, hard, cd)
        nb = len(td[(td['dir'] == 'BUY') & (td['reason'].str.contains('牛', na=False))])
        xb = len(td[(td['dir'] == 'BUY') & (td['reason'].str.contains('熊', na=False))])
        rows.append((c, ann, mdd, ret, wr, nsell, p, nb, xb))
        print(f'  MA{p:<6} {c:>7.2f} {ann:>+7.1f}% {mdd:>+7.1f}% {ret:>+8.1f}% {wr:>5.0f}% {nsell:>5} {nb:>5} {xb:>5}')
    rows.sort(key=lambda r: -r[0])
    d.REGIME_MA = 120
    return rows[0]


raw, dfs, dates = sp.load()
print(f'[数据] {len(dates)} 天 ({dates[0].date()} ~ {dates[-1].date()})')

best_full = scan(raw, dfs, dates, '全区间 2020-2026')
print(f'\n>>> 全区间最优: MA{best_full[6]}  Calmar {best_full[0]:.2f}')

_, _, in_dates, oos_dates = sv.load_split()
best_in = scan(raw, dfs, in_dates, '样本内 2020-2023')
print(f'\n>>> 样本内最优: MA{best_in[6]}  Calmar {best_in[0]:.2f}')

print(f'\n[样本外 2024-2026 验证] {len(oos_dates)} 天')
bp = sp.base_buy()
trail, hard, cd = d.TRAIL_STOP, d.HARD_STOP, d.COOLDOWN
d.REGIME_MA = 120
c0, ann0, mdd0, ret0, wr0, ns0, td0, spd0 = sp.sim(raw, dfs, oos_dates, bp, trail, hard, cd)
print(f'  基线 MA120:      Calmar {c0:.2f}  年化 {ann0:+.1f}%  回撤 {mdd0:+.1f}%  累计 {ret0:+.1f}%')
d.REGIME_MA = best_in[6]
c1, ann1, mdd1, ret1, wr1, ns1, td1, spd1 = sp.sim(raw, dfs, oos_dates, bp, trail, hard, cd)
print(f'  样本内最优 MA{best_in[6]:<3}: Calmar {c1:.2f}  年化 {ann1:+.1f}%  回撤 {mdd1:+.1f}%  累计 {ret1:+.1f}%')
d.REGIME_MA = 120
