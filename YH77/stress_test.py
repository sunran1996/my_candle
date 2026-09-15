# -*- coding: utf-8 -*-
"""YH77 压力测试:
  (1) 连亏加倍仓位的贡献 (有加倍 vs 无加倍)
  (2) 单票 -30% 跳空注入 (在最大集中度时点)
  (3) 全组合 -30% 统一跳空 (系统性利空一天)
"""
import daily_signal as d
import scan_params as sp


def fmt(tag, r):
    c, ann, mdd, ret, wr, nsell = r[:6]
    return f'{tag:<22} Calmar{c:>6.2f}  年化{ann:>+6.1f}%  回撤{mdd:>+6.1f}%  累计{ret:>+7.1f}%  胜率{wr:>4.0f}%  卖{nsell:>3}笔'


raw, dfs, dates = sp.load()
bp = sp.base_buy()

print('=' * 78)
print('【测试1】连亏加倍仓位的贡献 (全区间 2020-2026)')
r0 = sp.sim(raw, dfs, dates, bp, d.TRAIL_STOP, d.HARD_STOP, d.COOLDOWN)
print('  ' + fmt('基线(有连亏加倍)', r0))
orig_boost, orig_double = d.MAX_POS_BOOST, d.MAX_POS_DOUBLE
d.MAX_POS_BOOST = d.MAX_POS
d.MAX_POS_DOUBLE = d.MAX_POS
r1 = sp.sim(raw, dfs, dates, bp, d.TRAIL_STOP, d.HARD_STOP, d.COOLDOWN)
d.MAX_POS_BOOST, d.MAX_POS_DOUBLE = orig_boost, orig_double
print('  ' + fmt('无连亏加倍(固定上限)', r1))
print(f'  → 连亏加倍贡献的 Calmar 差: {r0[0]-r1[0]:+.2f}, 年化差 {r0[1]-r1[1]:+.1f}pp, 回撤差 {r0[2]-r1[2]:+.1f}pp')

# 重建持仓, 找最大单票集中度
td = r0[6]
shares = {n: 0 for n in d.ALL_STOCKS}
max_w = 0.0; max_info = None
for _, t in td.iterrows():
    n = t['name']; q = int(t['qty'])
    shares[n] += q if t['dir'] == 'BUY' else -q
    nav = float(t['nav'])
    if nav <= 0:
        continue
    px = dfs[n][dfs[n]['date'] == t['date']]['close']
    if len(px) == 0:
        continue
    w = shares[n] * float(px.iloc[0]) / nav
    if w > max_w:
        max_w = w; max_info = (t['date'], n, shares[n], float(px.iloc[0]), nav)
d_, n_, sh_, px_, nav_ = max_info
print()
print('=' * 78)
print(f'【测试2】单票跳空: {n_} 在 {d_.date()} 占净值 {max_w*100:.1f}% ({sh_}股 × {px_:.2f})')
print(f'  → 若当天跳空 -30%: 止损按当日收盘价成交, 实际-30%而非-12%, 单日净值冲击约 {max_w*30:.1f}%')

GAP = -0.30
raw2 = {k: v.copy() for k, v in raw.items()}
df = raw2[n_]
m = df['date'] == d_
i = df.index[m][0]
old_c = float(df.loc[i, 'close'])
df.loc[i, 'close'] = old_c * (1 + GAP)
df.loc[i, 'open'] = df.loc[i, 'close']
df.loc[i, 'low'] = min(float(df.loc[i, 'low']), float(df.loc[i, 'close']))
raw2[n_] = df
dfs2 = {n: d.add_indicators(x) for n, x in raw2.items()}
r2 = sp.sim(raw2, dfs2, dates, bp, d.TRAIL_STOP, d.HARD_STOP, d.COOLDOWN)
print('  ' + fmt(f'{n_} 单日-30%跳空后', r2))

print()
print('=' * 78)
print('【测试3】全组合 8 支同日 -30% 统一跳空 (系统性利空一天, 最后交易日注入)')
GAP3 = -0.30
last_date = dates[-1]
raw3 = {k: v.copy() for k, v in raw.items()}
for n in d.ALL_STOCKS:
    df = raw3[n]
    m = df['date'] == last_date
    if m.any():
        i = df.index[m][0]
        old = float(df.loc[i, 'close'])
        df.loc[i, 'close'] = old * (1 + GAP3)
        df.loc[i, 'open'] = df.loc[i, 'close']
        df.loc[i, 'low'] = min(float(df.loc[i, 'low']), float(df.loc[i, 'close']))
dfs3 = {n: d.add_indicators(x) for n, x in raw3.items()}
r3 = sp.sim(raw3, dfs3, dates, bp, d.TRAIL_STOP, d.HARD_STOP, d.COOLDOWN)
print('  ' + fmt('全组合-30%跳空后', r3))
print()
print('=' * 78)
