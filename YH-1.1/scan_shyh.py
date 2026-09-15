# -*- coding: utf-8 -*-
"""上海银行参数扫描"""
import sys, io, warnings
import pandas as pd, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from daily_signal import *

# 只测上海银行一只, 简化回测
def scan_one_stock(symbol, name, rsi_v, bb_v, tp_v, tp_hi_v):
    raw = fetch()  # 获取所有数据 (需要ALL_STOCKS里的sym)
    df = add_indicators(raw[name])
    dates = df['date'].tolist()

    cash = INIT; shares = 0; entry = 0; high = 0
    accel = False; cooldown = 0; loss_streak = 0
    trades = []; navs = []

    for i, date in enumerate(dates):
        row = df.iloc[i]; cp = row['close']

        # 卖出
        if shares > 0:
            if cp > high: high = cp
            pnl = cp / entry - 1; dd = cp / high - 1
            do = False; sell_px = cp
            if pnl <= -HARD_STOP: do = True
            elif accel:
                if pnl >= tp_hi_v: do = True
                elif dd <= -TRAIL_STOP:
                    floor = entry * (1 + tp_v)
                    stop_px = max(high * (1 - TRAIL_STOP), floor)
                    if cp <= stop_px: do = True; sell_px = max(cp, floor)
            elif dd <= -TRAIL_STOP: do = True
            elif pnl >= tp_v:
                d2 = row.get('bb_up_d2')
                if not pd.isna(d2) and d2 > 0: accel = True
                else: do = True
            if do:
                cash += shares * sell_px * (1 - COMM - SLIP)
                pnl_real = sell_px / entry - 1
                trades.append(pnl_real)
                shares = 0; entry = 0; high = 0; accel = False
                if pnl_real > 0: loss_streak = 0
                else: loss_streak += 1
                if pnl_real <= -HARD_STOP: cooldown = COOLDOWN

        if cooldown > 0: cooldown -= 1

        # 买入
        if shares <= 0 and cooldown <= 0:
            ok = True
            r = row['rsi']; bb_p = (cp - row['bb_lo']) / (row['bb_up'] - row['bb_lo']) if row['bb_up'] > row['bb_lo'] else 0.5
            if pd.isna(r) or pd.isna(bb_p): ok = False
            if r > rsi_v or bb_p > bb_v: ok = False
            if ok:
                val = cash * MAX_POS  # 25%仓位
                if val > 5000:
                    qty = val / cp * (1 - COMM - SLIP)
                    shares = qty; cash -= val
                    entry = cp; high = cp

        nav = cash + shares * cp
        navs.append(nav)

    ndf = pd.DataFrame({'nav': navs})
    final = ndf['nav'].iloc[-1]
    ret = (final / INIT - 1) * 100
    if len(trades) == 0: return ret, 0, 0, 0, 0, 0
    sells = pd.Series(trades)
    wr = (sells > 0).sum() / len(sells) * 100
    aw = sells[sells > 0].mean() * 100 if (sells > 0).any() else 0
    al = sells[sells < 0].mean() * 100 if (sells < 0).any() else 0
    dr = ndf['nav'].pct_change().dropna()
    vol = dr.std() * np.sqrt(252) * 100
    ann = ((1 + ret/100) ** (252/len(ndf)) - 1) * 100
    sr = (ann - 2) / vol if vol > 0 else 0
    mdd = ((ndf['nav'] - ndf['nav'].cummax()) / ndf['nav'].cummax()).min() * 100
    return ret, wr, aw, al, sr, mdd, len(sells)

# 临时替换ALL_STOCKS只测上海银行
orig_core = CORE_STOCKS.copy()
orig_all = ALL_STOCKS.copy()
import daily_signal as ds
ds.CORE_STOCKS = {'上海银行': 'sh601229'}
ds.ALL_STOCKS = {'上海银行': 'sh601229'}

print("上海银行参数扫描...\n")
print(f"{'RSI':>5} {'BB':>6} {'TP':>6} {'TP_HI':>6} {'回报':>8} {'胜率':>6} {'均盈':>7} {'均亏':>7} {'夏普':>6} {'回撤':>7} {'笔数':>5}")
print(f"{'─'*75}")

best = {'ret': -999, 'params': None}
results = []

for rsi in [25, 30, 35, 40, 45]:
    for bb in [0.08, 0.12, 0.18, 0.25, 0.35]:
        for tp in [0.10, 0.15, 0.20]:
            tp_hi_v = tp + 0.05
            ret, wr, aw, al, sr, mdd, cnt = scan_one_stock('sh601229', '上海银行', rsi, bb, tp, tp_hi_v)
            results.append((ret, rsi, bb, tp, tp_hi_v, wr, aw, al, sr, mdd, cnt))
            marker = ' ★' if ret > best['ret'] else ''
            print(f"{rsi:>5} {bb:>5.2f} {tp:>5.0%} {tp_hi_v:>5.0%} {ret:>+7.1f}% {wr:>5.0f}% {aw:>+6.1f}% {al:>+6.1f}% {sr:>5.2f} {mdd:>+6.1f}% {cnt:>5}{marker}")
            if ret > best['ret']:
                best = {'ret': ret, 'params': (rsi, bb, tp, tp_hi_v), 'wr': wr, 'aw': aw, 'al': al, 'sr': sr, 'mdd': mdd, 'cnt': cnt}

print(f"\n{'─'*75}")
print(f"最优: RSI={best['params'][0]} BB={best['params'][1]:.2f} TP={best['params'][2]:.0%} TP_HI={best['params'][3]:.0%}")
print(f"回报={best['ret']:+.1f}% 胜率={best['wr']:.0f}% 均盈={best['aw']:+.1f}% 均亏={best['al']:+.1f}% 夏普={best['sr']:.2f} 回撤={best['mdd']:+.1f}% 笔数={best['cnt']}")

# 恢复
ds.CORE_STOCKS = orig_core
ds.ALL_STOCKS = orig_all
