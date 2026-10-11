# -*- coding: utf-8 -*-
"""YH08_top6: 给 3 只未调优票(中国银行/国投电力/工商银行)套用与 4 只老票相同的
牛熊分状态调优逻辑 —— 每股按现价 vs MA50 分牛熊, 各扫 rsi×bb → tp×tp_hi, 以组合 Calmar 为目标
(与 YH08/scan_params.py 同一套网格). 坐标下降: 先扫 rsi/bb 再扫 tp/tp_hi, 按标的一只只推进.

并行: 每个 (regime, 标的) 的扫描轮次内, 各参数组合互相独立, 用 multiprocessing.Pool 并行跑.
最后输出 3 窗口(全/样本内/样本外)对比: 当前中性6股 vs 调优后6股.
用法: python tune_3.py
"""
import sys, os, pickle, time, itertools, copy
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CACHE = os.path.join(HERE, '_tune_cache.pkl')
START = '2020-01-01'
OOS = '2024-01-01'

TUNE = ['中国银行', '国投电力', '工商银行']   # 3 只未调优票, 按此顺序坐标下降

# rsi 网格: 25 与 30 等价(check_buy 里 'rsi<=30 就 +1' 使 rsi_th<=30 时与 30 等价), 去掉冗余 25
RSIS = [30, 35, 40, 45, 50, 55]
BBS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
BULL_TPS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
BULL_TPHI = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
BEAR_TPS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
BEAR_TPHI = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

import daily_signal as d   # 父进程仅用于 load/snapshot; worker 各自 import 一份

# ── worker 全局 ──
_W = {}


def _init(raw, dfs, dates):
    import daily_signal as dd
    _W['d'] = dd; _W['raw'] = raw; _W['dfs'] = dfs; _W['dates'] = dates


def _metrics(ndf, ti, td):
    final = ndf['nav'].iloc[-1]
    ret = final / ti - 1
    ann = (1 + ret) ** (252 / len(ndf)) - 1
    mdd = ((ndf['nav'] - ndf['nav'].cummax()) / ndf['nav'].cummax()).min()
    calmar = ann / abs(mdd) if mdd < 0 else 0
    sells = td[td['dir'] == 'SELL']
    wr = (sells['pnl'] > 0).mean() * 100 if len(sells) else 0
    return calmar, ann * 100, mdd * 100, ret * 100, wr, len(sells)


def _work(bp):
    dd = _W['d']
    dd.BUY_PARAMS = bp
    ndf, td, ti, sp = dd._simulate(_W['raw'], _W['dfs'], _W['dates'], True)
    c, ann, mdd, ret, wr, nsell = _metrics(ndf, ti, td)
    return c, ann, mdd, ret, wr, nsell, sp


def load():
    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            raw, dfs = pickle.load(f)
        print(f'[数据] 缓存加载 ({time.strftime("%H:%M", time.localtime(os.path.getmtime(CACHE)))})', flush=True)
    else:
        print('[数据] 拉取并缓存...', flush=True)
        raw = d.fetch()
        dfs = {n: d.add_indicators(x) for n, x in raw.items()}
        with open(CACHE, 'wb') as f:
            pickle.dump((raw, dfs), f)
    dates = sorted(set.intersection(*[set(x['date']) for x in dfs.values()]))
    dates = [x for x in dates if x >= pd.Timestamp(START)]
    in_dates = [x for x in dates if x < pd.Timestamp(OOS)]
    oos_dates = [x for x in dates if x >= pd.Timestamp(OOS)]
    return raw, dfs, dates, in_dates, oos_dates


def snapshot():
    return {reg: {n: dict(d.BUY_PARAMS[reg][n]) for n in d.ALL_STOCKS} for reg in ('bull', 'bear')}


def fmt(tag, r):
    c, ann, mdd, ret, wr, nsell = r[:6]
    return f'  {tag:<24} Calmar{c:>6.2f}  年化{ann:>+6.1f}%  回撤{mdd:>+6.1f}%  累计{ret:>+7.1f}%  胜率{wr:>4.0f}%  卖{nsell:>3}笔'


def tune_stock(pool, raw, dfs, dates, best, name, regime):
    tps, tphis = (BULL_TPS, BULL_TPHI) if regime == 'bull' else (BEAR_TPS, BEAR_TPHI)
    print(f'\n══ {name} · {regime} ══', flush=True)

    # 2a: rsi × bb
    cands = []
    for rsi, bb in itertools.product(RSIS, BBS):
        bp = copy.deepcopy(best)
        bp[regime][name]['rsi'] = rsi
        bp[regime][name]['bb'] = bb
        cands.append((f'rsi{rsi} bb{bb}', rsi, bb, bp))
    res = pool.map(_work, [c[3] for c in cands])
    rows = [(cands[i][0],) + res[i][:6] + (cands[i][1], cands[i][2], res[i][6][name]) for i in range(len(cands))]
    rows.sort(key=lambda r: -r[1])
    best_rsi, best_bb = rows[0][7], rows[0][8]
    best[regime][name]['rsi'] = int(best_rsi)
    best[regime][name]['bb'] = float(best_bb)
    print(f'  [rsi×bb] 最优 rsi{best_rsi} bb{best_bb} (组合 Calmar {rows[0][1]:.2f}, 该股 {rows[0][9]/10000:+.1f}w)', flush=True)
    for r in rows[:5]:
        print(fmt(r[0], r[1:7]) + f'  该股{r[9]/10000:+.1f}w', flush=True)

    # 2b: tp × tp_hi
    cands = []
    for tp, tp_hi in itertools.product(tps, tphis):
        if tp_hi <= tp:
            continue
        bp = copy.deepcopy(best)
        bp[regime][name]['tp'] = tp
        bp[regime][name]['tp_hi'] = tp_hi
        cands.append((f'tp{tp:.0%} tp_hi{tp_hi:.0%}', tp, tp_hi, bp))
    res = pool.map(_work, [c[3] for c in cands])
    rows = [(cands[i][0],) + res[i][:6] + (cands[i][1], cands[i][2], res[i][6][name]) for i in range(len(cands))]
    rows.sort(key=lambda r: -r[1])
    best[regime][name]['tp'] = rows[0][7]
    best[regime][name]['tp_hi'] = rows[0][8]
    print(f'  [tp×tp_hi] rsi{best_rsi} bb{best_bb} → 最优 tp{rows[0][7]:.0%} tp_hi{rows[0][8]:.0%} (组合 Calmar {rows[0][1]:.2f}, 该股 {rows[0][9]/10000:+.1f}w)', flush=True)
    for r in rows[:5]:
        print(fmt(r[0], r[1:7]) + f'  该股{r[9]/10000:+.1f}w', flush=True)
    return best


def main():
    import multiprocessing as mp
    raw, dfs, dates, in_dates, oos_dates = load()
    print(f'[数据] {len(dates)} 交易日 ({dates[0].date()} ~ {dates[-1].date()})', flush=True)

    base = snapshot()
    print('\n[基线] 当前(3只中性) 全区间:', flush=True)

    nproc = max(1, min(8, mp.cpu_count()))
    print(f'[并行] 使用 {nproc} 进程', flush=True)
    pool = mp.Pool(nproc, initializer=_init, initargs=(raw, dfs, dates))

    # 基线(全窗口)单独算一次
    c, ann, mdd, ret, wr, nsell, sp = pool.map(_work, [base])[0]
    print(fmt('中性6股', (c, ann, mdd, ret, wr, nsell)), flush=True)
    for n in d.ALL_STOCKS:
        print(f'    {n:<8} {sp[n]/10000:>+7.1f}w', flush=True)

    t0 = time.time()
    tuned = copy.deepcopy(base)
    for regime in ('bull', 'bear'):
        for name in TUNE:
            tune_stock(pool, raw, dfs, dates, tuned, name, regime)
    print(f'\n[调优] 耗时 {time.time()-t0:.0f}s', flush=True)

    # 汇总最优参数
    print('\n[最优参数] (3只未调优票 → 调优后):', flush=True)
    for regime in ('bull', 'bear'):
        for name in TUNE:
            b = base[regime][name]; t = tuned[regime][name]
            chg = [f'{k}: {b[k]}→{t[k]}' for k in ('rsi', 'bb', 'tp', 'tp_hi') if b[k] != t[k]]
            mark = '  '.join(chg) if chg else '  (无变化)'
            print(f'  {regime:<4} {name:<8}  {mark}', flush=True)

    pool.close()
    pool.join()

    # 三窗口用父进程单跑(仅6次, 快)
    print('\n[三窗口对比]', flush=True)
    d.BUY_PARAMS = base
    for ds, w in [(dates, '全'), (in_dates, '内'), (oos_dates, '外')]:
        ndf, td, ti, sp = d._simulate(raw, dfs, ds, True)
        print(fmt(f'中性6股·{w}', _metrics(ndf, ti, td)), flush=True)
    d.BUY_PARAMS = tuned
    for ds, w in [(dates, '全'), (in_dates, '内'), (oos_dates, '外')]:
        ndf, td, ti, sp = d._simulate(raw, dfs, ds, True)
        print(fmt(f'调优6股·{w}', _metrics(ndf, ti, td)), flush=True)

    # 打印可直接粘贴到 daily_signal.py 的参数块
    print('\n[可粘贴参数块]', flush=True)
    for regime in ('bull', 'bear'):
        for name in TUNE:
            t = tuned[regime][name]
            print(f"        '{name}': {{'rsi': {t['rsi']}, 'bb': {t['bb']}, 'tp': {t['tp']}, 'tp_hi': {t['tp_hi']}}},", flush=True)


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    main()
