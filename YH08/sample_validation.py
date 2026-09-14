# -*- coding: utf-8 -*-
"""YH08 样本外验证: 2020-2023 样本内调参, 2024-2026 样本外验证
用法:
    python sample_validation.py --stage in_global   # 样本内 全局卖出参数
    python sample_validation.py --stage in_rsi_bb   # 样本内 每标的 rsi/bb
    python sample_validation.py --stage in_tp       # 样本内 每标的 tp/tp_hi
    python sample_validation.py --stage oos         # 样本外验证(基线 vs 样本内最优)
"""
import os, json, itertools, time, argparse
import pandas as pd
import numpy as np
import daily_signal as d
import scan_params as sp  # 复用 load/sim/metrics/fmt_row

BEST_IN = os.path.join(d.SCRIPT, '_scan_best_in.json')
IN_END = '2024-01-01'
REGIME = 'bull'   # 样本内扫描的牛熊状态 (main 中由 --regime 覆盖)


def load_split():
    raw, dfs, dates = sp.load()
    in_dates = [x for x in dates if x < pd.Timestamp(IN_END)]
    oos_dates = [x for x in dates if x >= pd.Timestamp(IN_END)]
    return raw, dfs, in_dates, oos_dates


def save_in(data):
    with open(BEST_IN, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def load_in():
    if os.path.exists(BEST_IN):
        with open(BEST_IN, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def base_buy():
    return sp.base_buy()


def stage_in_global(raw, dfs, in_dates):
    trails = [0.05, 0.08, 0.10, 0.12]
    hards = [0.10, 0.12, 0.15, 0.20]
    cds = [10, 20, 30, 40]
    bb = base_buy()
    rows = []
    t0 = time.time()
    for trail, hard, cd in itertools.product(trails, hards, cds):
        if hard <= trail:
            continue
        c, ann, mdd, ret, wr, nsell, td, spd = sp.sim(raw, dfs, in_dates, bb, trail, hard, cd)
        rows.append((c, ann, mdd, ret, wr, nsell, trail, hard, cd))
    rows.sort(key=lambda r: -r[0])
    print(f'\n[样本内 2020-2023 全局卖出] Top5 ({time.time()-t0:.0f}s):')
    for r in rows[:5]:
        c, ann, mdd, ret, wr, nsell, trail, hard, cd = r
        print('  ' + sp.fmt_row(f'TS{trail:.0%}/HS{hard:.0%}/CD{cd}', c, ann, mdd, ret, wr, nsell))
    best = rows[0]
    _, _, _, _, _, _, trail, hard, cd = best
    save_in({'global': {'trail': trail, 'hard': hard, 'cooldown': cd}})
    print(f'  >>> 样本内全局最优: TS{trail:.0%}/HS{hard:.0%}/CD{cd} Calmar {best[0]:.2f}')
    return trail, hard, cd


def stage_in_rsi_bb(raw, dfs, in_dates, trail, hard, cd):
    rsis = [25, 30, 35, 40, 45, 50, 55]
    bbs = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    saved = load_in()
    merged = saved.get('rsi_bb', {})
    merged.setdefault(REGIME, {})
    t0 = time.time()
    for name in d.CORE_STOCKS:
        rows = []
        for rsi, b in itertools.product(rsis, bbs):
            bp = base_buy()
            bp[REGIME][name]['rsi'] = rsi
            bp[REGIME][name]['bb'] = b
            c, ann, mdd, ret, wr, nsell, td, spd = sp.sim(raw, dfs, in_dates, bp, trail, hard, cd)
            rows.append((c, ann, mdd, ret, wr, nsell, rsi, b, spd[name]))
        rows.sort(key=lambda r: -r[0])
        best = rows[0]
        merged[REGIME][name] = {'rsi': int(best[6]), 'bb': float(best[7])}
        print(f'  [样本内 {name}] 最优 rsi{best[6]} bb{best[7]} (该股{best[8]/10000:+.1f}w)')
        for r in rows[:3]:
            c, ann, mdd, ret, wr, nsell, rsi, b, pnl = r
            print('    ' + sp.fmt_row(f'rsi{rsi} bb{b}', c, ann, mdd, ret, wr, nsell))
    saved['global'] = {'trail': trail, 'hard': hard, 'cooldown': cd}
    saved['rsi_bb'] = merged
    save_in(saved)
    print(f'  ({time.time()-t0:.0f}s)')


def stage_in_tp(raw, dfs, in_dates, trail, hard, cd, best_rsi_bb):
    if REGIME == 'bull':
        tps = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
        tp_his = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    else:
        tps = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
        tp_his = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    saved = load_in()
    merged = saved.get('final', {})
    merged.setdefault(REGIME, {})
    t0 = time.time()
    for name in d.CORE_STOCKS:
        rsi = best_rsi_bb[REGIME][name]['rsi']
        b = best_rsi_bb[REGIME][name]['bb']
        rows = []
        for tp, tp_hi in itertools.product(tps, tp_his):
            if tp_hi <= tp:
                continue
            bp = base_buy()
            bp[REGIME][name]['rsi'] = rsi
            bp[REGIME][name]['bb'] = b
            bp[REGIME][name]['tp'] = tp
            bp[REGIME][name]['tp_hi'] = tp_hi
            c, ann, mdd, ret, wr, nsell, td, spd = sp.sim(raw, dfs, in_dates, bp, trail, hard, cd)
            rows.append((c, ann, mdd, ret, wr, nsell, tp, tp_hi, spd[name]))
        rows.sort(key=lambda r: -r[0])
        best = rows[0]
        merged[REGIME][name] = {'rsi': rsi, 'bb': b, 'tp': best[6], 'tp_hi': best[7]}
        print(f'  [样本内 {name}] 最优 tp{best[6]:.0%} tp_hi{best[7]:.0%} (该股{best[8]/10000:+.1f}w)')
    saved['final'] = merged
    save_in(saved)
    print(f'  ({time.time()-t0:.0f}s)')


def stage_oos(raw, dfs, oos_dates):
    b = load_in()
    if not b.get('final'):
        print('无样本内结果, 请先跑 in_global/in_rsi_bb/in_tp')
        return
    g = b['global']
    fin = b['final']
    print(f'\n[样本外 2024-2026 验证] (数据 {len(oos_dates)} 天, {oos_dates[0].date()} ~ {oos_dates[-1].date()})')

    # 基线 (当前默认参数)
    bb = base_buy()
    r = sp.sim(raw, dfs, oos_dates, bb, 0.08, 0.10, 20)
    print('\n  基线(默认参数):')
    print('  ' + sp.fmt_row('基线', *r[:6]))
    for n in d.ALL_STOCKS:
        print(f'      {n:<8} {r[7][n]/10000:>+7.1f}w')

    # 样本内最优
    opt = base_buy()
    for reg, stocks in fin.items():
        for n, p in stocks.items():
            opt[reg][n].update(p)
    r2 = sp.sim(raw, dfs, oos_dates, opt, g['trail'], g['hard'], g['cooldown'])
    print('\n  样本内最优参数(2020-2023调出):')
    print(f'    TS{g["trail"]:.0%}/HS{g["hard"]:.0%}/CD{g["cooldown"]}')
    for reg in ('bull', 'bear'):
        if reg in fin:
            print(f'    [{reg}] ' + ' '.join(f'{n}:{fin[reg][n]}' for n in d.CORE_STOCKS if n in fin[reg]))
    print('  ' + sp.fmt_row('样本内最优', *r2[:6]))
    for n in d.ALL_STOCKS:
        print(f'      {n:<8} {r2[7][n]/10000:>+7.1f}w')


def main():
    global REGIME
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='in_global', choices=['in_global', 'in_rsi_bb', 'in_tp', 'oos'])
    ap.add_argument('--regime', default='bull', choices=['bull', 'bear'], help='样本内扫描哪个牛熊状态')
    a = ap.parse_args()
    REGIME = a.regime
    raw, dfs, in_dates, oos_dates = load_split()
    print(f'[数据] 样本内 {len(in_dates)} 天 ({in_dates[0].date()}~{in_dates[-1].date()})  '
          f'样本外 {len(oos_dates)} 天 ({oos_dates[0].date()}~{oos_dates[-1].date()})')

    if a.stage == 'in_global':
        stage_in_global(raw, dfs, in_dates)
    elif a.stage == 'in_rsi_bb':
        saved = load_in()
        g = saved.get('global', {'trail': 0.10, 'hard': 0.12, 'cooldown': 40})
        stage_in_rsi_bb(raw, dfs, in_dates, g['trail'], g['hard'], g['cooldown'])
    elif a.stage == 'in_tp':
        saved = load_in()
        g = saved.get('global', {'trail': 0.10, 'hard': 0.12, 'cooldown': 40})
        rb = saved.get('rsi_bb', {REGIME: {n: {'rsi': 35, 'bb': 0.12} for n in d.CORE_STOCKS}})
        stage_in_tp(raw, dfs, in_dates, g['trail'], g['hard'], g['cooldown'], rb)
    elif a.stage == 'oos':
        stage_oos(raw, dfs, oos_dates)


if __name__ == '__main__':
    main()
