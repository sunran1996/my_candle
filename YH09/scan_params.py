# -*- coding: utf-8 -*-
"""YH08 参数网格扫描 (目标: 收益/回撤比 Calmar 最高, 加DCA)
阶段结果持久化到 _scan_best.json, 各阶段可独立/顺序运行:
    python scan_params.py --stage global    # 阶段1: 全局卖出参数
    python scan_params.py --stage rsi_bb    # 阶段2a: 每标的 rsi/bb
    python scan_params.py --stage tp        # 阶段2b: 每标的 tp/tp_hi
    python scan_params.py --stage all       # 顺序全跑
    python scan_params.py --regime bear     # 加 --regime bull|bear 指定扫哪个牛熊状态
"""
import sys, io, os, itertools, pickle, time, argparse, json
import pandas as pd
import daily_signal as d  # 已在其内部将 stdout 包装为 UTF-8

CACHE = os.path.join(d.SCRIPT, '_scan_cache.pkl')
BEST = os.path.join(d.SCRIPT, '_scan_best.json')
START = '2020-01-01'
REGIME = 'bull'   # 扫描的牛熊状态 (main 中由 --regime 覆盖)


def load():
    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            raw, dfs = pickle.load(f)
        print(f'[数据] 缓存加载 ({time.strftime("%H:%M", time.localtime(os.path.getmtime(CACHE)))})')
    else:
        print('[数据] 拉取并缓存...')
        raw = d.fetch()
        dfs = {n: d.add_indicators(x) for n, x in raw.items()}
        with open(CACHE, 'wb') as f:
            pickle.dump((raw, dfs), f)
    dates = sorted(set.intersection(*[set(x['date']) for x in dfs.values()]))
    dates = [x for x in dates if x >= pd.Timestamp(START)]
    return raw, dfs, dates


def save_best(data):
    with open(BEST, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def load_best():
    if os.path.exists(BEST):
        with open(BEST, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def base_buy():
    """嵌套参数快照: {regime: {name: {rsi,bb,tp,tp_hi}}} (深拷贝, 供扫描按 regime 覆盖)"""
    return {reg: {n: dict(d.BUY_PARAMS[reg][n]) for n in d.ALL_STOCKS}
            for reg in ('bull', 'bear')}


def metrics(ndf, total_injected, td):
    final = ndf['nav'].iloc[-1]
    ret = final / total_injected - 1
    ann = (1 + ret) ** (252 / len(ndf)) - 1
    mdd = ((ndf['nav'] - ndf['nav'].cummax()) / ndf['nav'].cummax()).min()
    calmar = ann / abs(mdd)
    sells = td[td['dir'] == 'SELL']
    wr = (sells['pnl'] > 0).mean() * 100 if len(sells) else 0
    return calmar, ann * 100, mdd * 100, ret * 100, wr, len(sells)


def sim(raw, dfs, dates, buy_params, trail, hard, cooldown):
    d.BUY_PARAMS = buy_params
    d.TRAIL_STOP = trail
    d.HARD_STOP = hard
    d.COOLDOWN = cooldown
    ndf, td, total_injected, stock_pnl = d._simulate(raw, dfs, dates, True)
    c, ann, mdd, ret, wr, nsell = metrics(ndf, total_injected, td)
    return c, ann, mdd, ret, wr, nsell, td, stock_pnl


def fmt_row(tag, c, ann, mdd, ret, wr, nsell):
    return f'{tag:<26} Calmar{c:>6.2f}  年化{ann:>+6.1f}%  回撤{mdd:>+6.1f}%  累计{ret:>+7.1f}%  胜率{wr:>4.0f}%  卖{nsell:>3}笔'


def baseline(raw, dfs, dates):
    bp = base_buy()
    c, ann, mdd, ret, wr, nsell, td, sp = sim(raw, dfs, dates, bp, d.TRAIL_STOP, d.HARD_STOP, d.COOLDOWN)
    print('\n[基线] 当前默认参数:')
    print('  ' + fmt_row('默认', c, ann, mdd, ret, wr, nsell))
    for n in d.ALL_STOCKS:
        print(f'    {n:<8} {sp[n]/10000:>+7.1f}w')


def stage_global(raw, dfs, dates):
    baseline(raw, dfs, dates)
    trails = [0.05, 0.08, 0.10, 0.12]
    hards = [0.10, 0.12, 0.15, 0.20]
    cooldowns = [10, 20, 30, 40]
    bb = base_buy()
    print('\n[阶段1] 全局卖出参数扫描 (trail × hard × cooldown):')
    rows = []
    t0 = time.time()
    for trail, hard, cd in itertools.product(trails, hards, cooldowns):
        if hard <= trail:
            continue
        c, ann, mdd, ret, wr, nsell, td, sp = sim(raw, dfs, dates, bb, trail, hard, cd)
        rows.append((c, ann, mdd, ret, wr, nsell, trail, hard, cd))
    rows.sort(key=lambda r: -r[0])
    print(f'  共{len(rows)}组, 耗时{time.time()-t0:.0f}s')
    for r in rows[:10]:
        c, ann, mdd, ret, wr, nsell, trail, hard, cd = r
        print('  ' + fmt_row(f'TS{trail:.0%}/HS{hard:.0%}/CD{cd}', c, ann, mdd, ret, wr, nsell))
    best = rows[0]
    _, _, _, _, _, _, trail, hard, cd = best
    save_best({'global': {'trail': trail, 'hard': hard, 'cooldown': cd}})
    print(f'\n  >>> 阶段1最优: TRAIL_STOP={trail} HARD_STOP={hard} COOLDOWN={cd} (Calmar {best[0]:.2f})')
    return best


def stage_rsi_bb(raw, dfs, dates, trail, hard, cd, only=None):
    names = [only] if only else list(d.CORE_STOCKS)
    print(f'\n[阶段2a] {REGIME} rsi×bb (TS{trail:.0%}/HS{hard:.0%}/CD{cd}, 标的={names}):')
    rsis = [25, 30, 35, 40, 45, 50, 55]
    bbs = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    saved = load_best()
    merged = saved.get('rsi_bb', {})
    merged.setdefault(REGIME, {})
    t0 = time.time()
    for name in names:
        rows = []
        for rsi, bb in itertools.product(rsis, bbs):
            bp = base_buy()
            bp[REGIME][name]['rsi'] = rsi
            bp[REGIME][name]['bb'] = bb
            c, ann, mdd, ret, wr, nsell, td, sp = sim(raw, dfs, dates, bp, trail, hard, cd)
            rows.append((c, ann, mdd, ret, wr, nsell, rsi, bb, sp[name]))
        rows.sort(key=lambda r: -r[0])
        best = rows[0]
        merged[REGIME][name] = {'rsi': int(best[6]), 'bb': float(best[7])}
        print(f'\n  [{name}] Top5 (最优该股 {best[8]/10000:+.1f}w):')
        for r in rows[:5]:
            c, ann, mdd, ret, wr, nsell, rsi, bb, pnl = r
            print('  ' + fmt_row(f'rsi{rsi} bb{bb}', c, ann, mdd, ret, wr, nsell) + f'  该股{pnl/10000:+.1f}w')
    saved['global'] = {'trail': trail, 'hard': hard, 'cooldown': cd}
    saved['rsi_bb'] = merged
    save_best(saved)
    print(f'\n  阶段2a 耗时{time.time()-t0:.0f}s')
    return merged


def stage_tp(raw, dfs, dates, trail, hard, cd, best_rsi_bb, only=None):
    names = [only] if only else list(d.CORE_STOCKS)
    print(f'\n[阶段2b] {REGIME} tp×tp_hi (TS{trail:.0%}/HS{hard:.0%}/CD{cd}, 标的={names}):')
    if REGIME == 'bull':
        # 牛市止盈偏低(上轮全压0.10下边界), 下边界扩到0.05
        tps = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
        tp_his = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    else:
        # 熊市止盈偏高(上轮全压0.25上边界), 上边界扩到0.40
        tps = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
        tp_his = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    saved = load_best()
    merged = saved.get('final', {})
    merged.setdefault(REGIME, {})
    t0 = time.time()
    for name in names:
        rsi = best_rsi_bb[REGIME][name]['rsi']
        bb = best_rsi_bb[REGIME][name]['bb']
        rows = []
        for tp, tp_hi in itertools.product(tps, tp_his):
            if tp_hi <= tp:
                continue
            bp = base_buy()
            bp[REGIME][name]['rsi'] = rsi
            bp[REGIME][name]['bb'] = bb
            bp[REGIME][name]['tp'] = tp
            bp[REGIME][name]['tp_hi'] = tp_hi
            c, ann, mdd, ret, wr, nsell, td, sp = sim(raw, dfs, dates, bp, trail, hard, cd)
            rows.append((c, ann, mdd, ret, wr, nsell, tp, tp_hi, sp[name]))
        rows.sort(key=lambda r: -r[0])
        best = rows[0]
        merged[REGIME][name] = {'rsi': rsi, 'bb': bb, 'tp': best[6], 'tp_hi': best[7]}
        print(f'\n  [{name}] rsi{rsi} bb{bb} Top5:')
        for r in rows[:5]:
            c, ann, mdd, ret, wr, nsell, tp, tp_hi, pnl = r
            print('  ' + fmt_row(f'tp{tp:.0%} tp_hi{tp_hi:.0%}', c, ann, mdd, ret, wr, nsell) + f'  该股{pnl/10000:+.1f}w')
    saved['final'] = merged
    save_best(saved)
    print(f'\n  阶段2b 耗时{time.time()-t0:.0f}s')
    return merged


def main():
    global REGIME
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='global', choices=['global', 'rsi_bb', 'tp', 'all'])
    ap.add_argument('--name', default=None, help='rsi_bb/tp 阶段只扫该标的')
    ap.add_argument('--regime', default='bull', choices=['bull', 'bear'], help='扫描哪个牛熊状态')
    a = ap.parse_args()
    REGIME = a.regime
    raw, dfs, dates = load()
    print(f'[数据] {len(dates)} 个交易日 ({dates[0].date()} ~ {dates[-1].date()})  扫描{REGIME}')

    saved = load_best()
    g = saved.get('global', {'trail': 0.10, 'hard': 0.12, 'cooldown': 40})  # 阶段1结论
    trail, hard, cd = g['trail'], g['hard'], g['cooldown']

    if a.stage in ('global', 'all'):
        best = stage_global(raw, dfs, dates)
        _, _, _, _, _, _, trail, hard, cd = best

    best_rsi_bb = saved.get('rsi_bb')
    if a.stage in ('rsi_bb', 'all'):
        merged = stage_rsi_bb(raw, dfs, dates, trail, hard, cd, a.name)
        best_rsi_bb = merged

    if a.stage in ('tp', 'all'):
        if not best_rsi_bb:
            best_rsi_bb = {REGIME: {n: {'rsi': 35, 'bb': 0.12} for n in d.CORE_STOCKS}}
        stage_tp(raw, dfs, dates, trail, hard, cd, best_rsi_bb, a.name)


if __name__ == '__main__':
    main()
