# -*- coding: utf-8 -*-
"""YH08 创业板(ETF) tp×tp_hi 牛熊分状态调优 + 样本外验证
创业板为 MACD 驱动买入(无 RSI/BB 参数), 仅卖出止盈 tp/tp_hi 两参数按牛熊各一套可调。
复用 scan_params 的缓存/模拟/指标, 与核心股调优保持同一 Calmar 口径。

用法:
    python scan_cyber.py            # 全区间(2020起) 扫 bull+bear, 结果存 _scan_cyber_best.json
    python scan_cyber.py --oos      # 样本外验证: 样本内(2020-2023)调 -> 样本外(2024-2026)验
"""
import os, json, itertools, time, argparse
import daily_signal as d
import scan_params as sp
import sample_validation as sv

NAME = '创业板'
BEST = os.path.join(d.SCRIPT, '_scan_cyber_best.json')

# 与核心股 stage_tp 相同的网格 (牛市低止盈 / 熊市高止盈)
GRIDS = {
    'bull': ([0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
             [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]),
    'bear': ([0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
             [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]),
}


def scan_regime(raw, dfs, dates, regime, trail, hard, cd):
    tps, tp_his = GRIDS[regime]
    rows = []
    t0 = time.time()
    for tp, tp_hi in itertools.product(tps, tp_his):
        if tp_hi <= tp:
            continue
        bp = sp.base_buy()
        bp[regime][NAME]['tp'] = tp
        bp[regime][NAME]['tp_hi'] = tp_hi
        c, ann, mdd, ret, wr, nsell, td, spd = sp.sim(raw, dfs, dates, bp, trail, hard, cd)
        rows.append((c, ann, mdd, ret, wr, nsell, tp, tp_hi, spd[NAME]))
    rows.sort(key=lambda r: -r[0])
    print(f'\n[{regime}] 创业板 tp×tp_hi 扫描 (TS{trail:.0%}/HS{hard:.0%}/CD{cd}, {len(rows)}组, {time.time()-t0:.0f}s):')
    for r in rows[:8]:
        c, ann, mdd, ret, wr, nsell, tp, tp_hi, pnl = r
        print('  ' + sp.fmt_row(f'tp{tp:.0%} tp_hi{tp_hi:.0%}', c, ann, mdd, ret, wr, nsell) + f'  创业板{pnl/10000:+.1f}w')
    best = rows[0]
    return {'tp': best[6], 'tp_hi': best[7]}, best


def scan_all(raw, dfs, dates):
    print('=' * 70)
    print('  YH08 创业板(ETF) 牛熊分状态 tp/tp_hi 调优 (全区间 2020 起)')
    print('=' * 70)
    saved = sp.load_best()
    g = saved.get('global', {'trail': d.TRAIL_STOP, 'hard': d.HARD_STOP, 'cooldown': d.COOLDOWN})
    trail, hard, cd = g['trail'], g['hard'], g['cooldown']

    bp = sp.base_buy()
    c, ann, mdd, ret, wr, nsell, td, spd = sp.sim(raw, dfs, dates, bp, trail, hard, cd)
    print('\n[基线] 创业板当前 tp/tp_hi (牛/熊均为默认 0.10/0.15):')
    print('  ' + sp.fmt_row('当前', c, ann, mdd, ret, wr, nsell) + f'  创业板{spd[NAME]/10000:+.1f}w')

    result = {}
    for regime in ('bull', 'bear'):
        best_param, best_row = scan_regime(raw, dfs, dates, regime, trail, hard, cd)
        result[regime] = best_param
        print(f'  >>> {regime} 最优: tp{best_param["tp"]:.0%} tp_hi{best_param["tp_hi"]:.0%}')

    bp_opt = sp.base_buy()
    for regime in ('bull', 'bear'):
        bp_opt[regime][NAME]['tp'] = result[regime]['tp']
        bp_opt[regime][NAME]['tp_hi'] = result[regime]['tp_hi']
    c, ann, mdd, ret, wr, nsell, td, spd = sp.sim(raw, dfs, dates, bp_opt, trail, hard, cd)
    print('\n[组合] bull+bear 同时应用创业板最优 tp/tp_hi:')
    print('  ' + sp.fmt_row('调优后', c, ann, mdd, ret, wr, nsell) + f'  创业板{spd[NAME]/10000:+.1f}w')

    with open(BEST, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f'\n  结果已存 {BEST}')
    return result


def oos():
    raw, dfs, in_dates, oos_dates = sv.load_split()
    saved_in = sv.load_in()
    g = saved_in.get('global', {'trail': 0.10, 'hard': 0.12, 'cooldown': 40})
    trail, hard, cd = g['trail'], g['hard'], g['cooldown']
    print('=' * 70)
    print('  YH08 创业板 样本外验证 (样本内 2020-2023 调 -> 样本外 2024-2026 验)')
    print('=' * 70)

    result = {}
    for regime in ('bull', 'bear'):
        best_param, _ = scan_regime(raw, dfs, in_dates, regime, trail, hard, cd)
        result[regime] = best_param

    print('\n[样本外 2024-2026 验证]')
    bp_base = sp.base_buy()
    r0 = sp.sim(raw, dfs, oos_dates, bp_base, trail, hard, cd)
    print('  基线(创业板默认 0.10/0.15):')
    print('  ' + sp.fmt_row('基线', *r0[:6]) + f'  创业板{r0[7][NAME]/10000:+.1f}w')

    bp_opt = sp.base_buy()
    for regime in ('bull', 'bear'):
        bp_opt[regime][NAME]['tp'] = result[regime]['tp']
        bp_opt[regime][NAME]['tp_hi'] = result[regime]['tp_hi']
    r1 = sp.sim(raw, dfs, oos_dates, bp_opt, trail, hard, cd)
    print('  样本内最优创业板参数:')
    print(f'    bull: tp{result["bull"]["tp"]:.0%} tp_hi{result["bull"]["tp_hi"]:.0%}  '
          f'bear: tp{result["bear"]["tp"]:.0%} tp_hi{result["bear"]["tp_hi"]:.0%}')
    print('  ' + sp.fmt_row('样本内最优', *r1[:6]) + f'  创业板{r1[7][NAME]/10000:+.1f}w')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oos', action='store_true')
    a = ap.parse_args()
    if a.oos:
        oos()
    else:
        raw, dfs, dates = sp.load()
        print(f'[数据] {len(dates)} 个交易日 ({dates[0].date()} ~ {dates[-1].date()})')
        scan_all(raw, dfs, dates)


if __name__ == '__main__':
    main()
