# -*- coding: utf-8 -*-
"""
YH02 Walk-Forward 滚动窗口验证
固定参数(不优化), 测试样本外稳健性
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import akshare as ak, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import os, warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

INITIAL_CAPITAL = 1_000_000; RESERVE = 100_000
COMMISSION = 0.0001; SLIPPAGE = 0.0001; ETF_SYMBOL = 'sh512890'
BB_PERIOD = 45; BB_STD = 2.0
RSI_PERIOD = 14; RSI_OVERSOLD = 30; RSI_OVERBOUGHT = 70
EXPAND_RSI_SELL = 65; BB_ACCEL_UP = 0.001; DCA = 0

TRAIN_YEARS = 2; TEST_YEARS = 1

def compute_indicators(df):
    ret = df['close'].pct_change().fillna(0); ret[abs(ret) > 0.1] = 0
    df = df.copy()
    df['adj_close'] = (1 + ret).cumprod()
    df['ma'] = df['adj_close'].rolling(BB_PERIOD).mean()
    df['std'] = df['adj_close'].rolling(BB_PERIOD).std()
    df['upper'] = df['ma'] + BB_STD * df['std']
    df['lower'] = df['ma'] - BB_STD * df['std']
    df['upper_vel'] = df['upper'].diff()
    df['upper_acc'] = df['upper_vel'].diff().rolling(3, min_periods=1).mean()
    df['price_vel'] = df['adj_close'].diff()
    df['price_acc'] = df['price_vel'].diff().rolling(3, min_periods=1).mean()
    delta = df['adj_close'].diff(); gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
    df['rsi'] = 100 - 100 / (1 + gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean() /
                             loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean().replace(0, np.nan))
    return df

def run_backtest(df, start_date, end_date):
    sub = df[(df['date'] >= start_date) & (df['date'] <= end_date)].reset_index(drop=True)
    if len(sub) < 60: return None
    cash = 0.0; total_invested = INITIAL_CAPITAL
    reserve_pool = RESERVE; reserve_active = False; reserve_shares = 0.0
    shares = INITIAL_CAPITAL / sub['close'].iloc[0] * (1 - COMMISSION - SLIPPAGE)
    peak_nav = INITIAL_CAPITAL
    prev_price = None; last_month = None; signal_count = 0; prev_bb_width = None
    nav_log = []

    for i, row in sub.iterrows():
        price = row['close']; d = row['date']; adj = row['adj_close']
        if prev_price and prev_price > 0 and abs(price/prev_price - 1) > 0.1:
            shares *= (prev_price / price)
        prev_price = price
        ym = (d.year, d.month)
        if last_month and ym != last_month: cash += DCA; total_invested += DCA
        last_month = ym
        lower = row['lower']; upper = row['upper']; rsi = row['rsi']; ma = row['ma']
        if np.isnan(lower) or np.isnan(rsi):
            nav_log.append(cash + (shares + reserve_shares) * price); continue

        nav = cash + (shares + reserve_shares) * price
        if nav > peak_nav: peak_nav = nav
        dd = (nav - peak_nav) / peak_nav if peak_nav > 0 else 0
        if dd < -0.08 and not reserve_active and reserve_pool > 0:
            cash += reserve_pool; reserve_pool = 0; reserve_active = True

        bb_buy = (adj <= lower); bb_sell = (adj >= upper)
        rsi_buy = (rsi <= RSI_OVERSOLD)
        bb_width = (upper - lower) / ma if ma > 0 else 0.1
        expanding = (prev_bb_width is not None and bb_width > prev_bb_width)
        prev_bb_width = bb_width

        if expanding:
            buy_sig = (bb_buy or rsi_buy)
            raw_sell = (bb_sell and rsi >= EXPAND_RSI_SELL)
            upper_acc = row['upper_acc'] if not np.isnan(row['upper_acc']) else 0
            price_acc = row['price_acc'] if not np.isnan(row['price_acc']) else 0
            blocked = (upper_acc > BB_ACCEL_UP) and (price_acc > 0)
            sell_sig = raw_sell and not blocked
        else:
            buy_sig = (bb_buy or rsi_buy)
            sell_sig = (bb_sell or (rsi >= RSI_OVERBOUGHT))

        if buy_sig or sell_sig: signal_count += 1
        if signal_count < 2:
            nav_log.append(nav); continue

        if buy_sig:
            if cash > nav * 0.01:
                bs = cash / price * (1 - COMMISSION - SLIPPAGE)
                if reserve_active:
                    reserve_shares += bs * (RESERVE / (INITIAL_CAPITAL + RESERVE))
                    shares += bs - bs * (RESERVE / (INITIAL_CAPITAL + RESERVE))
                else: shares += bs
                cash = 0
        elif sell_sig:
            if shares > 0 or reserve_shares > 0:
                cash += (shares + reserve_shares) * price * (1 - COMMISSION - SLIPPAGE)
                if reserve_active:
                    reserve_pool = reserve_shares * price * (1 - COMMISSION - SLIPPAGE)
                    cash -= reserve_pool; reserve_shares = 0; reserve_active = False
                shares = 0
        nav_log.append(cash + (shares + reserve_shares) * price)

    last_price = sub['close'].iloc[-1]; final_nav = cash + (shares + reserve_shares) * last_price
    total_ret = (final_nav / total_invested - 1) * 100
    nav_series = pd.Series(nav_log)
    n_days = len(nav_log)
    ann_ret = ((1 + total_ret/100) ** (252 / n_days) - 1) * 100 if n_days > 0 else 0
    daily_rets = nav_series.pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) * 100
    sharpe = (ann_ret/100 - 0.02) / (ann_vol/100) if ann_vol > 0 else 0
    sortino = (ann_ret/100 - 0.02) / (daily_rets[daily_rets < 0].std() * np.sqrt(252)) if len(daily_rets[daily_rets < 0]) > 0 else 0
    cum = nav_series / INITIAL_CAPITAL
    mdd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    calmar = (ann_ret/100) / abs(mdd/100) if mdd != 0 else 0
    return {'total_ret': total_ret, 'ann_ret': ann_ret, 'sharpe': sharpe,
            'sortino': sortino, 'mdd': mdd, 'calmar': calmar,
            'final_nav': final_nav, 'total_invested': total_invested, 'nav_series': nav_series}


def generate_windows(df, train_years, test_years):
    min_date = df['date'].min(); max_date = df['date'].max()
    windows = []
    window_start = pd.Timestamp(year=min_date.year, month=min_date.month, day=min_date.day)
    while True:
        train_end = window_start + pd.DateOffset(years=train_years) - pd.DateOffset(days=1)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(years=test_years) - pd.DateOffset(days=1)
        if test_end > max_date:
            if test_start < max_date:
                windows.append({'train_start': window_start, 'train_end': train_end,
                                'test_start': test_start, 'test_end': max_date})
            break
        windows.append({'train_start': window_start, 'train_end': train_end,
                        'test_start': test_start, 'test_end': test_end})
        window_start = test_start
    return windows


def main():
    print("=" * 70)
    print(f"  YH02 Walk-Forward 固定参数验证")
    print(f"  BB({BB_PERIOD},{BB_STD}) RSI({RSI_PERIOD},{RSI_OVERSOLD},{RSI_OVERBOUGHT})")
    print(f"  扩张卖出: BB∩RSI≥{EXPAND_RSI_SELL}")
    print(f"  窗口: {TRAIN_YEARS}年训练 → {TEST_YEARS}年测试 (不优化, 纯OOS)")
    print("=" * 70)

    # 数据
    print("\n[1/3] 数据...")
    df = ak.fund_etf_hist_sina(symbol=ETF_SYMBOL)
    df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date').reset_index(drop=True)
    df = compute_indicators(df)
    min_d = df['date'].min().strftime('%Y-%m-%d'); max_d = df['date'].max().strftime('%Y-%m-%d')
    print(f"  {min_d} ~ {max_d}  ({len(df)}天)")

    # 全区间基准
    print("\n[2/3] 全区间基准...")
    base = run_backtest(df, min_d, max_d)
    print(f"  收益={base['total_ret']:+.1f}%  年化={base['ann_ret']:+.1f}%  Sharpe={base['sharpe']:.3f}  "
          f"Sortino={base['sortino']:.3f}  MaxDD={base['mdd']:+.1f}%")

    # 滚动窗口
    print(f"\n[3/3] 滚动窗口 OOS 测试...")
    windows = generate_windows(df, TRAIN_YEARS, TEST_YEARS)
    print(f"  共 {len(windows)} 个窗口:")
    for i, w in enumerate(windows):
        print(f"    W{i+1}: 训练 [{w['train_start'].strftime('%Y-%m-%d')} ~ {w['train_end'].strftime('%Y-%m-%d')}]"
              f" → 测试 [{w['test_start'].strftime('%Y-%m-%d')} ~ {w['test_end'].strftime('%Y-%m-%d')}]")

    print(f"\n  ┌─ OOS结果 ──────────────────────────────────")
    print(f"  │ {'窗':>3} {'测试期':<22} {'收益':>8} {'年化':>7} {'Sharpe':>7} {'Sortino':>7} {'MaxDD':>7} {'Calmar':>6}")
    print(f"  │ {'─'*70}")

    is_results = []; oos_results = []
    for i, w in enumerate(windows):
        train_s = w['train_start'].strftime('%Y-%m-%d')
        train_e = w['train_end'].strftime('%Y-%m-%d')
        test_s = w['test_start'].strftime('%Y-%m-%d')
        test_e = w['test_end'].strftime('%Y-%m-%d')

        is_r = run_backtest(df, train_s, train_e)
        oos_r = run_backtest(df, test_s, test_e)

        if is_r: is_results.append(is_r)
        if oos_r: oos_results.append(oos_r)

        if oos_r:
            qual = '✓' if oos_r['sharpe'] > 1.0 else ('△' if oos_r['sharpe'] > 0.5 else '⚠')
            print(f"  │ {i+1:>3} {test_s}~{test_e:<11} {oos_r['total_ret']:>+7.1f}% {oos_r['ann_ret']:>+6.1f}% "
                  f"{oos_r['sharpe']:>7.3f} {oos_r['sortino']:>7.3f} {oos_r['mdd']:>+6.1f}% {oos_r['calmar']:>6.2f} {qual}")

    # 汇总
    if not oos_results:
        print("  无有效OOS结果!"); return

    oos_sharpes = [r['sharpe'] for r in oos_results]
    oos_rets = [r['total_ret'] for r in oos_results]
    oos_mdds = [r['mdd'] for r in oos_results]
    is_sharpes = [r['sharpe'] for r in is_results]
    is_rets = [r['total_ret'] for r in is_results]

    print(f"\n  ┌─ 汇总 ─────────────────────────────────────")
    print(f"  │ {'':<16} {'训练集(IS)':>12} {'测试集(OOS)':>12} {'IS→OOS':>10}")
    print(f"  │ {'─'*50}")
    print(f"  │ {'平均Sharpe':<16} {np.mean(is_sharpes):>12.2f} {np.mean(oos_sharpes):>12.2f} "
          f"{np.mean(oos_sharpes)-np.mean(is_sharpes):>+9.2f}")
    print(f"  │ {'最差Sharpe':<16} {np.min(is_sharpes):>12.2f} {np.min(oos_sharpes):>12.2f} "
          f"{np.min(oos_sharpes)-np.min(is_sharpes):>+9.2f}")
    print(f"  │ {'平均收益':<16} {np.mean(is_rets):>+11.1f}% {np.mean(oos_rets):>+11.1f}% "
          f"{np.mean(oos_rets)-np.mean(is_rets):>+9.1f}%")
    print(f"  │ {'平均MaxDD':<16} {np.mean([r['mdd'] for r in is_results]):>+11.1f}% "
          f"{np.mean(oos_mdds):>+11.1f}% {np.mean(oos_mdds)-np.mean([r['mdd'] for r in is_results]):>+9.1f}%")

    # 过拟合诊断
    sharpe_decay = (np.mean(is_sharpes) - np.mean(oos_sharpes)) / np.mean(is_sharpes) * 100 if np.mean(is_sharpes) != 0 else 0
    oos_sharpe_cv = np.std(oos_sharpes) / abs(np.mean(oos_sharpes)) * 100 if np.mean(oos_sharpes) != 0 else 0
    pos_wins = sum(1 for r in oos_rets if r > 0)

    print(f"\n  ┌─ 诊断 ─────────────────────────────────────")
    print(f"  │ Sharpe衰减: {sharpe_decay:.1f}%", end='')
    if sharpe_decay < 0:        print('  ★ OOS反超IS!')
    elif sharpe_decay < 10:     print('  ✓ 优秀')
    elif sharpe_decay < 20:     print('  △ 可接受')
    else:                       print('  ⚠ 过拟合')

    print(f"  │ OOS Sharpe CV: {oos_sharpe_cv:.1f}%", end='')
    if oos_sharpe_cv < 20:      print('  ✓ 稳定')
    elif oos_sharpe_cv < 40:    print('  △ 可接受')
    else:                       print('  ⚠ 波动大')

    print(f"  │ OOS正收益: {pos_wins}/{len(oos_rets)}", end='')
    if pos_wins == len(oos_rets): print('  ✓ 全部盈利')
    else:                         print('  ⚠ 有亏损窗口')

    # OOS累计净值
    wf_cum = 1.0
    for r in oos_rets: wf_cum *= (1 + r/100)
    wf_ret = (wf_cum - 1) * 100
    wf_yrs = len(oos_rets) * TEST_YEARS
    wf_ann = ((1 + wf_ret/100) ** (1/wf_yrs) - 1) * 100 if wf_yrs > 0 else 0
    print(f"  │ WF-OOS累计: {wf_ret:+.1f}%  年化: {wf_ann:+.1f}%  ({wf_yrs}年)")

    score = 100
    score -= min(max(sharpe_decay, 0) * 1.5, 40)
    score -= min(oos_sharpe_cv * 0.5, 20)
    if pos_wins < len(oos_rets): score -= (len(oos_rets) - pos_wins) * 10
    if sharpe_decay < 0: score = min(100, score + 15)
    score = max(0, min(100, score))
    grade = 'A' if score >= 80 else ('B' if score >= 60 else ('C' if score >= 40 else 'D'))
    print(f"  │ 综合: {score:.0f}/100  ({'A' if score >= 80 else ('B' if score >= 60 else ('C' if score >= 40 else 'D'))})")

    # 画图
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
        fig.subplots_adjust(wspace=0.25, top=0.88)

        # P1: IS vs OOS Sharpe
        ax = axes[0]
        ax.set_title('训练集 vs 测试集 Sharpe', fontsize=12, fontweight='bold')
        wins = range(1, len(oos_results)+1)
        ax.bar([w-0.2 for w in wins], is_sharpes, 0.35, color='#e74c3c', alpha=0.7, label='训练集(IS)')
        ax.bar([w+0.2 for w in wins], oos_sharpes, 0.35, color='#27ae60', alpha=0.7, label='测试集(OOS)')
        ax.axhline(y=1.0, color='black', lw=0.8, ls='--', alpha=0.3)
        ax.legend(fontsize=9); ax.set_xticks(list(wins))
        ax.set_xticklabels([f'W{i}' for i in wins])
        ax.set_ylabel('Sharpe'); ax.grid(True, alpha=0.2, axis='y')
        for i, (is_v, oos_v) in enumerate(zip(is_sharpes, oos_sharpes)):
            ax.text(i+1, max(is_v, oos_v) + 0.05, f'{oos_v:.2f}', ha='center', fontsize=8, color='#27ae60')

        # P2: OOS净值累积
        ax = axes[1]
        ax.set_title(f'Walk-Forward OOS 净值 ({wf_ret:+.1f}%)', fontsize=12, fontweight='bold')
        cum_nav = [1.0]
        for r in oos_rets: cum_nav.append(cum_nav[-1] * (1 + r/100))
        ax.fill_between(range(len(cum_nav)), 1.0, cum_nav, alpha=0.08, color='#27ae60')
        ax.plot(range(len(cum_nav)), cum_nav, 'o-', color='#27ae60', lw=2.5, markersize=10)
        # 全区间基准归一化
        b_nav = base['nav_series'] / INITIAL_CAPITAL
        b_idx = np.linspace(0, len(b_nav)-1, len(cum_nav)).astype(int)
        b_norm = [b_nav.iloc[i] for i in b_idx]
        ax.plot(range(len(cum_nav)), b_norm, '--', color='#e74c3c', lw=1.5, alpha=0.6, label='全区间基准')
        ax.axhline(y=1.0, color='black', lw=0.8, ls='--')
        ax.legend(fontsize=8); ax.set_xticks(range(len(cum_nav)))
        ax.set_xticklabels(['Start'] + [f'W{i}' for i in wins])
        ax.set_ylabel('净值倍数'); ax.grid(True, alpha=0.2)

        fig.suptitle(f'YH02 Walk-Forward | BB({BB_PERIOD},{BB_STD}) RSI({RSI_PERIOD},{RSI_OVERSOLD},{RSI_OVERBOUGHT}) '
                     f'扩张RSI≥{EXPAND_RSI_SELL} | 评级{grade}({score:.0f}/100)',
                     fontsize=13, fontweight='bold')
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'walkforward_chart.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
        print(f"\n  图表: {out_path}")
    except Exception as e:
        print(f"  画图失败: {e}")


if __name__ == '__main__':
    main()
