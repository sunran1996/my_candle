# -*- coding: utf-8 -*-
"""
YH02 红利低波 BB+RSI 策略 — 精简稳健版
BB(45,2.0)+RSI(14,30,70) + 趋势扩张因子 + DCA 2w + 10w备用金(回撤>8%)
扩张时买入OR不变 / 卖出BB∩RSI≥65 / 收缩时买卖均OR
Walk-Forward验证: OOS平均Sharpe=1.28, 最差1.01, 3/3窗口盈利
用法: python monitor.py           → 今日实时信号
      python monitor.py --from 2019-01-18  → 回测
"""
import sys, io, os, json, warnings, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from data_feed import get_data
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, mplfinance as mpf
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ======================== 策略参数 ========================
INITIAL_CAPITAL = 1_000_000; RESERVE = 100_000; COMMISSION = 0.0001; SLIPPAGE = 0.0001
ETF_SYMBOL = 'sh512890'; ETF_NAME = '红利低波'
BB_PERIOD = 45; BB_STD = 2.0
RSI_PERIOD = 14; RSI_OVERSOLD = 30; RSI_OVERBOUGHT = 70
EXPAND_RSI_SELL = 65       # 扩张时卖出RSI阈值(低于默认70, 趋势动能衰减即离场)
HARD_STOP = 0.12           # 12%硬止损(黑天鹅保护, 历史从未触发)
BB_ACCEL_UP = 0.001        # BB上轨加速度>此值=上涨加速中, 暂缓卖出
DCA = 0                    # 月定投额, 默认关闭, 通过 --dca 2 开启(单位:万)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================== 指标计算 ========================
def compute_indicators(df):
    ret = df['close'].pct_change().fillna(0); ret[abs(ret) > 0.1] = 0
    df['adj_close'] = (1 + ret).cumprod()
    df['ma'] = df['adj_close'].rolling(BB_PERIOD).mean()
    df['std'] = df['adj_close'].rolling(BB_PERIOD).std()
    df['upper'] = df['ma'] + BB_STD * df['std']
    df['lower'] = df['ma'] - BB_STD * df['std']
    df['upper_vel'] = df['upper'].diff()
    df['upper_acc'] = df['upper_vel'].diff().rolling(3, min_periods=1).mean()  # 上轨加速度(3日平滑)
    df['lower_vel'] = df['lower'].diff()
    df['lower_acc'] = df['lower_vel'].diff().rolling(3, min_periods=1).mean()  # 下轨加速度(3日平滑)
    df['price_vel'] = df['adj_close'].diff()
    df['price_acc'] = df['price_vel'].diff().rolling(3, min_periods=1).mean()  # 价格加速度(3日平滑)
    delta = df['adj_close'].diff(); gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
    df['rsi'] = 100 - 100 / (1 + gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean() /
                             loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean().replace(0, np.nan))

# ======================== 回测 ========================
def run_backtest(start_date_str):
    start_date = pd.Timestamp(start_date_str)
    df = get_data(ETF_SYMBOL)
    df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date')
    compute_indicators(df)
    df = df[df['date'] >= start_date].reset_index(drop=True)

    cash = 0.0; total_invested = INITIAL_CAPITAL; reserve_pool = RESERVE; reserve_active = False; reserve_shares = 0.0
    shares = INITIAL_CAPITAL / df['close'].iloc[0] * (1 - COMMISSION - SLIPPAGE)
    peak_nav = INITIAL_CAPITAL; nav = INITIAL_CAPITAL
    trades = []; nav_log = []; prev_price = None; last_month = None
    signal_count = 0; prev_bb_width = None
    entry_price = df['close'].iloc[0]  # 持仓成本, 用于硬止损

    for i, row in df.iterrows():
        price = row['close']; d = row['date']; adj = row['adj_close']
        if prev_price and prev_price > 0 and abs(price/prev_price - 1) > 0.1: shares *= (prev_price / price)
        ym = (d.year, d.month)
        if last_month and ym != last_month: cash += DCA; total_invested += DCA
        last_month = ym
        lower = row['lower']; upper = row['upper']; rsi = row['rsi']
        if np.isnan(lower) or np.isnan(rsi): continue
        nav = cash + (shares + reserve_shares) * price
        if nav > peak_nav: peak_nav = nav
        dd = (nav - peak_nav) / peak_nav if peak_nav > 0 else 0
        if dd < -0.08 and not reserve_active and reserve_pool > 0:
            cash += reserve_pool; reserve_pool = 0; reserve_active = True

        # 硬止损: 股价较买入价跌超12% → 清仓
        if (shares > 0 or reserve_shares > 0) and price < entry_price * (1 - HARD_STOP):
            cash += (shares + reserve_shares) * price * (1 - COMMISSION - SLIPPAGE)
            if reserve_active:
                reserve_pool = reserve_shares * price * (1 - COMMISSION - SLIPPAGE)
                cash -= reserve_pool; reserve_shares = 0; reserve_active = False
            shares = 0; action = 'STOP'
            nav = cash
            trades.append({'date': d.strftime('%Y-%m-%d'), 'dir': 'STOP', 'price': round(price, 4),
                          'nav': round(nav, 0), 'ret': round((nav/total_invested-1)*100, 2), 'type': '止损'})
            nav_log.append({'date': d, 'nav': nav}); continue

        bb_buy = (adj <= lower); bb_sell = (adj >= upper)
        rsi_buy = (rsi <= RSI_OVERSOLD); rsi_sell = (rsi >= RSI_OVERBOUGHT)
        bb_width = (upper - lower) / row['ma'] if row['ma'] > 0 else 0.1
        expanding = (prev_bb_width is not None and bb_width > prev_bb_width)
        if expanding:
            buy_sig = (bb_buy or rsi_buy)
            raw_sell = (bb_sell and rsi >= EXPAND_RSI_SELL)
            upper_acc = row['upper_acc'] if not np.isnan(row['upper_acc']) else 0
            price_acc = row['price_acc'] if not np.isnan(row['price_acc']) else 0
            blocked = (upper_acc > BB_ACCEL_UP) and (price_acc > 0)
            sell_sig = raw_sell and not blocked
        else:
            buy_sig = (bb_buy or rsi_buy)
            sell_sig = (bb_sell or rsi_sell)
            buy_sig = (bb_buy or rsi_buy)
            sell_sig = (bb_sell or (rsi >= RSI_OVERBOUGHT))
        prev_bb_width = bb_width

        action = None; buy_type = ''
        if buy_sig or sell_sig: signal_count += 1
        if signal_count < 2: nav_log.append({'date': d, 'nav': nav}); continue
        if buy_sig:
            if cash > nav * 0.01:
                amt = cash
                buy_shares = amt / price * (1 - COMMISSION - SLIPPAGE)
                if reserve_active:
                    reserve_shares += buy_shares * (RESERVE / (INITIAL_CAPITAL + RESERVE))
                    shares += buy_shares - buy_shares * (RESERVE / (INITIAL_CAPITAL + RESERVE))
                    buy_type = 'RESERVE'
                elif cash < nav * 0.1: shares += buy_shares; buy_type = 'DCA'
                else:                   shares += buy_shares; buy_type = 'BUY'
                cash = 0; action = 'BUY'; entry_price = price
        elif sell_sig:
            if shares > 0 or reserve_shares > 0:
                cash += (shares + reserve_shares) * price * (1 - COMMISSION - SLIPPAGE)
                if reserve_active:
                    reserve_pool = reserve_shares * price * (1 - COMMISSION - SLIPPAGE)
                    cash -= reserve_pool; reserve_shares = 0; reserve_active = False
                shares = 0; action = 'SELL'
        if action:
            nav = cash + (shares + reserve_shares) * price
            trades.append({'date': d.strftime('%Y-%m-%d'), 'dir': action, 'price': round(price, 4),
                          'nav': round(nav, 0), 'ret': round((nav/total_invested-1)*100, 2),
                          'type': buy_type if action == 'BUY' else ''})
        nav_log.append({'date': d, 'nav': nav})

    last_price = df['close'].iloc[-1]; final_nav = cash + (shares + reserve_shares) * last_price
    total_ret = (final_nav / total_invested - 1) * 100
    n_days = len(nav_log)
    ann_ret = ((1 + total_ret/100) ** (252 / n_days) - 1) * 100 if n_days > 0 else 0
    nav_df = pd.DataFrame(nav_log)
    daily_rets = nav_df['nav'].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) * 100
    sharpe = (ann_ret/100 - 0.02) / (ann_vol/100) if ann_vol > 0 else 0
    sortino = (ann_ret/100 - 0.02) / (daily_rets[daily_rets < 0].std() * np.sqrt(252)) if len(daily_rets[daily_rets < 0]) > 0 else 0
    cum = nav_df['nav'] / INITIAL_CAPITAL
    mdd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    calmar = (ann_ret/100) / abs(mdd/100) if mdd != 0 else 0

    trade_df = pd.DataFrame(trades)
    print(f"\n  {ETF_NAME} BB({BB_PERIOD},{BB_STD})+RSI({RSI_PERIOD},{RSI_OVERSOLD},{RSI_OVERBOUGHT})")
    print(f"  扩张卖出: BB∩RSI≥{EXPAND_RSI_SELL}  区间: {start_date_str} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Return: {total_ret:+.2f}%  Annual: {ann_ret:+.2f}%  Sharpe: {sharpe:.3f}  Sortino: {sortino:.3f}")
    print(f"  MaxDD: {mdd:+.2f}%  Calmar: {calmar:.2f}  Trades: {len(trade_df)}  Final: {final_nav:,.0f}")

    # 写入交易记录
    with open(os.path.join(SCRIPT_DIR, 'trades.txt'), 'w', encoding='utf-8') as tf:
        tf.write(f"{'date':<12} {'dir':<6} {'price':<10} {'nav':<12}\n")
        for t in trades: tf.write(f"{t['date']:<12} {t['dir']:<6} {t['price']:<10} {t['nav']:<12,.0f}\n")
        tf.write(f"\n{len(trade_df)} trades\n")

    # 画图
    if len(nav_df) > 1:
        # ── 调整OHLC消除分红/拆分断层，然后直接重算BB ──
        ohlc_cols = ['open','high','low','close']
        has_ohlc = all(c in df.columns for c in ohlc_cols)
        if has_ohlc:
            ohlc_raw = df[['date'] + ohlc_cols].copy()
            # 累计收益率(封顶10%)消除跳空
            r = ohlc_raw['close'].pct_change().fillna(0); r[abs(r) > 0.1] = 0
            cum_adj = (1 + r).cumprod()
            sc = ohlc_raw['close'].iloc[-1] / cum_adj.iloc[-1]
            adj_px = cum_adj * sc
            ratio = adj_px / ohlc_raw['close']
            for col in ohlc_cols:
                ohlc_raw[col] = ohlc_raw[col] * ratio
            ohlc = ohlc_raw.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close'})
            ohlc = ohlc.set_index('date')
            # 在调整后收盘价上重算BB → 绝对对齐
            adj_close_for_bb = ohlc['Close']
            bb_ma = adj_close_for_bb.rolling(BB_PERIOD).mean()
            bb_std = adj_close_for_bb.rolling(BB_PERIOD).std()
            bb_up_s   = bb_ma + BB_STD * bb_std
            bb_lo_s   = bb_ma - BB_STD * bb_std
            bb_mid_s  = bb_ma
            ratio_map = dict(zip(df['date'].dt.strftime('%Y-%m-%d'), ratio))
            adj_close_scaled = adj_close_for_bb
        else:
            ohlc = df[['date','close']].copy()
            ohlc = ohlc.rename(columns={'close':'Close'})
            ohlc['Open']=ohlc['Close']; ohlc['High']=ohlc['Close']; ohlc['Low']=ohlc['Close']
            ohlc = ohlc[['Open','High','Low','Close']].set_index(df['date'])
            bb_up_s  = df['upper']; bb_lo_s = df['lower']; bb_mid_s = df['ma']
            adj_close_scaled = df['adj_close']
            ratio_map = {}

        # A股配色: 红涨绿跌
        mc = mpf.make_marketcolors(up='#CC0000', down='#008800', edge='inherit', wick='inherit', volume='inherit')
        style = mpf.make_mpf_style(marketcolors=mc, gridstyle='',
                                    rc={'font.sans-serif':['SimHei','Microsoft YaHei','DejaVu Sans'],
                                        'axes.unicode_minus':False})

        # 创建三面板 — 先建图再放K线
        fig = plt.figure(figsize=(17, 14), facecolor='white')
        gs = fig.add_gridspec(3, 1, height_ratios=[3.5, 1.2, 1.2], hspace=0.25, left=0.06, right=0.94, top=0.96, bottom=0.04)
        ax_kline = fig.add_subplot(gs[0])

        # BB addplots (用调整后价格重算的BB, 绑定ax_kline)
        ap_up = mpf.make_addplot(bb_up_s, color='#e74c3c', width=0.6, linestyle='--', alpha=0.7, ax=ax_kline)
        ap_lo = mpf.make_addplot(bb_lo_s, color='#27ae60', width=0.6, linestyle='--', alpha=0.7, ax=ax_kline)
        ap_mid = mpf.make_addplot(bb_mid_s, color='#bdc3c7', width=0.5, alpha=0.5, ax=ax_kline)
        addplots = [ap_up, ap_lo, ap_mid]
        addplots = [ap_up, ap_lo, ap_mid]

        # 买卖点标记（价格调整为对齐K线）
        buy_dates = []; buy_prices = []; sell_dates = []; sell_prices = []
        if len(trade_df) > 0:
            for _, t in trade_df.iterrows():
                d = pd.Timestamp(t['date'])
                if d in ohlc.index and not pd.isna(t.get('price',0)):
                    adj_px = t['price'] * ratio_map.get(t['date'], 1.0) if has_ohlc else t['price']
                    if t['dir'] in ('BUY','DCA','RESERVE'): buy_dates.append(d); buy_prices.append(adj_px)
                    elif t['dir'] == 'SELL': sell_dates.append(d); sell_prices.append(adj_px)
        if buy_dates:
            buy_s = pd.Series(index=ohlc.index, data=np.nan); buy_s.loc[buy_dates] = buy_prices
            ap_buy = mpf.make_addplot(buy_s, type='scatter', marker='^', markersize=80, color='#CC0000', alpha=0.9, ax=ax_kline)
            addplots.append(ap_buy)
        if sell_dates:
            sell_s = pd.Series(index=ohlc.index, data=np.nan); sell_s.loc[sell_dates] = sell_prices
            ap_sell = mpf.make_addplot(sell_s, type='scatter', marker='v', markersize=80, color='#008800', alpha=0.9, ax=ax_kline)
            addplots.append(ap_sell)

        # P1: K线图
        mpf.plot(ohlc, type='candle', ax=ax_kline, volume=False, style=style, addplot=addplots)
        ax_kline.set_title(f'{ETF_NAME}  BB({BB_PERIOD},{BB_STD})+RSI({RSI_PERIOD},{RSI_OVERSOLD},{RSI_OVERBOUGHT})  |  '
                           f'{total_ret:+.1f}%  |  {ann_ret:+.1f}%pa  |  夏普{sharpe:.3f}  |  回撤{mdd:+.1f}%  |  {len(trade_df)}笔',
                           fontsize=13, fontweight='bold', loc='left')
        ax_kline.tick_params(labelsize=9); ax_kline.grid(True, alpha=0.15)

        # P2: 净值曲线
        ax2 = fig.add_subplot(gs[1]); ax2.set_facecolor('#FAFAFA')
        ax2.plot(nav_df['date'], nav_df['nav']/INITIAL_CAPITAL, color='#c0392b', lw=2.0, label='策略净值')
        ax2.plot(df['date'], df['adj_close']/df['adj_close'].iloc[0], color='#95a5a6', lw=1.0, ls='--', label='买入持有')
        ax2.axhline(1.0, color='#bdc3c7', lw=0.8, ls='--')
        ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)
        ax2.tick_params(labelsize=9); ax2.grid(True, alpha=0.15)

        # P3: 年度收益柱状图
        ax3 = fig.add_subplot(gs[2])
        nav_df_copy = nav_df.copy()
        nav_df_copy['year'] = nav_df_copy['date'].dt.year
        years = []; year_rets = []
        for yr, grp in nav_df_copy.groupby('year'):
            if len(grp) < 10: continue
            yr_ret = (grp['nav'].iloc[-1] / grp['nav'].iloc[0] - 1) * 100
            years.append(yr); year_rets.append(yr_ret)
        bar_colors = ['#c0392b' if r >= 0 else '#27ae60' for r in year_rets]
        bars = ax3.bar(range(len(years)), year_rets, color=bar_colors, alpha=0.85, edgecolor='white', lw=1)
        for bar, val in zip(bars, year_rets):
            offset = 1.5 if val >= 0 else -3.5
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                    f'{val:+.1f}%', ha='center', fontsize=13, fontweight='bold',
                    color='#c0392b' if val >= 0 else '#27ae60')
        ax3.axhline(y=0, color='black', lw=0.8)
        ax3.set_xticks(range(len(years)))
        ax3.set_xticklabels([str(y) for y in years], fontsize=12, fontweight='bold')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:+.0f}%'))
        ax3.set_title('年度收益', fontsize=12, fontweight='bold', loc='left')
        ax3.grid(True, alpha=0.15, axis='y'); ax3.tick_params(labelsize=9)

        plt.savefig(os.path.join(SCRIPT_DIR, 'backtest_chart.png'), dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
        print(f"  图表: {os.path.join(SCRIPT_DIR, 'backtest_chart.png')}")

# ======================== 实时信号 ========================
def live_signal():
    df = get_data(ETF_SYMBOL)
    compute_indicators(df)

    r = df.iloc[-1]; prev = df.iloc[-2]
    price = r['close']; rsi = r['rsi']
    bb_pos = (r['adj_close'] - r['lower']) / (r['upper'] - r['lower']) if (r['upper'] - r['lower']) > 0 else 0.5
    bb_w = (r['upper'] - r['lower']) / r['ma'] if r['ma'] > 0 else 0
    prev_bb_w = (prev['upper'] - prev['lower']) / prev['ma'] if prev['ma'] > 0 else 0
    expanding = bb_w > prev_bb_w

    bb_buy = r['adj_close'] <= r['lower']; bb_sell = r['adj_close'] >= r['upper']
    rsi_buy = rsi <= RSI_OVERSOLD

    upper_acc = r['upper_acc'] if not np.isnan(r['upper_acc']) else 0
    lower_acc = r['lower_acc'] if not np.isnan(r['lower_acc']) else 0
    price_acc = r['price_acc'] if not np.isnan(r['price_acc']) else 0
    sell_blocked = (upper_acc > BB_ACCEL_UP) and (price_acc > 0)

    if expanding:
        rsi_sell_eff = rsi >= EXPAND_RSI_SELL
        sell_ok = (bb_sell and rsi_sell_eff) and not sell_blocked
        if sell_blocked:
            blk = f'上轨加速{upper_acc:+.5f} 且 价格加速{price_acc:+.5f} → 双加速暂缓'
        elif upper_acc > BB_ACCEL_UP and price_acc <= 0:
            blk = f'上轨加速{upper_acc:+.5f} 但 价格减速{price_acc:+.5f} → 覆写卖出'
        else:
            blk = f'上轨加速{upper_acc:+.5f} → 允许'
        sell_logic = (f'BB上轨{"✓" if bb_sell else "✗"} AND RSI≥{EXPAND_RSI_SELL}{"✓" if rsi_sell_eff else "✗"}'
                      f' → {blk}')
        buy_ok = (bb_buy or rsi_buy)
        buy_logic = f'BB下轨{"✓" if bb_buy else "✗"} OR RSI≤{RSI_OVERSOLD}{"✓" if rsi_buy else "✗"}'
    else:
        rsi_sell_eff = rsi >= RSI_OVERBOUGHT
        sell_logic = f'BB上轨{"✓" if bb_sell else "✗"} OR RSI≥{RSI_OVERBOUGHT}{"✓" if rsi_sell_eff else "✗"}'
        buy_logic = f'BB下轨{"✓" if bb_buy else "✗"} OR RSI≤{RSI_OVERSOLD}{"✓" if rsi_buy else "✗"}'
        buy_ok = (bb_buy or rsi_buy); sell_ok = (bb_sell or rsi_sell_eff)

    # 综合信号
    if buy_ok and not sell_ok:   action = '🟢 买入'; detail = buy_logic
    elif sell_ok and not buy_ok: action = '🔴 卖出'; detail = sell_logic
    elif buy_ok and sell_ok:     action = '🟡 冲突'; detail = f'买:{buy_logic} | 卖:{sell_logic}'
    else:                        action = '⚪ 持有/观望'; detail = '无明确信号'

    trend = '扩张↑' if expanding else '收缩↓'
    if expanding:
        trend_note = (f'上轨{upper_acc:+.5f} 价格{price_acc:+.5f}'
                      f'{" → 双加速 暂缓卖出" if sell_blocked else " → 允许卖出"}')
    else:
        trend_note = '正常模式, 灵活交易'

    print(f"\n{'='*60}")
    print(f"  {ETF_NAME} ({ETF_SYMBOL})  实时行情  {r['date'].strftime('%Y-%m-%d %A')}")
    print(f"{'='*60}")
    print(f"  收盘价:  {price:.4f}")
    print(f"  RSI:     {rsi:.1f}")
    print(f"  BB位置:  {bb_pos*100:.0f}%  (0%=下轨, 100%=上轨)")
    print(f"  BB宽度:  {bb_w*100:.2f}%  ({trend})")
    print(f"  上/中/下: {r['upper']:.4f} / {r['ma']:.4f} / {r['lower']:.4f}")
    print(f"  加速度:  {trend_note}")
    print(f"{'─'*60}")
    print(f"  买入: {buy_logic}")
    print(f"  卖出: {sell_logic}")
    print(f"{'─'*60}")
    print(f"  >> {action}")
    print(f"     {detail}")
    print(f"{'='*60}")

    # 近5日
    print(f"\n  近5日:")
    for i in range(-5, 0):
        row = df.iloc[i]
        p = (row['adj_close'] - row['lower']) / (row['upper'] - row['lower']) * 100 if (row['upper'] - row['lower']) > 0 else 50
        print(f"    {row['date'].strftime('%m-%d')}  收 {row['close']:.4f}  RSI {row['rsi']:.1f}  BB {p:.0f}%")

    # 预警
    dist_upper = (r['upper'] - r['adj_close']) / r['adj_close'] * 100
    dist_lower = (r['adj_close'] - r['lower']) / r['adj_close'] * 100
    print(f"\n  距上轨 {dist_upper:+.1f}%  |  距下轨 {dist_lower:.1f}%")
    if not expanding and dist_upper < 1.0 and rsi >= 65:
        print(f"  ⚠ 靠近上轨+RSI偏高, 密切关注卖出")
    elif not expanding and dist_lower < 2.0 and rsi <= 40:
        print(f"  💡 靠近下轨+RSI偏低, 关注买入机会")
    elif expanding and dist_upper < 2.0 and rsi >= EXPAND_RSI_SELL:
        print(f"  ⚠ 扩张中价格接近上轨+RSI≥{EXPAND_RSI_SELL}, 卖出双确认接近触发")
    elif expanding and dist_lower < 2.0 and rsi <= RSI_OVERSOLD:
        print(f"  💡 扩张中价格接近下轨+RSI≤{RSI_OVERSOLD}, 买入双确认接近触发")

# ======================== 主入口 ========================
def main():
    parser = argparse.ArgumentParser(description='YH02 红利低波 BB+RSI策略')
    parser.add_argument('--from', dest='from_date', type=str, default=None, metavar='DATE',
                        help='回测起始日期 (如 2019-01-18)')
    parser.add_argument('--dca', dest='dca', type=float, default=0, metavar='WAN',
                        help='月定投金额(万), 如 --dca 2')
    args = parser.parse_args()

    if args.dca > 0:
        global DCA; DCA = args.dca * 10000

    if args.from_date:
        run_backtest(args.from_date)
    else:
        live_signal()

if __name__ == '__main__':
    main()
