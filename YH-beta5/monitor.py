# -*- coding: utf-8 -*-
"""
=============================================================================
红利低波 BB+RSI 组合信号 (OR逻辑)
=============================================================================
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import akshare as ak, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json, os, warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

INITIAL_CAPITAL = 1_000_000; RESERVE = 100_000; COMMISSION = 0.0001; SLIPPAGE = 0.0001
ETF_SYMBOL = 'sh512890'; ETF_NAME = '红利低波'
BB_PERIOD = 52; BB_STD = 2.0
RSI_PERIOD = 20; RSI_OVERSOLD = 38; RSI_OVERBOUGHT = 68
BUY_MODE = 'OR'    # 买入: OR=任一触发
SELL_MODE = 'OR'   # 卖出: OR=任一触发
POSITION_FILE = 'd:\\策略\\YH-beta5\\positions.json'

def compute_indicators(df):
    ret = df['close'].pct_change().fillna(0); ret[abs(ret) > 0.1] = 0
    df['adj_close'] = (1 + ret).cumprod()
    df['ma'] = df['adj_close'].rolling(BB_PERIOD).mean()
    df['std'] = df['adj_close'].rolling(BB_PERIOD).std()
    df['upper'] = df['ma'] + BB_STD * df['std']
    df['lower'] = df['ma'] - BB_STD * df['std']
    delta = df['adj_close'].diff()
    gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
    df['rsi'] = 100 - 100 / (1 + gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean() /
                             loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean().replace(0, np.nan))

def run_backtest(start_date_str):
    start_date = pd.Timestamp(start_date_str)
    df = ak.fund_etf_hist_sina(symbol=ETF_SYMBOL)
    df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date')
    compute_indicators(df)
    df = df[df['date'] >= start_date].reset_index(drop=True)

    cash = 0.0; total_invested = INITIAL_CAPITAL; reserve_pool = RESERVE; reserve_active = False; reserve_shares = 0.0
    shares = INITIAL_CAPITAL / df['close'].iloc[0] * (1 - COMMISSION - SLIPPAGE)
    peak_nav = INITIAL_CAPITAL; nav = INITIAL_CAPITAL
    trades = []; nav_log = []; prev_price = None; last_month = None; dca = 20000

    for i, row in df.iterrows():
        price = row['close']; d = row['date']; adj = row['adj_close']
        if prev_price and prev_price > 0 and abs(price/prev_price - 1) > 0.1: shares *= (prev_price / price)
        prev_price = price
        ym = (d.year, d.month)
        if last_month and ym != last_month: cash += dca; total_invested += dca
        last_month = ym
        lower = row['lower']; upper = row['upper']; rsi = row['rsi']
        if np.isnan(lower) or np.isnan(rsi): continue
        nav = cash + (shares + reserve_shares) * price
        if nav > peak_nav: peak_nav = nav
        dd = (nav - peak_nav) / peak_nav if peak_nav > 0 else 0
        # 回撤>10%启动备用金
        if dd < -0.1 and not reserve_active and reserve_pool > 0:
            cash += reserve_pool; reserve_pool = 0; reserve_active = True

        bb_buy = (adj <= lower); bb_sell = (adj >= upper)
        rsi_buy = (rsi <= RSI_OVERSOLD); rsi_sell = (rsi >= RSI_OVERBOUGHT)
        buy_sig = (bb_buy and rsi_buy) if BUY_MODE == 'AND' else (bb_buy or rsi_buy)
        sell_sig = (bb_sell and rsi_sell) if SELL_MODE == 'AND' else (bb_sell or rsi_sell)

        action = None
        if buy_sig and cash > nav * 0.1:
            buy_shares = cash / price * (1 - COMMISSION - SLIPPAGE)
            if reserve_active:
                reserve_shares += buy_shares * (RESERVE / (INITIAL_CAPITAL + RESERVE))
                shares += buy_shares - buy_shares * (RESERVE / (INITIAL_CAPITAL + RESERVE))
            else:
                shares += buy_shares
            cash = 0; action = 'BUY'
        elif sell_sig and (shares > 0 or reserve_shares > 0):
            cash += (shares + reserve_shares) * price * (1 - COMMISSION - SLIPPAGE)
            if reserve_active:
                reserve_pool = reserve_shares * price * (1 - COMMISSION - SLIPPAGE)
                cash -= reserve_pool; reserve_shares = 0; reserve_active = False
            shares = 0; action = 'SELL'
        if action:
            nav = cash + (shares + reserve_shares) * price
            trades.append({'日期': d.strftime('%Y-%m-%d'), '方向': action, '价格': round(price, 4),
                          '净值': round(nav, 0), '累计收益%': round((nav/total_invested-1)*100, 2)})
        nav_log.append({'date': d, 'nav': nav})

    last_price = df['close'].iloc[-1]; final_nav = cash + shares * last_price
    total_ret = (final_nav / total_invested - 1) * 100
    n_days = len(nav_log)
    ann_ret = ((1 + total_ret/100) ** (252 / n_days) - 1) * 100 if n_days > 0 else 0
    nav_df = pd.DataFrame(nav_log)
    daily_rets = nav_df['nav'].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252) * 100
    sharpe = (ann_ret/100 - 0.02) / (ann_vol/100) if ann_vol > 0 else 0
    cum = nav_df['nav'] / INITIAL_CAPITAL
    mdd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    print(f"\n  {ETF_NAME} BB+RSI({BUY_MODE+'/'+SELL_MODE}) BB({BB_PERIOD},{BB_STD}) RSI({RSI_PERIOD},{RSI_OVERSOLD},{RSI_OVERBOUGHT})")
    print(f"  Return: {total_ret:+.2f}%  Annual: {ann_ret:+.2f}%  Sharpe: {sharpe:.3f}")
    print(f"  MaxDD: {mdd:+.2f}%  Trades: {len(trades)}  Final: {final_nav:,.0f}")
    # 交易明细写到txt
    with open('d:\\策略\\YH-beta5\\trades.txt', 'w', encoding='utf-8') as tf:
        tf.write(f"{'日期':<12} {'方向':<6} {'价格':<10} {'净值':<12}\n")
        for t in trades:
            tf.write(f"{t['日期']:<12} {t['方向']:<6} {t['价格']:<10} {t['净值']:<12,.0f}\n")
        tf.write(f"\n共{len(trades)}笔\n")
    print(f"  明细: d:\\策略\\YH-beta5\\trades.txt")

    # Chart
    trade_df = pd.DataFrame(trades)
    if len(nav_df) > 1:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.suptitle(f'{ETF_NAME} BB+RSI  Ret {total_ret:+.1f}%  Trades {len(trade_df)}', fontsize=14, fontweight='bold')
        ax = axes[0]
        ax.plot(df['date'], df['adj_close'], color='#2c3e50', lw=1, label='Price')
        ax.fill_between(df['date'], df['upper'], df['lower'], alpha=0.08, color='#3498db')
        ax.plot(df['date'], df['upper'], color='#e74c3c', lw=0.8, ls='--', label='BB Upper')
        ax.plot(df['date'], df['ma'], color='#3498db', lw=0.8, label='BB MA')
        ax.plot(df['date'], df['lower'], color='#2ecc71', lw=0.8, ls='--', label='BB Lower')
        pm = dict(zip(df['date'].dt.strftime('%Y-%m-%d'), df['adj_close']))
        if len(trade_df) > 0:
            bd = trade_df[trade_df['方向']=='BUY'].copy(); sd = trade_df[trade_df['方向']=='SELL'].copy()
            bd['adj'] = bd['日期'].map(pm); sd['adj'] = sd['日期'].map(pm)
            if len(bd) > 0: ax.scatter(pd.to_datetime(bd['日期']), bd['adj'], color='#c0392b', s=40, marker='^', zorder=5, label='Buy')
            if len(sd) > 0: ax.scatter(pd.to_datetime(sd['日期']), sd['adj'], color='#27ae60', s=40, marker='v', zorder=5, label='Sell')
        ax.set_ylabel('Price'); ax.legend(fontsize=8, loc='upper left', ncol=2); ax.grid(True, alpha=0.3)
        ax = axes[1]
        ax.plot(nav_df['date'], nav_df['nav']/INITIAL_CAPITAL, color='#e74c3c', lw=2.5, label='BB+RSI')
        ax.plot(df['date'], df['adj_close']/df['adj_close'].iloc[0], color='#3498db', lw=1.5, ls='--', label='Buy&Hold')
        ax.axhline(1.0, color='gray', lw=0.5, ls='--'); ax.set_ylabel('NAV'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax = axes[2]
        cum_ret = (nav_df['nav'] / INITIAL_CAPITAL - 1) * 100
        ax.fill_between(nav_df['date'], 0, cum_ret, color='#e74c3c', alpha=0.15); ax.plot(nav_df['date'], cum_ret, color='#c0392b', lw=1.2)
        ax.axhline(0, color='gray', lw=0.5); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig('d:\\策略\\YH-beta5\\backtest_chart.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
        print(f"  Chart: d:\\策略\\YH-beta5\\backtest_chart.png")

def main():
    import argparse, urllib.request, json
    parser = argparse.ArgumentParser(); parser.add_argument('--from', dest='from_date', type=str, default=None)
    args = parser.parse_args()
    if args.from_date: run_backtest(args.from_date); return
    df = ak.fund_etf_hist_sina(symbol=ETF_SYMBOL); df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date')
    compute_indicators(df)
    r = df.iloc[-1]; price = r['close']; rsi = r['rsi']; bb_pos = (r['adj_close']-r['lower'])/(r['upper']-r['lower'])
    sig = '买入!' if (bb_pos<=0 or rsi<=RSI_OVERSOLD) else ('卖出!' if (bb_pos>=1 or rsi>=RSI_OVERBOUGHT) else '持有')
    print(f"\n  {ETF_NAME} BB+RSI  {r['date'].strftime('%Y-%m-%d')}  Price {price:.4f}  RSI {rsi:.1f}  BB {bb_pos*100:.0f}%")
    print(f"  建议: {sig}  备用金: {RESERVE/10000:.0f}万")
    try:
        bark_url = 'https://api.day.app/eoq8G58fJtDDFxHjhNueGH'
        lines = [f"{ETF_NAME} BB+RSI  Price {price:.4f}", f"RSI {rsi:.1f}  BB {bb_pos*100:.0f}%  建议: {sig}"]
        payload = json.dumps({'title': f'{ETF_NAME} {sig}', 'body': '\n'.join(lines)}).encode()
        urllib.request.urlopen(urllib.request.Request(bark_url, data=payload,
            headers={'Content-Type': 'application/json'}), timeout=5)
        print("  Pushed to phone")
    except: pass

if __name__ == '__main__': main()
