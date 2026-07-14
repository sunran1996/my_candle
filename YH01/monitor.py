# -*- coding: utf-8 -*-
"""
YH01 红利低波 BB+RSI 策略
BB(52,2.0)+RSI(20,38,68) OR/OR + DCA 2w + 10w备用金(回撤>8%)
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
BUY_MODE = 'OR'; SELL_MODE = 'OR'
POSITION_FILE = 'd:\\策略\\YH01\\positions.json'

def compute_indicators(df):
    ret = df['close'].pct_change().fillna(0); ret[abs(ret) > 0.1] = 0
    df['adj_close'] = (1 + ret).cumprod()
    df['ma'] = df['adj_close'].rolling(BB_PERIOD).mean()
    df['std'] = df['adj_close'].rolling(BB_PERIOD).std()
    df['upper'] = df['ma'] + BB_STD * df['std']
    df['lower'] = df['ma'] - BB_STD * df['std']
    delta = df['adj_close'].diff(); gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
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
    trades = []; missed = []; nav_log = []; prev_price = None; last_month = None; dca = 20000
    signal_count = 0  # 信号计数, 跳过第一个

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
        if dd < -0.08 and not reserve_active and reserve_pool > 0:
            cash += reserve_pool; reserve_pool = 0; reserve_active = True

        bb_buy = (adj <= lower); bb_sell = (adj >= upper)
        rsi_buy = (rsi <= RSI_OVERSOLD); rsi_sell = (rsi >= RSI_OVERBOUGHT)
        buy_sig = (bb_buy or rsi_buy)
        sell_sig = (bb_sell or rsi_sell)

        action = None; buy_type = ''
        if buy_sig or sell_sig: signal_count += 1
        if signal_count < 2:  # 跳过第一个信号
            nav_log.append({'date': d, 'nav': nav}); continue
        if buy_sig:
            if cash > nav * 0.01:
                buy_shares = cash / price * (1 - COMMISSION - SLIPPAGE)
                if reserve_active:
                    reserve_shares += buy_shares * (RESERVE / (INITIAL_CAPITAL + RESERVE))
                    shares += buy_shares - buy_shares * (RESERVE / (INITIAL_CAPITAL + RESERVE))
                    buy_type = 'RESERVE'
                elif cash < nav * 0.1:
                    shares += buy_shares; buy_type = 'DCA'
                else:
                    shares += buy_shares; buy_type = 'BUY'
                cash = 0; action = 'BUY'
            else:
                missed.append({'date': d, 'type': 'buy'})
        elif sell_sig:
            if shares > 0 or reserve_shares > 0:
                cash += (shares + reserve_shares) * price * (1 - COMMISSION - SLIPPAGE)
                if reserve_active:
                    reserve_pool = reserve_shares * price * (1 - COMMISSION - SLIPPAGE)
                    cash -= reserve_pool; reserve_shares = 0; reserve_active = False
                shares = 0; action = 'SELL'
            else:
                missed.append({'date': d, 'type': 'sell'})
        if action:
            nav = cash + (shares + reserve_shares) * price
            trades.append({'date': d.strftime('%Y-%m-%d'), 'dir': action, 'price': round(price, 4),
                          'nav': round(nav, 0), 'ret': round((nav/total_invested-1)*100, 2),
                          'type': buy_type if action == 'BUY' else ''})
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

    trade_df = pd.DataFrame(trades)
    print(f"\n  {ETF_NAME} BB({BB_PERIOD},{BB_STD})+RSI({RSI_PERIOD},{RSI_OVERSOLD},{RSI_OVERBOUGHT})")
    print(f"  Return: {total_ret:+.2f}%  Annual: {ann_ret:+.2f}%  Sharpe: {sharpe:.3f}")
    print(f"  MaxDD: {mdd:+.2f}%  Trades: {len(trade_df)}  Final: {final_nav:,.0f}")

    with open('d:\\策略\\YH01\\trades.txt', 'w', encoding='utf-8') as tf:
        tf.write(f"{'date':<12} {'dir':<6} {'price':<10} {'nav':<12}\n")
        for t in trades:
            tf.write(f"{t['date']:<12} {t['dir']:<6} {t['price']:<10} {t['nav']:<12,.0f}\n")
        tf.write(f"\n{len(trade_df)} trades\n")

    if len(nav_df) > 1:
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [4, 1]})
        fig.suptitle(f'{ETF_NAME} BB+RSI  Ret {total_ret:+.1f}%  Trades {len(trade_df)}', fontsize=14, fontweight='bold')
        ax = axes[0]
        ax.plot(df['date'], df['adj_close'], color='#2c3e50', lw=1)
        ax.fill_between(df['date'], df['upper'], df['lower'], alpha=0.08, color='#3498db')
        ax.plot(df['date'], df['upper'], color='#e74c3c', lw=0.8, ls='--')
        ax.plot(df['date'], df['ma'], color='#3498db', lw=0.8)
        ax.plot(df['date'], df['lower'], color='#2ecc71', lw=0.8, ls='--')
        pm = dict(zip(df['date'].dt.strftime('%Y-%m-%d'), df['adj_close']))
        if len(trade_df) > 0:
            sd = trade_df[trade_df['dir']=='SELL'].copy(); sd['adj'] = sd['date'].map(pm)
            if len(sd) > 0: ax.scatter(pd.to_datetime(sd['date']), sd['adj'], color='#27ae60', s=40, marker='v', zorder=5, label='Sell')
            for c, t, lbl in [('#c0392b','BUY','Buy'),('#e67e22','DCA','Buy(DCA)'),('#8e44ad','RESERVE','Buy(Reserve)')]:
                bd = trade_df[(trade_df['dir']=='BUY')&(trade_df['type']==t)]
                if len(bd) > 0:
                    bd = bd.copy(); bd['adj'] = bd['date'].map(pm)
                    ax.scatter(pd.to_datetime(bd['date']), bd['adj'], color=c, s=40 if t=='BUY' else 30, marker='^', zorder=5, label=lbl)
        if missed:
            md = pd.DataFrame(missed); md['date'] = pd.to_datetime(md['date'])
            mb = md[md['type']=='buy']; ms = md[md['type']=='sell']
            mb['adj'] = mb['date'].dt.strftime('%Y-%m-%d').map(pm); ms['adj'] = ms['date'].dt.strftime('%Y-%m-%d').map(pm)
            if len(mb) > 0: ax.scatter(mb['date'], mb['adj'], color='#e74c3c', s=20, marker='^', alpha=0.35, zorder=4, label='Buy(missed)')
            if len(ms) > 0: ax.scatter(ms['date'], ms['adj'], color='#2ecc71', s=20, marker='v', alpha=0.35, zorder=4, label='Sell(missed)')
        ax.set_ylabel('Price'); ax.legend(fontsize=8, loc='upper left', ncol=2); ax.grid(True, alpha=0.3)
        ax = axes[1]
        ax.plot(nav_df['date'], nav_df['nav']/INITIAL_CAPITAL, color='#e74c3c', lw=2.5, label='Strategy')
        ax.plot(df['date'], df['adj_close']/df['adj_close'].iloc[0], color='#3498db', lw=1.5, ls='--', label='Buy&Hold')
        ax.axhline(1.0, color='gray', lw=0.5, ls='--'); ax.set_ylabel('NAV'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig('d:\\策略\\YH01\\backtest_chart.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close()

def main():
    import argparse, urllib.request, json
    parser = argparse.ArgumentParser(); parser.add_argument('--from', dest='from_date', type=str, default=None)
    args = parser.parse_args()
    if args.from_date: run_backtest(args.from_date); return
    df = ak.fund_etf_hist_sina(symbol=ETF_SYMBOL); df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date')
    compute_indicators(df)
    r = df.iloc[-1]; price = r['close']; rsi = r['rsi']; bb_pos = (r['adj_close']-r['lower'])/(r['upper']-r['lower'])
    sig = 'Buy' if (bb_pos<=0 or rsi<=RSI_OVERSOLD) else ('Sell' if (bb_pos>=1 or rsi>=RSI_OVERBOUGHT) else 'Hold')
    print(f"\n  {ETF_NAME}  {r['date'].strftime('%Y-%m-%d')}  Price {price:.4f}  RSI {rsi:.1f}  BB {bb_pos*100:.0f}%")
    print(f"  Signal: {sig}")
    try:
        bark_url = 'https://api.day.app/eoq8G58fJtDDFxHjhNueGH'
        payload = json.dumps({'title': f'{ETF_NAME} {sig}', 'body': f'Price {price:.4f} RSI {rsi:.1f} BB {bb_pos*100:.0f}%'}).encode()
        urllib.request.urlopen(urllib.request.Request(bark_url, data=payload,
            headers={'Content-Type': 'application/json'}), timeout=5)
    except: pass

if __name__ == '__main__': main()
