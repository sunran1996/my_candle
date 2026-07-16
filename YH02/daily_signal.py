# -*- coding: utf-8 -*-
"""GitHub Actions 每日自动信号 — 推送到手机Bark"""
import sys, io, os, json, ssl, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import akshare as ak, pandas as pd, numpy as np
import urllib.request as ur
warnings.filterwarnings('ignore')

ETF_SYMBOL = 'sh512890'; ETF_NAME = '红利低波'
BB_PERIOD = 45; BB_STD = 2.0
RSI_PERIOD = 14; RSI_OVERSOLD = 30; RSI_OVERBOUGHT = 70
EXPAND_RSI_SELL = 65; BB_ACCEL_UP = 0.001

def compute_indicators(df):
    ret = df['close'].pct_change().fillna(0); ret[abs(ret) > 0.1] = 0
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

def send_bark(title, body):
    """推送到Bark"""
    bark_key = os.environ.get('BARK_KEY', 'eoq8G58fJtDDFxHjhNueGH')
    try:
        data = json.dumps({'title': title, 'body': body}).encode()
        ur.urlopen(ur.Request(f'https://api.day.app/{bark_key}', data=data,
                   headers={'Content-Type': 'application/json'}), timeout=10)
        print("  已推送到手机")
    except Exception as e:
        print(f"  推送失败: {e}")

def main():
    # 1. 获取数据
    df = ak.fund_etf_hist_sina(symbol=ETF_SYMBOL)
    df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date').reset_index(drop=True)
    df = compute_indicators(df)

    r = df.iloc[-1]; prev = df.iloc[-2]
    price = r['close']; rsi = r['rsi']
    bb_pos = (r['adj_close'] - r['lower']) / (r['upper'] - r['lower']) * 100
    bb_w = (r['upper'] - r['lower']) / r['ma'] * 100
    expanding = bb_w > (prev['upper'] - prev['lower']) / prev['ma'] * 100

    bb_buy = r['adj_close'] <= r['lower']; bb_sell = r['adj_close'] >= r['upper']
    rsi_buy = rsi <= RSI_OVERSOLD
    upper_acc = r['upper_acc'] if not np.isnan(r['upper_acc']) else 0
    price_acc = r['price_acc'] if not np.isnan(r['price_acc']) else 0

    if expanding:
        buy_ok = (bb_buy or rsi_buy)
        rsi_sell_ok = rsi >= EXPAND_RSI_SELL
        raw_sell = bb_sell and rsi_sell_ok
        blocked = (upper_acc > BB_ACCEL_UP) and (price_acc > 0)
        sell_ok = raw_sell and not blocked
    else:
        buy_ok = (bb_buy or rsi_buy)
        sell_ok = (bb_sell or rsi >= RSI_OVERBOUGHT)

    # 2. 信号判断
    if buy_ok and not sell_ok:
        signal = '🟢 买入'
        detail = (f'BB下轨{"✓" if bb_buy else "✗"} RSI≤30{"✓" if rsi_buy else "✗"}'
                  if expanding else f'BB下轨{"✓" if bb_buy else "✗"} OR RSI≤30{"✓" if rsi_buy else "✗"}')
    elif sell_ok and not buy_ok:
        signal = '🔴 卖出'
        detail = f'BB上轨{"✓" if bb_sell else "✗"} RSI≥{EXPAND_RSI_SELL if expanding else 70}{"✓" if (rsi>=EXPAND_RSI_SELL if expanding else rsi>=70) else "✗"}'
    elif buy_ok and sell_ok:
        signal = '🟡 冲突'
        detail = '买卖同时触发'
    else:
        signal = '⚪ 持有/观望'
        detail = '无信号'

    # 3. 打印
    trend = '扩张↑' if expanding else '收缩↓'
    print(f"\n{ETF_NAME} {r['date'].strftime('%Y-%m-%d')}  价格{price:.4f}  RSI{rsi:.1f}  BB{bb_pos:.0f}%  {trend}")
    print(f"信号: {signal}  {detail}")

    # 4. 推送手机
    body = (f'价格 {price:.4f}  RSI {rsi:.1f}  BB {bb_pos:.0f}%  {trend}\n'
            f'上轨加速度 {upper_acc:+.5f}  价格加速度 {price_acc:+.5f}\n'
            f'{signal}: {detail}')
    send_bark(f'{ETF_NAME} {signal}', body)

if __name__ == '__main__':
    main()
