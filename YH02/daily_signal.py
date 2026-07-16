# -*- coding: utf-8 -*-
"""GitHub Actions 每日自动信号 — 生成收益图 + 推送到手机Bark"""
import sys, io, os, json, ssl, time, base64, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import akshare as ak, pandas as pd, numpy as np
import urllib.request as ur
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ETF_SYMBOL = 'sh512890'; ETF_NAME = '红利低波'
BB_PERIOD = 45; BB_STD = 2.0
RSI_PERIOD = 14; RSI_OVERSOLD = 30; RSI_OVERBOUGHT = 70
EXPAND_RSI_SELL = 65; BB_ACCEL_UP = 0.001
BARK_KEY = 'eoq8G58fJtDDFxHjhNueGH'
REPO = 'sunran1996/my_candle'

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

def gen_chart(df, nav_start=1_000_000, lookback=180):
    """生成近期收益图"""
    recent = df.tail(lookback).copy()
    # 模拟净值: 用 adj_close 归一化
    recent['nav'] = nav_start * recent['adj_close'] / recent['adj_close'].iloc[0]

    # 上证指数对比
    try:
        sh = ak.stock_zh_index_daily(symbol='sh000001')
        sh['date'] = pd.to_datetime(sh['date']); sh = sh.sort_values('date')
        sh = sh[sh['date'] >= recent['date'].iloc[0]]
        if len(sh) > 1:
            sh['sh_nav'] = nav_start * sh['close'] / sh['close'].iloc[0]
            sh_dates = sh['date'].tolist()
            sh_vals = (sh['sh_nav'] / nav_start).tolist()
        else:
            sh_vals = []
    except:
        sh_vals = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor='white',
                              width_ratios=[2.5, 1], gridspec_kw={'wspace': 0.05})
    fig.subplots_adjust(left=0.06, right=0.94, top=0.85, bottom=0.15)

    # 左: 净值曲线
    ax = axes[0]
    nvs = (recent['nav'] / nav_start).tolist()
    dates = recent['date'].dt.strftime('%m-%d').tolist()
    ax.fill_between(range(len(nvs)), 1.0, nvs, alpha=0.08, color='#27ae60')
    ax.plot(range(len(nvs)), nvs, color='#27ae60', lw=1.8, label=ETF_NAME)
    if sh_vals:
        ax.plot(range(len(sh_vals)), sh_vals, color='#bdc3c7', lw=1, ls='--', alpha=0.6, label='上证指数')
    ax.scatter(len(nvs)-1, nvs[-1], color='#27ae60', s=50, zorder=5)
    ax.scatter(len(nvs)-1, nvs[-1], color='#27ae60', s=150, zorder=4, alpha=0.12)
    ax.axhline(y=1.0, color='#2c3e50', lw=1.2)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.8)
    ax.set_xlim(-0.5, len(nvs)-0.5)
    if len(nvs) >= 2:
        ax.set_xticks([0, len(nvs)-1])
        ax.set_xticklabels([dates[0], dates[-1]], fontsize=9)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values(): spine.set_visible(True); spine.set_color('#2c3e50'); spine.set_linewidth(1.2)
    ax.grid(True, alpha=0.1)

    # 右: 信息卡
    r = df.iloc[-1]; prev = df.iloc[-2]
    price = r['close']; rsi = r['rsi']
    bb_pos = (r['adj_close'] - r['lower']) / (r['upper'] - r['lower']) * 100
    bb_w = (r['upper'] - r['lower']) / r['ma'] * 100
    expanding = bb_w > (prev['upper'] - prev['lower']) / prev['ma'] * 100
    upper_acc = r['upper_acc'] if not np.isnan(r['upper_acc']) else 0
    price_acc = r['price_acc'] if not np.isnan(r['price_acc']) else 0

    bb_buy = r['adj_close'] <= r['lower']; bb_sell = r['adj_close'] >= r['upper']
    rsi_buy = rsi <= RSI_OVERSOLD

    if expanding:
        buy_ok = (bb_buy or rsi_buy)
        sell_ok = (bb_sell and rsi >= EXPAND_RSI_SELL) and not ((upper_acc > BB_ACCEL_UP) and (price_acc > 0))
    else:
        buy_ok = (bb_buy or rsi_buy)
        sell_ok = (bb_sell or rsi >= RSI_OVERBOUGHT)

    if buy_ok and not sell_ok: sig = '🟢 买入'
    elif sell_ok and not buy_ok: sig = '🔴 卖出'
    else: sig = '⚪ 持有'

    trend = '扩张↑' if expanding else '收缩↓'
    ret_pct = (nvs[-1] - 1) * 100

    ax = axes[1]; ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 12)
    ax.text(0, 11, ETF_NAME, fontsize=16, fontweight='bold', color='black')
    ax.text(0, 10, f'YH02 每日信号', fontsize=10, color='#95a5a6')
    ax.text(0, 8.5, f'{price:.4f}', fontsize=22, color='#c0392b', fontweight='bold')
    ax.text(0, 7.2, '最新价', fontsize=9, color='#95a5a6')
    ax.text(0, 5.8, sig, fontsize=22)
    ax.text(0, 3.8, f'RSI {rsi:.1f}  |  BB {bb_pos:.0f}%  |  {trend}', fontsize=11, color='black')
    ax.text(0, 2.5, f'近{lookback}日收益 {ret_pct:+.1f}%', fontsize=13, color='#27ae60' if ret_pct>=0 else '#e74c3c', fontweight='bold')
    ax.text(0, 1.2, r['date'].strftime('%Y-%m-%d'), fontsize=10, color='#95a5a6')

    buf = io.BytesIO()
    plt.savefig(buf, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    return buf.getvalue(), sig, price, rsi, bb_pos, trend, upper_acc, price_acc, ret_pct

def upload_chart(token, img_bytes):
    """上传到 GitHub, 返回 CDN URL"""
    api = f'https://api.github.com/repos/{REPO}/contents/YH02/live_chart.png'
    ctx = ssl._create_unverified_context()
    headers = {'Authorization': 'Bearer ' + token, 'User-Agent': 'YH02-daily'}

    sha = None
    try:
        req = ur.Request(api, headers=headers)
        r = json.loads(ur.urlopen(req, timeout=10, context=ctx).read())
        sha = r.get('sha')
    except:
        pass

    body = json.dumps({
        'message': 'daily chart',
        'content': base64.b64encode(img_bytes).decode('ascii'),
        'branch': 'main',
        **({'sha': sha} if sha else {})
    }).encode()
    ur.urlopen(ur.Request(api, data=body, headers={**headers, 'Content-Type': 'application/json'}, method='PUT'),
               timeout=15, context=ctx)

    ts = int(time.time())
    return f'https://cdn.jsdelivr.net/gh/{REPO}@main/YH02/live_chart.png?t={ts}'

def send_bark(title, body, chart_url):
    """推送到手机Bark (带图片)"""
    data = json.dumps({'title': title, 'body': body, 'url': chart_url}).encode()
    try:
        ur.urlopen(ur.Request(f'https://api.day.app/{BARK_KEY}', data=data,
                   headers={'Content-Type': 'application/json'}), timeout=10)
        print("已推送到手机")
    except Exception as e:
        print(f"推送失败: {e}")

def main():
    # 1. 数据
    df = ak.fund_etf_hist_sina(symbol=ETF_SYMBOL)
    df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date').reset_index(drop=True)
    df = compute_indicators(df)

    # 2. 生成图表
    print("生成图表...")
    img_bytes, sig, price, rsi, bb_pos, trend, upper_acc, price_acc, ret_pct = gen_chart(df)

    # 3. 上传 GitHub
    token = os.environ.get('GH_TOKEN', '')
    if not token:
        # 尝试从文件读取
        try:
            token = open('github_token.txt').read().strip()
        except:
            pass

    chart_url = ''
    if token:
        print("上传图表...")
        chart_url = upload_chart(token, img_bytes)
        print(f"  {chart_url}")
    else:
        print("无 GitHub Token, 跳过图表上传")

    # 4. 推送
    body = (f'价格 {price:.4f}  RSI {rsi:.1f}  BB {bb_pos:.0f}%  {trend}\n'
            f'近半年收益 {ret_pct:+.1f}%\n'
            f'上轨加速度 {upper_acc:+.5f}  价格加速度 {price_acc:+.5f}')
    send_bark(f'{ETF_NAME} {sig}', body, chart_url)

if __name__ == '__main__':
    main()
