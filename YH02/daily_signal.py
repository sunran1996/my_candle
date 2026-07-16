# -*- coding: utf-8 -*-
"""GitHub Actions 每日自动信号 — 生成收益图 + 推送到手机Bark"""
import sys, io, os, json, ssl, time, base64, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import akshare as ak, pandas as pd, numpy as np
import urllib.request as ur
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

# 跨平台中文字体
_fonts = [f.name for f in fm.fontManager.ttflist]
if 'WenQuanYi Zen Hei' in _fonts:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
elif 'SimHei' in _fonts:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
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

def month_returns(df, n_months=12):
    """计算最近N个月的月度收益"""
    df = df.copy()
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month').agg(
        open=('adj_close', 'first'),
        close=('adj_close', 'last')
    )
    monthly['ret'] = (monthly['close'] / monthly['open'] - 1) * 100
    return monthly.tail(n_months)

def gen_chart(df, nav_start=1_000_000, lookback=180):
    """iPhone竖屏收益图: 信号卡 + 净值曲线 + 月度收益表"""
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

    if buy_ok and not sell_ok: sig = '买入'
    elif sell_ok and not buy_ok: sig = '卖出'
    else: sig = '持有'

    trend = '扩张' if expanding else '收缩'

    # 净值曲线数据
    recent = df.tail(lookback).copy()
    recent['nav'] = nav_start * recent['adj_close'] / recent['adj_close'].iloc[0]
    nvs = (recent['nav'] / nav_start).tolist()
    dates = recent['date'].dt.strftime('%m/%d').tolist()
    ret_lookback = (nvs[-1] - 1) * 100

    # 月度收益
    monthly = month_returns(df, 12)

    # ===== iPhone 比例: 6×10 inches =====
    fig = plt.figure(figsize=(6, 10), facecolor='#0d1117')
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 2.0, 2.5], hspace=0.25,
                          left=0.08, right=0.92, top=0.97, bottom=0.03)

    bg = '#0d1117'; fg = '#c9d1d9'; accent = '#58a6ff'; green = '#3fb950'; red = '#f85149'
    card_bg = '#161b22'; border = '#30363d'

    # ── P1: 信号卡 ──
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(bg)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

    # 信号圆点
    if sig == '买入': dot_c = green; bg_c = '#1a3a2a'
    elif sig == '卖出': dot_c = red; bg_c = '#3a1a1a'
    else: dot_c = '#8b949e'; bg_c = '#1a1c20'

    ax.add_patch(plt.Rectangle((0, 0), 10, 10, color=bg_c, zorder=0))
    ax.add_patch(plt.Circle((1.8, 5), 0.45, color=dot_c, zorder=2))
    ax.text(2.6, 5, sig, fontsize=28, fontweight='bold', color=dot_c, va='center')
    ax.text(2.6, 3.2, r['date'].strftime('%Y年%m月%d日'), fontsize=12, color='#8b949e', va='center')
    ax.text(0.3, 8.5, ETF_NAME, fontsize=14, color='#8b949e')
    ax.text(0.3, 7.2, 'YH02', fontsize=11, color='#484f58')

    # 价格和指标
    ax.text(7.0, 8.5, '价格', fontsize=9, color='#8b949e', ha='right')
    ax.text(7.0, 6.8, f'{price:.4f}', fontsize=22, fontweight='bold', color=fg, ha='right')
    ax.text(7.0, 4.5, f'RSI {rsi:.0f}  BB {bb_pos:.0f}%  {trend}', fontsize=11, color='#8b949e', ha='right')
    ax.text(7.0, 3.2, f'{lookback}日 {ret_lookback:+.1f}%', fontsize=13, color=green if ret_lookback>=0 else red, ha='right', fontweight='bold')

    # ── P2: 净值曲线 ──
    ax = fig.add_subplot(gs[1])
    ax.set_facecolor(card_bg)
    ax.fill_between(range(len(nvs)), 1.0, nvs, alpha=0.15, color=green if nvs[-1]>=1 else red)
    ax.plot(range(len(nvs)), nvs, color=fg, lw=2.0)
    ax.scatter(len(nvs)-1, nvs[-1], color=green if nvs[-1]>=1 else red, s=40, zorder=5)
    ax.axhline(y=1.0, color=border, lw=1, ls='--')
    ax.set_xlim(-0.5, len(nvs)-0.5)
    ymin = min(0.85, min(nvs)*0.98); ymax = max(1.15, max(nvs)*1.02)
    ax.set_ylim(ymin, ymax)
    if len(nvs) >= 2:
        ax.set_xticks([0, len(nvs)//2, len(nvs)-1])
        ax.set_xticklabels([dates[0], dates[len(dates)//2], dates[-1]], fontsize=8, color='#8b949e')
    ax.tick_params(labelsize=8, colors='#8b949e', left=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
    for spine in ax.spines.values(): spine.set_color(border); spine.set_linewidth(0.5)
    ax.grid(True, alpha=0.1, color=fg)
    ax.set_title(f'{lookback}日净值', fontsize=11, color='#8b949e', loc='left', pad=4)

    # ── P3: 月度收益看板 ──
    ax = fig.add_subplot(gs[2])
    ax.set_facecolor(card_bg)
    ax.set_xlim(0, 12); ax.set_ylim(-1, len(monthly)); ax.axis('off')

    ax.text(0, len(monthly), '月度收益', fontsize=11, color='#8b949e', va='bottom')
    ax.text(12, len(monthly), 'YH02', fontsize=9, color='#484f58', ha='right', va='bottom')

    col_w = [3.0, 2.2, 2.2, 2.2, 2.2]  # 月份, 本策略, 买入持有, 超额
    col_x = [0]
    for w in col_w[:-1]: col_x.append(col_x[-1] + w)

    headers = ['月份', '策略', '持有', '超额', '信号']
    for j, (hdr, cx) in enumerate(zip(headers, col_x)):
        ax.text(cx, len(monthly)-1, hdr, fontsize=8, color='#484f58', fontweight='bold')

    # 计算买入持有月度收益
    df_m = df.copy()
    df_m['month'] = df_m['date'].dt.to_period('M')
    bh_monthly = df_m.groupby('month').agg(bh_open=('close','first'), bh_close=('close','last'))
    bh_monthly['bh_ret'] = (bh_monthly['bh_close'] / bh_monthly['bh_open'] - 1) * 100
    bh_map = bh_monthly['bh_ret'].to_dict()

    for i, (m, row) in enumerate(monthly.iterrows()):
        y_pos = len(monthly) - 2 - i
        bh_ret = bh_map.get(m, 0)
        alpha_v = row['ret'] - bh_ret
        # 背景条纹
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, y_pos-0.35), 12, 0.7, color='#ffffff04', zorder=0))

        ax.text(col_x[0], y_pos, str(m), fontsize=9, color=fg)
        ax.text(col_x[1], y_pos, f'{row["ret"]:+.1f}%', fontsize=9,
                color=green if row['ret']>=0 else red, fontweight='bold')
        ax.text(col_x[2], y_pos, f'{bh_ret:+.1f}%', fontsize=8, color='#8b949e')
        ax.text(col_x[3], y_pos, f'{alpha_v:+.1f}%', fontsize=8,
                color=green if alpha_v>=0 else red)
        # 信号摘要
        sig_count = len(df[(df['date'].dt.to_period('M') == m)])
        if row['ret'] > 2: s = '★'
        elif row['ret'] > 0: s = '↑'
        elif row['ret'] > -2: s = '—'
        else: s = '↓'
        ax.text(col_x[4], y_pos, s, fontsize=9, color=fg, ha='center')

    # 分割线
    for spine in ax.spines.values(): spine.set_color(border); spine.set_linewidth(0.5)

    # 底部署名
    fig.text(0.5, 0.01, 'YH02 · 自动生成 · 投资有风险', fontsize=7, color='#484f58',
             ha='center', va='bottom')

    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor=bg, edgecolor='none')
    plt.close()
    return buf.getvalue(), sig, price, rsi, bb_pos, trend, upper_acc, price_acc, ret_lookback

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
    # jsDelivr国内能访问, 加时间戳绕过缓存
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
    try:
        # 1. 数据
        print("获取数据...")
        df = ak.fund_etf_hist_sina(symbol=ETF_SYMBOL)
        df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date').reset_index(drop=True)
        df = compute_indicators(df)

        # 2. 生成图表
        print("生成图表...")
        img_bytes, sig, price, rsi, bb_pos, trend, upper_acc, price_acc, ret_pct = gen_chart(df)

        # 3. 上传 GitHub
        token = os.environ.get('GH_TOKEN', '')
        if not token:
            for p in ['../github_token.txt', 'github_token.txt', 'd:/策略/github_token.txt']:
                try:
                    token = open(p).read().strip()
                    if token: break
                except: pass
        chart_url = ''
        if token:
            print("上传图表...")
            chart_url = upload_chart(token, img_bytes)
            print(f"  {chart_url}")
        else:
            print("无 GH_TOKEN, 跳过图表上传")

        # 4. 推送 — url参数让通知可点击查看收益图
        body = (f'价格 {price:.4f}  RSI {rsi:.1f}  BB {bb_pos:.0f}%  {trend}\n'
                f'近半年收益 {ret_pct:+.1f}%\n'
                f'上轨加速度 {upper_acc:+.5f}  价格加速度 {price_acc:+.5f}')
        emoji = {'买入': '🟢', '卖出': '🔴'}.get(sig, '⚪')
        send_bark(f'{ETF_NAME} {emoji} {sig}', body, chart_url)
        print("完成!")
    except Exception as e:
        print(f"执行失败: {e}")
        import traceback; traceback.print_exc()
        # 至少发一条失败通知
        send_bark(f'{ETF_NAME} 信号失败', str(e)[:200], '')

if __name__ == '__main__':
    main()
