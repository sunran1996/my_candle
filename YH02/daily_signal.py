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

    # ===== 浅色主题 (A股配色: 红涨绿跌) =====
    bg = '#FFFFFF'
    card_bg = '#F8F9FA'
    fg = '#1A1A2E'
    sub = '#6B7280'
    up_c = '#DC2626'; down_c = '#16A34A'  # 中国: 红涨绿跌
    border = '#E5E7EB'

    if sig == '买入': accent_c = up_c; accent_bg = '#FEF2F2'
    elif sig == '卖出': accent_c = down_c; accent_bg = '#F0FDF4'
    else: accent_c = '#6B7280'; accent_bg = '#F9FAFB'

    # 6×10.5 inches, 适配 iPhone
    fig = plt.figure(figsize=(6, 10.5), facecolor=bg)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.8, 1.8, 3.0], hspace=0.25,
                          left=0.06, right=0.94, top=0.82, bottom=0.02)

    # ── P1: 信号卡 ──
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(bg); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

    # 标题 + 日期
    ax.text(0.3, 9.0, f'{ETF_NAME}({ETF_SYMBOL})', fontsize=20, fontweight='bold', color=fg)
    ax.text(0.3, 8.2, r['date'].strftime('%Y/%m/%d'), fontsize=9, color=sub)

    # 价格: 红涨绿跌 + 涨跌幅
    day_chg = (price - prev['close']) / prev['close'] * 100
    price_c = up_c if day_chg >= 0 else down_c
    ax.text(9.7, 9.2, f'{price:.3f}', fontsize=24, fontweight='bold', color=price_c, ha='right')
    ax.text(9.7, 8.2, f'{day_chg:+.2f}%', fontsize=10, color=price_c, ha='right')

    # 指标信息 — 两列对齐
    info_font = 10
    info_y = 6.2
    c1, c2 = 0.5, 5.5
    row_gap = 0.9
    ax.text(c1, info_y,       f'RSI {rsi:.0f}',           fontsize=info_font, color=fg)
    ax.text(c2, info_y,       f'BB {bb_pos:.0f}%',         fontsize=info_font, color=fg)
    ax.text(c1, info_y-row_gap,   f'上轨 {r["upper"]:.3f}',    fontsize=info_font, color=fg)
    ax.text(c2, info_y-row_gap,   f'下轨 {r["lower"]:.3f}',    fontsize=info_font, color=fg)
    ax.text(c1, info_y-row_gap*2, f'上轨加速 {upper_acc:+.4f}', fontsize=info_font, color=fg)
    ax.text(c2, info_y-row_gap*2, f'价格加速 {price_acc:+.4f}', fontsize=info_font, color=fg)
    ax.text(c1, info_y-row_gap*3, f'趋势 {trend}',              fontsize=info_font, color=fg)
    ax.text(c2, info_y-row_gap*3, f'近{lookback}日 {ret_lookback:+.1f}%', fontsize=info_font,
            color=up_c if ret_lookback>=0 else down_c, fontweight='bold')

    # 操作建议 — 拉开间距
    ax.text(0.5, 1.0, '操作建议：', fontsize=13, color=fg)
    ax.text(3.0, 1.0, sig, fontsize=15, fontweight='bold', color='#000000', va='baseline')

    # ── P2: 净值曲线 ──
    ax = fig.add_subplot(gs[1])
    ax.set_facecolor(bg)
    line_c = up_c if nvs[-1] >= 1 else down_c
    ax.fill_between(range(len(nvs)), 1.0, nvs, alpha=0.1, color=line_c)
    ax.plot(range(len(nvs)), nvs, color=line_c, lw=2.0)
    ax.scatter(len(nvs)-1, nvs[-1], color=line_c, s=50, zorder=5)
    ax.axhline(y=1.0, color=border, lw=1, ls='--')
    ax.set_xlim(-0.5, len(nvs)-0.5)
    ymin = min(0.85, min(nvs)*0.98); ymax = max(1.15, max(nvs)*1.02)
    ax.set_ylim(ymin, ymax)

    # 标注峰值和谷值
    peak_idx = np.argmax(nvs); valley_idx = np.argmin(nvs)
    peak_val = nvs[peak_idx]; valley_val = nvs[valley_idx]
    ax.annotate(f'{(peak_val-1)*100:+.1f}%', xy=(peak_idx, peak_val),
                xytext=(peak_idx, peak_val + (ymax-ymin)*0.06), fontsize=9, color=up_c,
                ha='center', fontweight='bold')
    ax.annotate(f'{(valley_val-1)*100:+.1f}%', xy=(valley_idx, valley_val),
                xytext=(valley_idx, valley_val - (ymax-ymin)*0.06), fontsize=9, color=down_c,
                ha='center', fontweight='bold')
    if len(nvs) >= 2:
        step = max(1, len(dates)//4)
        ticks = list(range(0, len(dates), step))
        if len(dates)-1 not in ticks: ticks.append(len(dates)-1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([dates[t] for t in ticks], fontsize=7, color=sub)
    ax.tick_params(labelsize=7, colors=sub, left=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    for spine in ax.spines.values(): spine.set_color(border); spine.set_linewidth(0.5)
    ax.grid(True, alpha=0.15, color=border)
    ax.set_title(f'净值走势 ({lookback}日)', fontsize=11, fontweight='bold', color=fg, loc='left', pad=6)

    # ── P3: 月度收益看板 ──
    ax = fig.add_subplot(gs[2])
    ax.set_facecolor(card_bg)
    ax.set_xlim(0, 10); ax.set_ylim(-0.5, len(monthly)+1.5); ax.axis('off')

    # 标题行
    ax.text(0, len(monthly)+1, '月度收益看板', fontsize=12, fontweight='bold', color=fg, va='bottom')
    ax.text(10, len(monthly)+1, '策略 / 持有 / 超额', fontsize=8, color=sub, ha='right', va='bottom')

    # 列头
    cols = [2.5, 2.0, 2.5, 3.0]  # 月份, 策略, 持有, 超额
    cx = [0.5]
    for w in cols[:-1]: cx.append(cx[-1] + w)
    for j, (hdr, x) in enumerate(zip(['月份', '策略', '买入持有', '超额收益'], cx)):
        ax.text(x, len(monthly), hdr, fontsize=7, color=sub)

    # 买入持有
    df_m = df.copy(); df_m['month'] = df_m['date'].dt.to_period('M')
    bh_m = df_m.groupby('month').agg(bh_open=('close','first'), bh_close=('close','last'))
    bh_m['bh_ret'] = (bh_m['bh_close'] / bh_m['bh_open'] - 1) * 100
    bh_map = bh_m['bh_ret'].to_dict()

    row_h = 0.85
    for i, (m, row) in enumerate(monthly.iterrows()):
        y = len(monthly) - 0.8 - i * row_h
        bh_ret = bh_map.get(m, 0)
        alpha_v = row['ret'] - bh_ret

        # 斑马纹
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0.2, y - row_h/2 + 0.05), 9.6, row_h,
                         color='#00000004', zorder=0))

        ax.text(cx[0], y, str(m)[-7:], fontsize=8, color=fg)
        ax.text(cx[1], y, f'{row["ret"]:+.1f}%', fontsize=9,
                fontweight='bold', color=up_c if row['ret']>=0 else down_c)
        ax.text(cx[2], y, f'{bh_ret:+.1f}%', fontsize=8, color=sub)
        ax.text(cx[3], y, f'{alpha_v:+.1f}%', fontsize=8, color=up_c if alpha_v>=0 else down_c)

        # 右侧色条
        bar_w = 0.3
        if alpha_v > 0:
            ax.add_patch(plt.Rectangle((9.5, y-row_h/3), bar_w, row_h*2/3, color=up_c, alpha=0.6))
        elif alpha_v < 0:
            ax.add_patch(plt.Rectangle((9.5, y-row_h/3), bar_w, row_h*2/3, color=down_c, alpha=0.4))

    # 边框
    for spine in ax.spines.values(): spine.set_color(border); spine.set_linewidth(0.6)

    fig.text(0.5, 0.005, 'YH02 · 每日自动生成', fontsize=7, color=sub, ha='center')

    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor=bg, edgecolor='none')
    plt.close()
    return buf.getvalue(), sig, price, rsi, bb_pos, trend, upper_acc, price_acc, ret_lookback

def upload_chart(token, img_bytes):
    """上传到 GitHub, 每次用新文件名避免CDN缓存, 同时更新latest"""
    now_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f'chart_{now_str}.png'
    ctx = ssl._create_unverified_context()
    headers = {'Authorization': 'Bearer ' + token, 'User-Agent': 'YH02-daily'}

    # 上传当日文件 (检查是否已存在)
    api = f'https://api.github.com/repos/{REPO}/contents/YH02/{filename}'
    sha_day = None
    try:
        req = ur.Request(api, headers=headers)
        r = json.loads(ur.urlopen(req, timeout=10, context=ctx).read())
        sha_day = r.get('sha')
    except:
        pass
    body = json.dumps({
        'message': f'daily chart {now_str}',
        'content': base64.b64encode(img_bytes).decode('ascii'),
        'branch': 'main',
        **({'sha': sha_day} if sha_day else {})
    }).encode()
    ur.urlopen(ur.Request(api, data=body, headers={**headers, 'Content-Type': 'application/json'}, method='PUT'),
               timeout=15, context=ctx)

    # 同时更新 live_chart.png (覆写, jsDelivr会慢慢刷新)
    api2 = f'https://api.github.com/repos/{REPO}/contents/YH02/live_chart.png'
    sha = None
    try:
        req = ur.Request(api2, headers=headers)
        r = json.loads(ur.urlopen(req, timeout=10, context=ctx).read())
        sha = r.get('sha')
    except:
        pass
    body2 = json.dumps({
        'message': 'daily chart',
        'content': base64.b64encode(img_bytes).decode('ascii'),
        'branch': 'main',
        **({'sha': sha} if sha else {})
    }).encode()
    ur.urlopen(ur.Request(api2, data=body2, headers={**headers, 'Content-Type': 'application/json'}, method='PUT'),
               timeout=15, context=ctx)

    # 返回当日文件URL (新URL不会被CDN缓存)
    return f'https://cdn.jsdelivr.net/gh/{REPO}@main/YH02/{filename}'

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
