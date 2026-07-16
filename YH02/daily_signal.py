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

# 跨平台字体
_fonts = [f.name for f in fm.fontManager.ttflist]
CN_SANS = 'WenQuanYi Zen Hei' if 'WenQuanYi Zen Hei' in _fonts else ('SimHei' if 'SimHei' in _fonts else 'DejaVu Sans')
CN_SERIF = 'SimSun' if 'SimSun' in _fonts else CN_SANS  # 宋体
# 全局黑体, 信息区单独用SimSun
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [CN_SANS, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 字体方案: 标题/操作建议用黑体(sans-serif), 信息区用SimSun(自带衬线英文≈Times New Roman)

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
    """高级量化风格看板"""
    r = df.iloc[-1]; prev = df.iloc[-2]
    price = r['close']; rsi = r['rsi']
    bb_pos = (r['adj_close'] - r['lower']) / (r['upper'] - r['lower']) * 100
    expanding = (r['upper'] - r['lower']) / r['ma'] > (prev['upper'] - prev['lower']) / prev['ma']
    upper_acc = r['upper_acc'] if not np.isnan(r['upper_acc']) else 0
    price_acc = r['price_acc'] if not np.isnan(r['price_acc']) else 0

    if expanding:
        buy_ok = (r['adj_close'] <= r['lower'] or rsi <= RSI_OVERSOLD)
        sell_raw = r['adj_close'] >= r['upper'] and rsi >= EXPAND_RSI_SELL
        sell_ok = sell_raw and not (upper_acc > BB_ACCEL_UP and price_acc > 0)
    else:
        buy_ok = (r['adj_close'] <= r['lower'] or rsi <= RSI_OVERSOLD)
        sell_ok = (r['adj_close'] >= r['upper'] or rsi >= RSI_OVERBOUGHT)

    if buy_ok and not sell_ok: sig = 'BUY'
    elif sell_ok and not buy_ok: sig = 'SELL'
    else: sig = 'HOLD'

    sig_cn = {'BUY': '买入', 'SELL': '卖出', 'HOLD': '持有'}[sig]
    trend = 'EXPANSION' if expanding else 'CONTRACTION'
    trend_cn = '扩张' if expanding else '收缩'

    recent = df.tail(lookback).copy()
    recent['nav'] = nav_start * recent['adj_close'] / recent['adj_close'].iloc[0]
    nvs = (recent['nav'] / nav_start).tolist()
    dates = recent['date'].dt.strftime('%m/%d').tolist()
    ret_lookback = (nvs[-1] - 1) * 100
    day_chg = (price - prev['close']) / prev['close'] * 100

    monthly = month_returns(df, 12)

    # ===== 高级深色主题 =====
    BG  = '#0B0E14'
    CARD= '#131720'
    FG  = '#E8E6E3'
    SUB = '#6B7280'
    GOLD= '#C9A96E'
    UP  = '#E0524B'; DN = '#34A853'
    LINE= '#1F2937'

    sig_color = { 'BUY': UP, 'SELL': DN, 'HOLD': '#9CA3AF' }[sig]
    sig_bg    = { 'BUY': '#1C1113', 'SELL': '#0F1A12', 'HOLD': '#15171A' }[sig]

    fig = plt.figure(figsize=(6, 11), facecolor=BG)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 2.0, 3.5], hspace=0.3,
                          left=0.08, right=0.92, top=0.96, bottom=0.02)

    # ═══════ P1: HEADER ═══════
    ax = fig.add_subplot(gs[0]); ax.set_facecolor(BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis('off')

    # 标题行 (下移, 远离顶线)
    ax.text(0.2, 11.2, ETF_NAME, fontsize=24, fontweight='bold', color=FG)
    ax.text(4.5, 11.2, ETF_SYMBOL.upper(), fontsize=11, color=SUB, va='baseline')
    ax.text(0.2, 10.4, r['date'].strftime('%Y.%m.%d'), fontsize=10, color=SUB)
    ax.text(0.2, 9.6, 'YH02  QUANT  STRATEGY', fontsize=7, color=SUB)

    # 顶部分割线
    ax.axhline(y=11.6, color=GOLD, lw=1.2, xmin=0.02, xmax=0.98)

    # 右侧: 价格大字
    price_c = UP if day_chg >= 0 else DN
    ax.text(9.8, 11.0, f'{price:.3f}', fontsize=28, fontweight='bold', color=price_c, ha='right')
    ax.text(9.8, 10.2, f'{day_chg:+.2f}%', fontsize=11, color=price_c, ha='right')

    # 信号条
    ax.add_patch(plt.Rectangle((0.2, 6.8), 9.6, 1.8, color=sig_bg, zorder=0))
    ax.text(0.8, 7.7, sig, fontsize=11, fontweight='bold', color=sig_color,
            family='monospace', va='center')
    ax.text(2.8, 7.7, sig_cn, fontsize=14, fontweight='bold', color=sig_color, va='center')
    ax.text(6.0, 7.7, f'RSI {rsi:.0f}    BB {bb_pos:.0f}%    {trend_cn}',
            fontsize=10, color=SUB, va='center', family='SimSun')

    # 指标网格 (2x4)
    metrics = [
        ('上轨', f'{r["upper"]:.3f}'), ('下轨', f'{r["lower"]:.3f}'),
        ('上轨加速', f'{upper_acc:+.4f}'), ('价格加速', f'{price_acc:+.4f}'),
        ('BB 宽度', f'{(r["upper"]-r["lower"])/r["ma"]*100:.1f}%'),
        ('波动率', f'{recent["adj_close"].pct_change().std()*100:.2f}%'),
        ('近{lookback}日收益'.format(lookback=lookback), f'{ret_lookback:+.1f}%'),
        ('当日涨跌', f'{day_chg:+.2f}%'),
    ]
    for idx, (label, value) in enumerate(metrics):
        col, row = idx % 2, idx // 2
        mx = 0.5 + col * 5.0; my = 5.5 - row * 1.2
        vc = UP if value.startswith('+') else (DN if value.startswith('-') else FG)
        ax.text(mx, my, label, fontsize=10, color=SUB, family='SimSun')
        ax.text(mx, my-0.55, value, fontsize=10, color=vc, fontweight='bold')

    # ═══════ P2: NAV CHART ═══════
    ax = fig.add_subplot(gs[1]); ax.set_facecolor(CARD)
    line_c = UP if nvs[-1] >= 1 else DN
    ax.fill_between(range(len(nvs)), 1.0, nvs, alpha=0.12, color=line_c)
    ax.plot(range(len(nvs)), nvs, color=line_c, lw=2.2)
    ax.scatter(len(nvs)-1, nvs[-1], color=line_c, s=60, zorder=5)
    ax.axhline(y=1.0, color=LINE, lw=0.8, ls='--')
    ax.set_xlim(-0.5, len(nvs)-0.5)
    ymin = min(0.85, min(nvs)*0.98); ymax = max(1.15, max(nvs)*1.02)
    ax.set_ylim(ymin, ymax)

    peak_idx = np.argmax(nvs); valley_idx = np.argmin(nvs)
    ax.annotate(f'{(nvs[peak_idx]-1)*100:+.1f}%', xy=(peak_idx, nvs[peak_idx]),
                xytext=(peak_idx, nvs[peak_idx]+(ymax-ymin)*0.06), fontsize=8, color=UP, ha='center', fontweight='bold')
    ax.annotate(f'{(nvs[valley_idx]-1)*100:+.1f}%', xy=(valley_idx, nvs[valley_idx]),
                xytext=(valley_idx, nvs[valley_idx]-(ymax-ymin)*0.06), fontsize=8, color=DN, ha='center', fontweight='bold')

    if len(nvs) >= 2:
        step = max(1, len(dates)//5)
        ticks = list(range(0, len(dates), step))
        if len(dates)-1 not in ticks: ticks.append(len(dates)-1)
        ax.set_xticks(ticks); ax.set_xticklabels([dates[t] for t in ticks], fontsize=6, color=SUB)
    ax.tick_params(labelsize=6, colors=SUB, left=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
    for spine in ax.spines.values(): spine.set_color(LINE); spine.set_linewidth(0.5)
    ax.grid(True, alpha=0.08, color=FG)
    ax.set_title(f'NAV  {lookback}D', fontsize=10, color=SUB, loc='left', pad=6, family='monospace')

    # ═══════ P3: MONTHLY RETURNS ═══════
    ax = fig.add_subplot(gs[2]); ax.set_facecolor(CARD)
    ax.set_xlim(0, 10); ax.set_ylim(-0.5, len(monthly)+1.5); ax.axis('off')

    ax.text(0.3, len(monthly)+1, 'MONTHLY RETURNS', fontsize=10, color=SUB, family='monospace')
    ax.text(9.7, len(monthly)+1, '月度收益', fontsize=9, color=SUB, ha='right', family='SimSun')

    # 列头
    cols = [2.5, 2.5, 2.5, 2.5]
    cx = [0.3]; [cx.append(cx[-1]+w) for w in cols[:-1]]
    headers = ['MONTH', 'STRATEGY', 'BENCHMARK', 'ALPHA']
    for hdr, x in zip(headers, cx):
        ax.text(x, len(monthly)+0.2, hdr, fontsize=6, color=SUB, family='monospace')

    df_m = df.copy(); df_m['month'] = df_m['date'].dt.to_period('M')
    bh_m = df_m.groupby('month').agg(o=('close','first'), c=('close','last'))
    bh_m['r'] = (bh_m['c']/bh_m['o']-1)*100; bh_map = bh_m['r'].to_dict()

    row_h = 0.90
    for i, (m, mrow) in enumerate(monthly.iterrows()):
        y = len(monthly) - 0.5 - i * row_h
        bh_ret = bh_map.get(m, 0); alpha_v = mrow['ret'] - bh_ret

        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0.1, y-row_h/2+0.05), 9.8, row_h, color='#FFFFFF03', zorder=0))

        ax.text(cx[0], y, str(m)[-7:], fontsize=8, color=FG)
        ax.text(cx[1], y, f'{mrow["ret"]:+.1f}%', fontsize=8, fontweight='bold',
                color=UP if mrow['ret']>=0 else DN)
        ax.text(cx[2], y, f'{bh_ret:+.1f}%', fontsize=7, color=SUB)
        ax.text(cx[3], y, f'{alpha_v:+.1f}%', fontsize=7, color=UP if alpha_v>=0 else DN)

        bar_w = 0.25
        ax.add_patch(plt.Rectangle((9.5, y-row_h/3), bar_w, row_h*2/3,
                     color=UP if alpha_v>0 else (DN if alpha_v<0 else SUB), alpha=0.5))

    for spine in ax.spines.values(): spine.set_color(LINE); spine.set_linewidth(0.5)

    # Footer
    fig.text(0.5, 0.006, 'YH02  ·  QUANT  ·  AUTO  GENERATED  ·  NOT  FINANCIAL  ADVICE',
             fontsize=6, color=SUB, ha='center', family='monospace')

    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close()
    return buf.getvalue(), sig_cn, price, rsi, bb_pos, trend_cn, upper_acc, price_acc, ret_lookback

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

        # 检测是否交易日 (周末 + 数据太旧)
        today = pd.Timestamp.now().normalize()
        last_date = df['date'].iloc[-1].normalize()
        is_weekend = today.dayofweek >= 5
        data_stale = (today - last_date).days > 2

        if is_weekend or data_stale:
            # 非交易日: 推送前一日持仓摘要
            r = df.iloc[-1]; prev = df.iloc[-2]
            price = r['close']; rsi = r['rsi']
            bb_pos = (r['adj_close'] - r['lower']) / (r['upper'] - r['lower']) * 100
            bb_w = (r['upper'] - r['lower']) / r['ma'] * 100
            expanding = bb_w > (prev['upper'] - prev['lower']) / prev['ma'] * 100
            trend = '扩张' if expanding else '收缩'

            bb_buy = r['adj_close'] <= r['lower']; bb_sell = r['adj_close'] >= r['upper']
            rsi_buy = rsi <= RSI_OVERSOLD
            upper_acc = r['upper_acc'] if not np.isnan(r['upper_acc']) else 0
            price_acc = r['price_acc'] if not np.isnan(r['price_acc']) else 0

            if expanding:
                buy_ok = (bb_buy or rsi_buy)
                sell_ok = (bb_sell and rsi >= EXPAND_RSI_SELL) and not ((upper_acc > BB_ACCEL_UP) and (price_acc > 0))
            else:
                buy_ok = (bb_buy or rsi_buy)
                sell_ok = (bb_sell or rsi >= RSI_OVERBOUGHT)

            if buy_ok and not sell_ok: sig = '买入'
            elif sell_ok and not buy_ok: sig = '卖出'
            else: sig = '持有'

            reason = '周末休市' if is_weekend else '数据未更新(可能节假日)'
            print(f"\n{reason} (最新: {r['date'].strftime('%Y-%m-%d')})")
            print(f"信号: {sig}  价格: {price:.4f}  RSI: {rsi:.1f}  BB: {bb_pos:.0f}%  {trend}")

            body = (f'{reason}\n'
                    f'前次交易日: {r["date"].strftime("%m月%d日")}\n'
                    f'持仓建议: {sig}\n'
                    f'价格 {price:.4f}  RSI {rsi:.1f}  {trend}')
            send_bark(f'{ETF_NAME} {reason} | {sig}', body, '')
            return

        # 2. 生成图表 (交易日)
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
