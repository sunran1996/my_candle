# -*- coding: utf-8 -*-
"""GitHub Actions 每日自动信号 — 高级量化看板 + 推送到手机Bark"""
import sys, io, os, json, ssl, time, base64, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import akshare as ak, pandas as pd, numpy as np
import urllib.request as ur
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

ETF_SYMBOL = 'sh512890'; ETF_NAME = '红利低波'
BB_PERIOD = 45; BB_STD = 2.0
RSI_PERIOD = 14; RSI_OVERSOLD = 30; RSI_OVERBOUGHT = 70
EXPAND_RSI_SELL = 65; BB_ACCEL_UP = 0.001
BARK_KEY = 'eoq8G58fJtDDFxHjhNueGH'
REPO = 'sunran1996/my_candle'

# 字体
_fonts = [f.name for f in fm.fontManager.ttflist]
CN_SANS = 'WenQuanYi Zen Hei' if 'WenQuanYi Zen Hei' in _fonts else ('SimHei' if 'SimHei' in _fonts else 'DejaVu Sans')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [CN_SANS, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
    df = df.copy(); df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month').agg(open=('adj_close', 'first'), close=('adj_close', 'last'))
    monthly['ret'] = (monthly['close'] / monthly['open'] - 1) * 100
    return monthly.tail(n_months)

def gen_chart(df, nav_start=1_000_000, lookback=180):
    """浅色高级量化看板 — key:value同行"""
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
    trend_cn = '扩张' if expanding else '收缩'

    recent = df.tail(lookback).copy()
    recent['nav'] = nav_start * recent['adj_close'] / recent['adj_close'].iloc[0]
    nvs = (recent['nav'] / nav_start).tolist()
    dates = recent['date'].dt.strftime('%m/%d').tolist()
    ret_lookback = (nvs[-1] - 1) * 100
    day_chg = (price - prev['close']) / prev['close'] * 100
    monthly = month_returns(df, 12)

    # ===== 主题 =====
    BG   = '#FFF0F3'
    CARD = '#FFE4E8'
    FG   = '#1C1C1E'
    SUB  = '#555555'
    UP   = '#CC0000'; DN = '#008800'
    LINE = '#DDDDDD'

    sig_color = {'BUY': UP, 'SELL': DN, 'HOLD': SUB}[sig]

    fig = plt.figure(figsize=(6, 10.5), facecolor=BG)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.3, 1.8, 3.0], hspace=0.25,
                          left=0.08, right=0.92, top=0.96, bottom=0.02)

    # ── P1: 信号卡 ──
    ax = fig.add_subplot(gs[0]); ax.set_facecolor(CARD)
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis('off')

    ax.text(0.3, 11.3, ETF_NAME, fontsize=28, fontweight='bold', color=FG)
    ax.text(0.3, 10.2, r['date'].strftime('%Y.%m.%d'), fontsize=14, color=SUB)
    ax.text(0.3, 9.8, 'YH02  QUANT  STRATEGY', fontsize=7, color=SUB)

    price_c = UP if day_chg >= 0 else DN
    ax.text(9.7, 11.3, f'{price:.3f}', fontsize=28, fontweight='bold', color=price_c, ha='right')
    ax.text(9.7, 10.4, f'{day_chg:+.2f}%', fontsize=11, color=price_c, ha='right')

    # 计算 Sharpe 和 MaxDD
    daily_ret = recent['adj_close'].pct_change().dropna()
    lookback_sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    cum_nav = recent['nav'] / nav_start
    lookback_mdd = ((cum_nav - cum_nav.cummax()) / cum_nav.cummax()).min() * 100

    # 指标信息 — key:value 同行, 两列
    metrics = [
        ('上轨', f'{r["upper"]:.3f}'),      ('下轨', f'{r["lower"]:.3f}'),
        ('上轨加速', f'{upper_acc:+.4f}'),  ('价格加速', f'{price_acc:+.4f}'),
        ('RSI', f'{rsi:.0f}'),              ('BB 位置', f'{bb_pos:.0f}%'),
        ('BB 宽度', f'{(r["upper"]-r["lower"])/r["ma"]*100:.1f}%'),
        ('波动率', f'{daily_ret.std()*100:.2f}%'),
        ('夏普比率', f'{lookback_sharpe:.2f}'), ('最大回撤', f'{lookback_mdd:+.1f}%'),
    ]
    info_font = 11
    info_y0 = 8.5; row_gap = 1.15; col_gap = 5.5
    for idx, (label, value) in enumerate(metrics):
        c, rw = idx % 2, idx // 2
        mx = 0.5 + c * col_gap; my = info_y0 - rw * row_gap
        ax.text(mx, my, f'{label}', fontsize=info_font, color=FG, family='SimSun')
        ax.text(mx + 2.0, my, value, fontsize=info_font, color=FG, fontweight='bold', family='Times New Roman')

    # 信号
    ax.add_patch(plt.Rectangle((0.3, 0.2), 9.4, 1.6, color='#FAD5D5' if sig == 'SELL' else ('#D5F0D8' if sig == 'BUY' else '#F0E8F0'), zorder=0))
    ax.text(0.8, 1.0, sig, fontsize=18, fontweight='bold', color='#000000', family='monospace', va='center')
    ax.text(3.0, 1.0, sig_cn, fontsize=18, fontweight='bold', color='#000000', va='center')

    # ── P2: NAV ──
    ax = fig.add_subplot(gs[1]); ax.set_facecolor(CARD)
    line_c = UP if nvs[-1] >= 1 else DN
    ax.fill_between(range(len(nvs)), 1.0, nvs, alpha=0.08, color=line_c)
    ax.plot(range(len(nvs)), nvs, color=line_c, lw=1.4)
    ax.scatter(len(nvs)-1, nvs[-1], color=line_c, s=40, zorder=5)
    ax.axhline(y=1.0, color=LINE, lw=0.8, ls='--')
    ax.set_xlim(-0.5, len(nvs)-0.5)
    ymin = min(0.85, min(nvs)*0.98); ymax = max(1.15, max(nvs)*1.02)
    ax.set_ylim(ymin, ymax)

    pi = np.argmax(nvs); vi = np.argmin(nvs)
    ax.annotate(f'{(nvs[pi]-1)*100:+.1f}%', xy=(pi, nvs[pi]),
                xytext=(pi, nvs[pi]+(ymax-ymin)*0.07), fontsize=8, color=UP, ha='center', fontweight='bold')
    ax.annotate(f'{(nvs[vi]-1)*100:+.1f}%', xy=(vi, nvs[vi]),
                xytext=(vi, nvs[vi]-(ymax-ymin)*0.07), fontsize=8, color=DN, ha='center', fontweight='bold')

    if len(nvs) >= 2:
        step = max(1, len(dates)//5)
        ticks = list(range(0, len(dates), step))
        if len(dates)-1 not in ticks: ticks.append(len(dates)-1)
        ax.set_xticks(ticks); ax.set_xticklabels([dates[t] for t in ticks], fontsize=7, color='#444444')
    ax.tick_params(labelsize=7, colors='#444444', left=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
    for spine in ax.spines.values(): spine.set_color(LINE); spine.set_linewidth(0.5)
    ax.grid(True, alpha=0.1, color=LINE)
    ax.set_title(f'NAV  {lookback}D', fontsize=10, color='#444444', loc='left', pad=5, family='monospace')

    # ── P3: 月度收益 ──
    ax = fig.add_subplot(gs[2]); ax.set_facecolor(CARD)
    ax.set_xlim(0, 10); ax.set_ylim(-0.5, len(monthly)+1.5); ax.axis('off')

    cols = [2.5, 2.5, 2.5, 2.5]
    cx = [0.3]; [cx.append(cx[-1]+w) for w in cols[:-1]]
    for hdr, x in zip(['MONTH', 'STRATEGY', 'BENCHMARK', 'ALPHA'], cx):
        ax.text(x, len(monthly)+1.2, hdr, fontsize=7, color='#444444', family='monospace')

    df_m = df.copy(); df_m['month'] = df_m['date'].dt.to_period('M')
    bh = df_m.groupby('month').agg(o=('close','first'), c=('close','last'))
    bh['r'] = (bh['c']/bh['o']-1)*100; bh_map = bh['r'].to_dict()

    rh = 0.90
    for i, (m, mr) in enumerate(monthly.iterrows()):
        y = len(monthly) - 0.5 - i * rh
        br = bh_map.get(m, 0); al = mr['ret'] - br
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0.1, y-rh/2+0.05), 9.8, rh, color='#00000003', zorder=0))
        ax.text(cx[0], y, str(m)[-7:], fontsize=9, color=FG)
        ax.text(cx[1], y, f'{mr["ret"]:+.1f}%', fontsize=9, fontweight='bold', color=UP if mr['ret']>=0 else DN)
        ax.text(cx[2], y, f'{br:+.1f}%', fontsize=8, color='#666666')
        ax.text(cx[3], y, f'{al:+.1f}%', fontsize=8, color=UP if al>=0 else DN)
        ax.add_patch(plt.Rectangle((9.5, y-rh/3), 0.2, rh*2/3,
                     color=UP if al>0 else (DN if al<0 else SUB), alpha=0.4))

    for spine in ax.spines.values(): spine.set_color(LINE); spine.set_linewidth(0.5)
    fig.text(0.5, 0.006, 'YH02  ·  QUANT  ·  AUTO  GENERATED',
             fontsize=6, color=SUB, ha='center', family='monospace')

    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close()
    return buf.getvalue(), sig_cn, price, rsi, bb_pos, trend_cn, upper_acc, price_acc, ret_lookback

def upload_chart(token, img_bytes):
    now_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f'chart_{now_str}.png'
    ctx = ssl._create_unverified_context()
    headers = {'Authorization': 'Bearer ' + token, 'User-Agent': 'YH02-daily'}

    api = f'https://api.github.com/repos/{REPO}/contents/YH02/{filename}'
    sha_day = None
    try:
        req = ur.Request(api, headers=headers)
        r = json.loads(ur.urlopen(req, timeout=10, context=ctx).read())
        sha_day = r.get('sha')
    except: pass
    body = json.dumps({
        'message': f'daily chart {now_str}',
        'content': base64.b64encode(img_bytes).decode('ascii'),
        'branch': 'main',
        **({'sha': sha_day} if sha_day else {})
    }).encode()
    ur.urlopen(ur.Request(api, data=body, headers={**headers, 'Content-Type': 'application/json'}, method='PUT'),
               timeout=15, context=ctx)

    api2 = f'https://api.github.com/repos/{REPO}/contents/YH02/live_chart.png'
    sha = None
    try:
        req = ur.Request(api2, headers=headers)
        r = json.loads(ur.urlopen(req, timeout=10, context=ctx).read())
        sha = r.get('sha')
    except: pass
    body2 = json.dumps({
        'message': 'daily chart',
        'content': base64.b64encode(img_bytes).decode('ascii'),
        'branch': 'main',
        **({'sha': sha} if sha else {})
    }).encode()
    ur.urlopen(ur.Request(api2, data=body2, headers={**headers, 'Content-Type': 'application/json'}, method='PUT'),
               timeout=15, context=ctx)

    return f'https://cdn.jsdelivr.net/gh/{REPO}@main/YH02/{filename}'

def send_bark(title, body, chart_url):
    data = json.dumps({'title': title, 'body': body, 'url': chart_url}).encode()
    try:
        ur.urlopen(ur.Request(f'https://api.day.app/{BARK_KEY}', data=data,
                   headers={'Content-Type': 'application/json'}), timeout=10)
        print("已推送到手机")
    except Exception as e:
        print(f"推送失败: {e}")

def main():
    try:
        print("获取数据...")
        df = ak.fund_etf_hist_sina(symbol=ETF_SYMBOL)
        df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date').reset_index(drop=True)
        df = compute_indicators(df)

        today = pd.Timestamp.now().normalize()
        last_date = df['date'].iloc[-1].normalize()
        is_weekend = today.dayofweek >= 5
        data_stale = (today - last_date).days > 2

        non_trading = is_weekend or data_stale
        reason = ''
        if non_trading:
            reason = '周末休市' if is_weekend else '数据未更新'
            print(f"\n{reason} (最新: {df['date'].iloc[-1].strftime('%Y-%m-%d')})")

        print("生成图表...")
        img_bytes, sig, price, rsi, bb_pos, trend, upper_acc, price_acc, ret_pct = gen_chart(df)

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

        prefix = f'{reason} · ' if non_trading else ''
        body = (f'{prefix}价格 {price:.4f}  RSI {rsi:.1f}  BB {bb_pos:.0f}%  {trend}\n'
                f'近半年收益 {ret_pct:+.1f}%\n'
                f'上轨加速度 {upper_acc:+.4f}  价格加速度 {price_acc:+.4f}')
        emoji = {'买入': '🟢', '卖出': '🔴'}.get(sig, '⚪')
        send_bark(f'{ETF_NAME} {emoji} {sig}', body, chart_url)
        print("完成!")
    except Exception as e:
        print(f"执行失败: {e}")
        import traceback; traceback.print_exc()
        send_bark(f'{ETF_NAME} 信号失败', str(e)[:200], '')

if __name__ == '__main__':
    main()
