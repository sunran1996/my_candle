# -*- coding: utf-8 -*-
"""YH_ultra 个股跟踪: 山东高速 600350 + 渝农商行 601077"""
import sys, io, os, json, ssl, base64, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import akshare as ak, pandas as pd, numpy as np
import urllib.request as ur
import matplotlib; matplotlib.use('Agg')
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

_fonts = [f.name for f in fm.fontManager.ttflist]
CN = 'WenQuanYi Zen Hei' if 'WenQuanYi Zen Hei' in _fonts else ('SimHei' if 'SimHei' in _fonts else 'DejaVu Sans')
plt.rcParams['font.sans-serif'] = [CN]; plt.rcParams['axes.unicode_minus'] = False

STOCKS = {
    '山东高速': '600350',
    '渝农商行': '601077',
}
BARK_KEYS = ['eoq8G58fJtDDFxHjhNueGH', 'WtAJhZtoGpU44fAiJCfJmb', 'WdcFKWZiVMyDsiDJqoZrvj']
REPO = 'sunran1996/my_candle'
LOOKBACK = 120  # K线展示天数

def fetch():
    """获取A股日线数据 (新浪源)"""
    dfs = {}
    for name, code in STOCKS.items():
        sym = f'sh{code}'  # 新浪格式
        df = ak.stock_zh_a_daily(symbol=sym, adjust='qfq')
        df = df.rename(columns={'date': 'date', 'open': 'open', 'high': 'high',
                                 'low': 'low', 'close': 'close', 'volume': 'volume'})
        df['date'] = pd.to_datetime(df['date'])
        dfs[name] = df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date').reset_index(drop=True)
    return dfs

def add_indicators(df):
    """添加常用指标"""
    df = df.copy()
    close = df['close']
    df['ma20'] = close.rolling(20).mean()
    df['ma60'] = close.rolling(60).mean()
    df['ma120'] = close.rolling(120).mean()
    # BB(20,2)
    df['bb_ma'] = close.rolling(20).mean()
    df['bb_std'] = close.rolling(20).std()
    df['bb_up'] = df['bb_ma'] + 2 * df['bb_std']
    df['bb_lo'] = df['bb_ma'] - 2 * df['bb_std']
    # RSI(14)
    d = close.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    df['rsi'] = 100 - 100 / (1 + g.ewm(alpha=1/14, adjust=False).mean() /
                   l.ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan))
    # 涨跌幅
    df['chg'] = close.pct_change()
    return df

def send_bark(title, body, url=''):
    if not BARK_KEYS: return
    for bk in BARK_KEYS:
        try:
            data = json.dumps({'title': title, 'body': body, 'url': url}).encode()
            ur.urlopen(ur.Request(f'https://api.day.app/{bk}', data=data,
                       headers={'Content-Type': 'application/json'}), timeout=10)
        except: pass
    print("已推送")

def upload_chart(token, img_bytes):
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    fn = f'chart_{ts}.png'
    ctx = ssl._create_unverified_context()
    h = {'Authorization': 'Bearer ' + token, 'User-Agent': 'YH_ultra'}
    api = f'https://api.github.com/repos/{REPO}/contents/YH_ultra/{fn}'
    try:
        r = json.loads(ur.urlopen(ur.Request(api, headers=h), timeout=10, context=ctx).read())
        sha = r.get('sha')
    except:
        sha = None
    body = json.dumps({'message': 'YH_ultra chart',
                       'content': base64.b64encode(img_bytes).decode('ascii'),
                       'branch': 'main', **({'sha': sha} if sha else {})}).encode()
    ur.urlopen(ur.Request(api, data=body, headers={**h, 'Content-Type': 'application/json'},
                          method='PUT'), timeout=15, context=ctx)
    return f'https://cdn.jsdelivr.net/gh/{REPO}@main/YH_ultra/{fn}'

def main():
    try:
        print("获取数据...")
        raw = fetch()
        dfs = {n: add_indicators(d) for n, d in raw.items()}

        # 实时行情
        is_weekend = pd.Timestamp.now().dayofweek >= 5
        if not is_weekend:
            try:
                spot = ak.stock_zh_a_spot_em()
                for name, code in STOCKS.items():
                    s = spot[spot['代码'] == code]
                    if len(s) > 0:
                        rt = float(s['最新价'].iloc[0])
                        old = raw[name]['close'].iloc[-1]
                        raw[name].loc[raw[name].index[-1], 'close'] = rt
                        raw[name].loc[raw[name].index[-1], 'date'] = pd.Timestamp.now()
                        print(f'  {name} {old:.2f} → 实时 {rt:.2f}')
                dfs = {n: add_indicators(d) for n, d in raw.items()}
            except Exception as e:
                print(f'  实时行情失败: {e}')

        # ── 当前行情快照 ──
        lines = []
        chg_list = []
        for name in STOCKS:
            d = dfs[name]
            last = d.iloc[-1]
            px = last['close']
            ma20 = last['ma20']; ma60 = last['ma60']; ma120 = last['ma120']
            rsi = last['rsi']; bb_lo = last['bb_lo']; bb_up = last['bb_up']
            bb_pos = (px - bb_lo) / (bb_up - bb_lo) * 100 if bb_up > bb_lo else 50
            chg = d['chg'].iloc[-1] * 100 if not pd.isna(d['chg'].iloc[-1]) else 0
            chg_list.append(chg)

            # 均线状态
            above_ma20 = px >= ma20 if not pd.isna(ma20) else None
            above_ma60 = px >= ma60 if not pd.isna(ma60) else None
            above_ma120 = px >= ma120 if not pd.isna(ma120) else None

            lines.append(f'{name} {px:.2f} {chg:+.2f}%')
            lines.append(f'  MA20={ma20:.2f} MA60={ma60:.2f} MA120={ma120:.2f}')
            lines.append(f'  RSI={rsi:.0f} BB位置={bb_pos:.0f}%')

        # ── 图表: 双K线 ──
        fig = plt.figure(figsize=(6, 11), facecolor='#FAFAFA')
        gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 2.5, 2.5],
                              hspace=0.28, left=0.06, right=0.94, top=0.97, bottom=0.03)

        cn_colors = mpf.make_marketcolors(up='#CC0000', down='#008800', edge='inherit',
                                           wick='inherit', volume='inherit')
        cn_style = mpf.make_mpf_style(marketcolors=cn_colors, gridstyle='',
                                       rc={'font.sans-serif': [CN], 'axes.unicode_minus': False})

        # P0: 信息栏
        ax0 = fig.add_subplot(gs[0]); ax0.axis('off'); ax0.set_ylim(0, 9)
        today_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
        ax0.text(0, 8.0, f'个股跟踪 {today_str}', fontsize=14, fontweight='bold', color='#1A1A1A')
        y = 6.5
        for line in lines:
            ax0.text(0, y, line, fontsize=10, color='#333')
            y -= 1.2

        for idx, (name, _) in enumerate(STOCKS.items()):
            ohlc = raw[name].tail(LOOKBACK).copy()
            ohlc = ohlc.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                                         'close': 'Close', 'volume': 'Volume'})
            ohlc = ohlc.set_index('date')[['Open', 'High', 'Low', 'Close', 'Volume']]

            ax = fig.add_subplot(gs[idx + 1])
            mpf.plot(ohlc, type='candle', ax=ax, volume=False, style=cn_style)

            # BB band overlay
            bb = dfs[name].tail(LOOKBACK)
            ax.plot(range(len(ohlc)), bb['bb_up'].values, color='#9B59B6', lw=0.6, ls='--', alpha=0.6)
            ax.plot(range(len(ohlc)), bb['bb_lo'].values, color='#9B59B6', lw=0.6, ls='--', alpha=0.6)
            ax.plot(range(len(ohlc)), bb['bb_ma'].values, color='#888', lw=0.8, ls='--', alpha=0.5)
            ax.plot(range(len(ohlc)), bb['ma20'].values, color='#3498DB', lw=1.0, alpha=0.7, label='MA20')
            ax.plot(range(len(ohlc)), bb['ma60'].values, color='#E67E22', lw=1.0, alpha=0.7, label='MA60')

            last = dfs[name].iloc[-1]
            px = last['close']; rsi = last['rsi']
            bb_lo = last['bb_lo']; bb_up = last['bb_up']
            bb_pos = (px - bb_lo) / (bb_up - bb_lo) * 100 if bb_up > bb_lo else 50
            chg = dfs[name]['chg'].iloc[-1] * 100 if not pd.isna(dfs[name]['chg'].iloc[-1]) else 0
            color = '#CC0000' if chg >= 0 else '#008800'
            ax.set_title(f'{name} {px:.2f} {chg:+.2f}%  RSI{rsi:.0f} BB{bb_pos:.0f}%',
                        fontsize=11, loc='left', color=color)
            ax.tick_params(labelsize=7); ax.grid(True, alpha=0.12)
            ax.legend(fontsize=6, loc='upper left')

        buf = io.BytesIO()
        fig.savefig(buf, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
        plt.close(fig)
        img_bytes = buf.getvalue()

        # ── GitHub上传 ──
        token = os.environ.get('GH_TOKEN', '')
        if not token:
            for p in ['../github_token.txt', 'github_token.txt', 'd:/策略/github_token.txt']:
                try: token = open(p).read().strip(); break
                except: pass
        chart_url = ''
        if token: chart_url = upload_chart(token, img_bytes)

        # ── Bark推送 ──
        title = f'个股跟踪 {" ".join(f"{n}{c:+.1f}%" for n,c in zip(STOCKS.keys(),chg_list))}'
        body = '\n'.join(lines)
        send_bark(title, body, chart_url)

        with open('_preview.png', 'wb') as f: f.write(img_bytes)
        print(f"完成! 图表: _preview.png")

    except Exception as e:
        print(f"失败: {e}"); import traceback; traceback.print_exc()
        send_bark('个股跟踪失败', str(e)[:200])

if __name__ == '__main__':
    main()
