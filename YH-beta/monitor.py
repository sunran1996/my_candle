# -*- coding: utf-8 -*-
"""
=============================================================================
ETF轮动策略 — 每日监控脚本
=============================================================================
功能: 获取三大ETF最新数据, 计算评分, 输出当日交易建议
用法: python monitor.py
=============================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import json, os, warnings, urllib.request, urllib.error, urllib.parse, ssl, base64
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 配置 (与策略一致)
# ============================================================
INITIAL_CAPITAL = 1_000_000
TOTAL_LAYERS = 10
LAYER_SIZE = INITIAL_CAPITAL / TOTAL_LAYERS
MAX_ETF_LAYERS = 5
COMMISSION = 0.0001
SLIPPAGE = 0.0001
MA_SHORT, MA_MED, MA_LONG = 25, 75, 200
MACD_FAST, MACD_SLOW, MACD_SIG = 13, 26, 9

# 因子权重(可调)
W_MA, W_MACD, W_CAL, W_MOM, W_RSI = 1.0, 1.7, 1.1, 0.9, 0.5

CALENDAR = {
    '沪深300': {'best_wd':[1], 'good_wd':[0,4], 'bad_wd':[2,3], 'best_mo':[2,4,9,11,12], 'bad_mo':[1,3,8]},
    '创业板':  {'best_wd':[0], 'good_wd':[1], 'bad_wd':[3], 'best_mo':[2,5,9], 'bad_mo':[1,7,8]},
    '科创50':  {'best_wd':[3], 'good_wd':[0,2], 'bad_wd':[1,4], 'best_mo':[2,4,5,6,9,10], 'bad_mo':[1,3]},
}

ETF_MAP = {'沪深300': 'sh510300', '创业板': 'sz159915', '科创50': 'sh588000'}
ETF_NAMES = list(ETF_MAP.keys())
POSITION_FILE = 'd:\\策略\\YH-beta\\positions.json'

# ============================================================
# 0. 共享工具函数
# ============================================================
def _get_gh_token():
    """GitHub token: 优先环境变量, 其次本地文件 d:\策略\github_token.txt"""
    t = os.getenv('GITHUB_TOKEN', '')
    if t: return t
    f = 'd:\\策略\\github_token.txt'
    if os.path.exists(f):
        try:
            with open(f, 'r') as fh: return fh.read().strip()
        except: pass
    return ''

def calc_profit_multiplier(nav, total_invested):
    """利润乘数: 亏损时缩小仓位, 盈利时上限1.0不再追涨"""
    profit_ratio = nav / total_invested - 1
    if profit_ratio >= -0.1:  return 1.0   # 盈利或小幅亏损: 不放大
    elif profit_ratio >= -0.2: return 0.85 # 亏损10~20%: 适度缩小
    else:                      return 0.6  # 亏损超20%: 大幅缩小

# ============================================================
# 1. 数据获取
# ============================================================
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def fetch_data():
    """获取三只ETF近300个交易日数据(够算MA200)"""
    data = {}
    for name, sym in ETF_MAP.items():
        try:
            df = ak.fund_etf_hist_sina(symbol=sym)
        except Exception as e:
            print(f"  *** 获取{name}({sym})数据失败: {e}")
            raise RuntimeError(f"无法获取{name}数据, 请检查网络或 akshare 版本") from e
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        # 只保留最近350天(够算所有指标)
        df = df.tail(350).reset_index(drop=True)
        data[name] = df
    print(f"  数据范围: {data['沪深300']['date'].dt.date.iloc[0]} ~ {data['沪深300']['date'].dt.date.iloc[-1]}")
    return data

def compute_indicators(etf_data):
    """计算所有技术指标"""
    for name, df in etf_data.items():
        df['ret'] = df['close'].pct_change()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['weekday'] = df['date'].dt.weekday
        df['td_of_m'] = df.groupby(['year','month']).cumcount() + 1
        df['td_rev'] = df.groupby(['year','month'])['date'].transform(lambda x: x.rank(ascending=False)).astype(int)
        df['maS'] = df['close'].rolling(MA_SHORT).mean()
        df['maM'] = df['close'].rolling(MA_MED).mean()
        df['maL'] = df['close'].rolling(MA_LONG).mean()
        df['maS_slope'] = df['maS'].diff(5) / df['maS'].shift(5)
        df['mom5'] = df['close'].pct_change(5)
        df['mom20'] = df['close'].pct_change(20)
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma20']
        # RSI(14): 比MACD快3~5根K线, 捕捉超买超卖拐点
        delta = df['close'].diff()
        gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi14'] = 100 - 100 / (1 + rs)
        # 波动率指标: 用于极端行情过滤器
        df['vol20'] = df['ret'].rolling(20).std()
        df['vol_ma60'] = df['vol20'].rolling(60).mean()
        df['ema_fast'] = ema(df['close'], MACD_FAST)
        df['ema_slow'] = ema(df['close'], MACD_SLOW)
        df['macd_dif'] = df['ema_fast'] - df['ema_slow']
        df['macd_dea'] = ema(df['macd_dif'], MACD_SIG)
        df['macd_hist'] = df['macd_dif'] - df['macd_dea']
        df['macd_dif_slope'] = df['macd_dif'].diff(3)
        # 标准MACD(12,26,9) 对比
        ef2 = ema(df['close'], 12); es2 = ema(df['close'], 26)
        df['std_dif'] = ef2 - es2
        df['std_dea'] = ema(df['std_dif'], 9)

# ============================================================
# 2. 评分引擎 (与策略完全一致)
# ============================================================
def score_etf(row, name):
    """评分: row是单只ETF的DataFrame行, 不带前缀; 返回(总分, 因子分解字典)"""
    cal = CALENDAR[name]
    wd = int(row['weekday']); mo = int(row['month'])
    close=row['close']; maS=row['maS']; maM=row['maM']; maL=row['maL']; slope=row['maS_slope']

    # MA
    s_ma_raw = 0.0
    ma_detail = []
    if not np.isnan(maS) and close > maS: s_ma_raw += 1.0; ma_detail.append(f'价>MA{MA_SHORT}')
    if not np.isnan(maM) and close > maM: s_ma_raw += 1.0; ma_detail.append(f'价>MA{MA_MED}')
    if not np.isnan(maL) and close > maL: s_ma_raw += 0.5; ma_detail.append(f'价>MA{MA_LONG}')
    if not np.isnan(slope):
        if slope > 0.005: s_ma_raw += 0.5; ma_detail.append('MA斜率↑')
        elif slope > 0.002: s_ma_raw += 0.25
    if not np.isnan(maS) and not np.isnan(maM) and maS > maM: s_ma_raw += 0.5; ma_detail.append('MA金叉')
    s_ma_raw = min(3.5, s_ma_raw)
    s_ma = s_ma_raw * W_MA

    # MACD
    dif=row['macd_dif']; dea=row['macd_dea']; hist=row['macd_hist']
    s_macd_raw = 0.0
    macd_detail = []
    if not np.isnan(dif) and not np.isnan(dea):
        if dif > 0: s_macd_raw += 0.8; macd_detail.append('DIF>0')
        if dif > dea: s_macd_raw += 0.6; macd_detail.append('金叉')
        if not np.isnan(hist) and hist > 0: s_macd_raw += 0.3; macd_detail.append('红柱')
        if not np.isnan(row['macd_dif_slope']) and row['macd_dif_slope'] > 0: s_macd_raw += 0.3; macd_detail.append('DIF↑')
    s_macd_raw = min(2.0, s_macd_raw)
    s_macd = s_macd_raw * W_MACD

    # 日历
    sw = 1.0 if wd in cal['best_wd'] else (0.7 if wd in cal['good_wd'] else (0.0 if wd in cal['bad_wd'] else 0.5))
    sm = 1.0 if mo in cal['best_mo'] else (0.0 if mo in cal['bad_mo'] else 0.5)
    sp = 0.5*(1 if row['td_of_m']<=3 else 0) + 0.5*(1 if row['td_rev']<=3 else 0)
    s_cal_raw = min(3.0, sw+sm+sp)
    s_cal = s_cal_raw * W_CAL
    cal_detail = []
    if wd in cal['best_wd']: cal_detail.append('最佳星期')
    elif wd in cal['good_wd']: cal_detail.append('好星期')
    if mo in cal['best_mo']: cal_detail.append('最佳月份')
    if row['td_of_m']<=3: cal_detail.append('月初')
    if row['td_rev']<=3: cal_detail.append('月末')

    # 动量
    mom5=row['mom5']; mom20=row['mom20']
    s_mom_raw = 0.5
    mom_detail = [f'mom5={mom5*100:+.1f}%', f'mom20={mom20*100:+.1f}%']
    if not np.isnan(mom5):
        if mom5>0.05: s_mom_raw+=1.0
        elif mom5>0.02: s_mom_raw+=0.6
        elif mom5<-0.05: s_mom_raw-=0.8
        elif mom5<-0.02: s_mom_raw-=0.4
    if not np.isnan(mom20):
        if mom20>0.15: s_mom_raw+=1.0
        elif mom20>0.08: s_mom_raw+=0.7
        elif mom20>0.03: s_mom_raw+=0.3
        elif mom20<-0.10: s_mom_raw-=0.8
        elif mom20<-0.05: s_mom_raw-=0.4
    s_mom_raw = max(0, min(2.5, s_mom_raw))
    s_mom = s_mom_raw * W_MOM

    # RSI(14): 比MACD快3~5根K线, 提前捕捉超买超卖拐点
    rsi = row.get('rsi14', np.nan)
    s_rsi_raw = 0.0
    rsi_detail = []
    if not np.isnan(rsi):
        if rsi < 25:      s_rsi_raw += 2.5; rsi_detail.append(f'深度超卖({rsi:.0f})')
        elif rsi < 35:    s_rsi_raw += 1.5; rsi_detail.append(f'超卖({rsi:.0f})')
        elif rsi < 45:    s_rsi_raw += 0.5; rsi_detail.append(f'偏超卖({rsi:.0f})')
        elif rsi < 55:    pass
        elif rsi < 65:    s_rsi_raw -= 0.3; rsi_detail.append(f'偏超买({rsi:.0f})')
        elif rsi < 75:    s_rsi_raw -= 0.8; rsi_detail.append(f'超买({rsi:.0f})')
        else:             s_rsi_raw -= 1.5; rsi_detail.append(f'深度超买({rsi:.0f})')
    s_rsi_raw = max(-1.5, min(2.5, s_rsi_raw))
    s_rsi = s_rsi_raw * W_RSI

    # 除数 = 五因子加权理论最大值/10, 自动归一化到10分制
    raw_max = 3.5*W_MA + 2.0*W_MACD + 3.0*W_CAL + 2.5*W_MOM + 2.5*W_RSI
    div = raw_max / 10.0
    raw = s_ma + s_macd + s_cal + s_mom + s_rsi
    score = raw / div
    amp = 1.0
    if not np.isnan(maS) and not np.isnan(maM) and not np.isnan(mom20):
        if close > maS and close > maM and mom20 > 0.05:
            amp = 1.25
            score *= amp

    detail = {
        'ma_raw': s_ma_raw, 'ma_weighted': round(s_ma,2), 'ma_detail': ma_detail,
        'macd_raw': s_macd_raw, 'macd_weighted': round(s_macd,2), 'macd_detail': macd_detail,
        'cal_raw': s_cal_raw, 'cal_weighted': round(s_cal,2), 'cal_detail': cal_detail,
        'mom_raw': s_mom_raw, 'mom_weighted': round(s_mom,2), 'mom_detail': mom_detail,
        'rsi_raw': s_rsi_raw, 'rsi_weighted': round(s_rsi,2), 'rsi_detail': rsi_detail,
        'amplifier': amp
    }
    return round(min(10, score), 2), detail

def allocate(scores, portfolio_dd):
    avg_score = np.mean(list(scores.values()))
    raw = {}
    for name, s in scores.items():
        if s >= 7.5: raw[name]=5
        elif s >= 6.0: raw[name]=4
        elif s >= 5.0: raw[name]=3
        elif s >= 4.0: raw[name]=2
        elif s >= 3.5: raw[name]=1
        else: raw[name]=0

    strong_bull = avg_score >= 6.5 and portfolio_dd > -0.08
    mild_bull   = avg_score >= 5.5 and portfolio_dd > -0.12

    boosted = dict(raw); etf_cap = MAX_ETF_LAYERS
    if strong_bull:
        etf_cap = 6
        for name in scores:
            s = scores[name]
            if s >= 7.0: boosted[name]=max(boosted[name],5)
            elif s >= 6.0: boosted[name]=max(boosted[name],4)
            elif s >= 5.0: boosted[name]=max(boosted[name],3)
            elif s >= 4.0: boosted[name]=max(boosted[name],2)
        score_cap = max(4, min(12, int((avg_score-1.5)*2.5)))
    elif mild_bull:
        etf_cap = 6
        for name in scores:
            if scores[name]>=6.0:
                bonus = min(2,6-boosted[name])
                if bonus>0: boosted[name]+=bonus
        score_cap = max(3, min(10, int((avg_score-2.0)*2.5+2)))
    else:
        score_cap = max(1, min(10, int((avg_score-2.0)*2.5)))

    if portfolio_dd <= -0.25: max_total = 1
    elif portfolio_dd <= -0.18: max_total = max(1, score_cap//2)
    else: max_total = min(score_cap, 14)

    total = sum(boosted.values())
    if total > max_total:
        ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)
        target = {n:0 for n in scores}; remaining = max_total
        for n,_ in ranked:
            if remaining<=0: break
            alloc = min(boosted[n], remaining, etf_cap)
            if alloc>0: target[n]=alloc; remaining-=alloc
        return target
    return boosted

# ============================================================
# 3. 持仓管理
# ============================================================
def load_positions():
    if os.path.exists(POSITION_FILE):
        with open(POSITION_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'cash': INITIAL_CAPITAL,
        'total_invested': INITIAL_CAPITAL,
        'shares': {n: 0 for n in ETF_NAMES},
        'cost_basis': {n: 0 for n in ETF_NAMES},
        'peak_nav': INITIAL_CAPITAL,
        'trade_history': []
    }

_last_blank = False
def p(*args, **kwargs):
    global _last_blank
    s = ' '.join(str(a) for a in args) if args else ''
    if s == '' or s.isspace():
        if not _last_blank:
            print()
            _last_blank = True
        return
    print(*args, **kwargs)
    _last_blank = False

def display_positions(pos, etf_data):
    """打印当前持仓详情"""
    if not etf_data:
        return
    shares = pos.get('shares', {})
    cost_basis = pos.get('cost_basis', {})
    cash = pos.get('cash', INITIAL_CAPITAL)
    total_invested = pos.get('total_invested', INITIAL_CAPITAL)

    # 获取最新价格
    prices = {}
    for name in etf_data:
        prices[name] = etf_data[name]['close'].iloc[-1]

    holdings = sum(shares.get(n, 0) * prices[n] for n in prices)
    nav = cash + holdings

    print(f"\n  {''}")
    print(f"  当前持仓详情")
    print(f"  {''}")
    print(f"  现金:         {cash:,.0f} 元")
    print(f"  总投入:       {total_invested:,.0f} 元")
    print(f"  总净值:       {nav:,.0f} 元")
    print(f"  累计收益:     {(nav/total_invested-1)*100:+.2f}%")
    print(f"  仓位占比:     {holdings/nav*100:.1f}%" if nav > 0 else "  仓位占比:     0%")
    p()

    for name in ETF_NAMES:
        s = shares.get(name, 0)
        cb = cost_basis.get(name, 0)
        price = prices.get(name, 0)
        mkt_val = s * price
        cost_val = s * cb
        pnl = (mkt_val - cost_val) / cost_val * 100 if cost_val > 0 else 0
        print(f"  {name:<8}  股数{s:,.0f}  成本价{cb:.3f}  现价{price:.3f}  市值{mkt_val:,.0f}  浮动{pnl:+.1f}%")

def save_positions(pos):
    with open(POSITION_FILE, 'w', encoding='utf-8') as f:
        json.dump(pos, f, ensure_ascii=False, indent=2)

# ============================================================
# 4. 主程序
# ============================================================
def _push_backtest_chart(start_date, days):
    """上传回测图表到GitHub并推送手机"""
    chart_file = 'd:\\策略\\YH-beta\\backtest_chart.png'
    if not os.path.exists(chart_file):
        print("  图表未找到")
        return
    gh_token = _get_gh_token()
    if not gh_token:
        print("  未设置token, 跳过推送")
        return
    try:
        with open(chart_file, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('ascii')
        # 唯一文件名防缓存
        import time
        fname = 'backtest_' + str(int(time.time())) + '.png'
        gh_api = 'https://api.github.com/repos/sunran1996/my_candle/contents/YH-beta/' + fname
        ctx = ssl._create_unverified_context()
        body = {'message': 'backtest', 'content': img_b64}
        urllib.request.urlopen(urllib.request.Request(gh_api,
            data=json.dumps(body, ensure_ascii=True).encode('ascii'),
            headers={'Authorization': 'Bearer ' + gh_token, 'User-Agent': 'gh', 'Content-Type': 'application/json'},
            method='PUT'), timeout=15, context=ctx)
        chart_url = 'https://cdn.jsdelivr.net/gh/sunran1996/my_candle@main/YH-beta/' + fname
        payload = json.dumps({'title': 'YH-beta backtest', 'body': start_date + ' to today', 'url': chart_url}).encode('utf-8')
        urllib.request.urlopen(urllib.request.Request('https://api.day.app/eoq8G58fJtDDFxHjhNueGH',
            data=payload, headers={'Content-Type': 'application/json'}), timeout=10)
        print("  已推送到手机")
    except Exception as e:
        print("  推送失败: " + str(type(e).__name__))

def _send_quick_bark(pos, etf_data, trade_day):
    """快速推送: 仅含持仓摘要和图表, 用于非交易日/重复执行"""
    try:
        bark_url = 'https://api.day.app/eoq8G58fJtDDFxHjhNueGH'
        # 组合估值
        prices = {n: etf_data[n]['close'].iloc[-1] for n in etf_data}
        shares = pos.get('shares', {})
        holdings = sum(shares.get(n, 0) * prices[n] for n in prices)
        nav = pos['cash'] + holdings
        ret = (nav / pos['total_invested'] - 1) * 100

        body = f"净值{nav:,.0f}  收益{ret:+.1f}%  仓位{holdings/nav*100:.1f}%\n"
        body += f"现金{pos['cash']:,.0f}\n"
        for n in ['沪深300', '创业板', '科创50']:
            s = shares.get(n, 0); v = s * prices[n]
            if s > 0:
                body += f"{n}: {v:,.0f}元 ({s:,.0f}股)\n"
        body += f"\n(非交易日/已执行过, 持仓快照)"

        # 保存图表到本地 + 推送文本
        _make_chart_and_save(nav, shares, prices, pos['cash'], trade_day)
        # 文本推送
        chart_url = f'https://cdn.jsdelivr.net/gh/sunran1996/my_candle@main/YH-beta/summary_chart.png?t={trade_day}'
        payload = {'title': f'YH-beta {trade_day} 持仓', 'body': body,
                   'url': chart_url}
        data = json.dumps(payload).encode('utf-8')
        urllib.request.urlopen(urllib.request.Request(bark_url, data=data,
            headers={'Content-Type': 'application/json'}), timeout=10)
        print(f"  [推送] 已发送(点击通知可查看图表)")
    except Exception as e:
        print(f"  [推送] 失败: {e}")

def _make_chart_and_save(nav, shares, prices, cash, trade_day):
    """保存持仓图表到本地 + 上传GitHub"""
    try:
        chart_file = 'd:\\策略\\YH-beta\\summary_chart.png'
        ret = (nav / INITIAL_CAPITAL - 1) * 100
        holdings_val = nav - cash

        # 读取净值历史
        nav_history = []
        pos_file = POSITION_FILE
        if os.path.exists(pos_file):
            try:
                with open(pos_file, 'r', encoding='utf-8') as f:
                    pos_data = json.load(f)
                nav_history = pos_data.get('nav_history', [])
            except Exception:
                pass

        fig = plt.figure(figsize=(12, 8), facecolor='white')
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.35)

        # === 上半: 净值走势 ===
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor('#fafafa')
        if len(nav_history) >= 2:
            dates = [e['date'] for e in nav_history]
            nvs = [e['nav'] / INITIAL_CAPITAL for e in nav_history]
            # 填充区域
            ax1.fill_between(range(len(nvs)), 1.0, nvs,
                             where=[v >= 1.0 for v in nvs],
                             color='#e74c3c', alpha=0.15)
            ax1.fill_between(range(len(nvs)), 1.0, nvs,
                             where=[v < 1.0 for v in nvs],
                             color='#2ecc71', alpha=0.15)
            ax1.plot(range(len(nvs)), nvs, color='#2c3e50', lw=2)
            ax1.axhline(1.0, color='#bdc3c7', lw=0.8, ls='--')
            # 标注首尾日期
            ax1.set_xticks([0, len(nvs)-1])
            ax1.set_xticklabels([dates[0], dates[-1]], fontsize=9)
        else:
            ax1.text(0.5, 0.5, '净值走势\n(运行一段时间后生成)', transform=ax1.transAxes,
                     ha='center', va='center', fontsize=14, color='#bdc3c7')
        ax1.set_title(f'净值走势  累计{ret:+.1f}%', fontsize=13, fontweight='bold', color='#2c3e50')
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('净值', fontsize=10)

        # === 下半: 持仓分布 ===
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor('white')
        ax2.set_xlim(0, 10); ax2.set_ylim(0, 8); ax2.axis('off')

        # 净值大字
        ax2.text(0.5, 7.2, f'{nav:,.0f}', ha='left', fontsize=32,
                color='#2c3e50', fontweight='bold')
        ax2.text(0.5, 6.6, f'元  |  累计{ret:+.1f}%  |  仓位{holdings_val/nav*100:.1f}%  |  现金{cash:,.0f}',
                ha='left', fontsize=10, color='#7f8c8d')

        # 分隔线
        ax2.axhline(y=6.2, xmin=0.03, xmax=0.97, color='#ecf0f1', lw=1)

        # 水平堆叠条
        etf_colors = {'沪深300': '#e74c3c', '创业板': '#2980b9', '科创50': '#e67e22', '现金': '#95a5a6'}
        bar_y, bar_h, left, bar_w = 5.0, 0.6, 0.5, 9.0

        items = []
        for name in ['沪深300', '创业板', '科创50']:
            val = shares.get(name, 0) * prices.get(name, 0)
            if val > 0:
                items.append((name, val, etf_colors[name]))
        if cash > 0:
            items.append(('现金', cash, etf_colors['现金']))
        if not items:
            items.append(('现金', nav, etf_colors['现金']))

        total = sum(v for _, v, _ in items)
        x = left
        for label, val, color in items:
            w = max(val / total * bar_w, 0.2)
            ax2.barh(bar_y, w, bar_h, left=x, color=color, edgecolor='white', lw=1.5)
            pct = val / total * 100
            if pct > 10:
                ax2.text(x + w/2, bar_y, f'{label} {pct:.0f}%', ha='center', va='center',
                        fontsize=10, color='white', fontweight='bold')
            elif pct > 3:
                ax2.text(x + w + 0.1, bar_y, f'{label} {pct:.0f}%', ha='left', va='center',
                        fontsize=9, color='#555555')
            x += w

        # 品种明细
        y = 4.0
        for name in ['沪深300', '创业板', '科创50']:
            s = shares.get(name, 0)
            p = prices.get(name, 0)
            v = s * p
            c = etf_colors[name]
            ax2.plot(1.5, y + 0.15, 'o', color=c, markersize=8)
            if s > 0:
                ax2.text(2.2, y, f'{name}  {s:,.0f}股×{p:.3f}  =  {v:,.0f}元',
                        fontsize=10, color='#2c3e50', va='center')
            else:
                ax2.text(2.2, y, f'{name}  空仓', fontsize=10, color='#bdc3c7', va='center')
            y -= 0.55

        # 保存
        plt.savefig(chart_file, dpi=140, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        # GitHub上传
        gh_token = _get_gh_token()
        if gh_token and os.path.exists(chart_file):
            with open(chart_file, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode('ascii')
            gh_api = 'https://api.github.com/repos/sunran1996/my_candle/contents/YH-beta/summary_chart.png'
            ctx = ssl._create_unverified_context()
            sha = None
            try:
                req = urllib.request.Request(gh_api, headers={'Authorization': f'Bearer {gh_token}', 'User-Agent': 'gh'})
                r = json.loads(urllib.request.urlopen(req, timeout=10, context=ctx).read())
                sha = r.get('sha')
            except Exception:
                pass
            body = {'message': f'YH-beta {trade_day}', 'content': img_b64}
            if sha: body['sha'] = sha
            urllib.request.urlopen(urllib.request.Request(gh_api, data=json.dumps(body).encode('ascii'),
                headers={'Authorization': f'Bearer {gh_token}', 'User-Agent': 'gh', 'Content-Type': 'application/json'},
                method='PUT'), timeout=15, context=ctx)
        # 刷新jsDelivr CDN缓存
        try:
            purge_url = 'https://purge.jsdelivr.net/gh/sunran1996/my_candle@main/YH-beta/summary_chart.png'
            urllib.request.urlopen(purge_url, timeout=5, context=ctx)
        except Exception:
            pass
        print(f"  [GitHub] 图表上传成功")
    except Exception as e:
        print(f"  [GitHub] 上传失败: {e}")
def run_backtest(start_date_str, dca=0):
    """回测模式, dca=每月追加金额"""
    start_date = pd.Timestamp(start_date_str)
    p()
    print(f"  ETF轮动策略 — 回测模式")
    print(f"  区间: {start_date.strftime('%Y-%m-%d')} → 最新交易日")
    print(f"  规则: 每天最多买1个/±2层")
    p()

    # 加载完整数据
    print("\n[1] 加载历史数据...")
    etf_data = {}
    for name, sym in ETF_MAP.items():
        try:
            df = ak.fund_etf_hist_sina(symbol=sym)
        except Exception as e:
            print(f"  *** 获取{name}({sym})历史数据失败: {e}")
            raise RuntimeError(f"无法获取{name}数据, 请检查网络或 akshare 版本") from e
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        etf_data[name] = df
    compute_indicators(etf_data)

    # 检查各ETF数据覆盖情况, ETF可中途加入(如科创50于2020年上市)
    for n in ETF_NAMES:
        first_date = etf_data[n]['date'].iloc[0]
        if first_date > start_date:
            print(f"  ⚠ {n} 最早数据为 {first_date.strftime('%Y-%m-%d')}, 将在该日期后自动加入交易")
    # 取所有ETF日期的并集, 支持ETF动态加入
    date_sets = [set(etf_data[n]['date']) for n in ETF_NAMES]
    all_dates = sorted(set.union(*date_sets))
    trade_dates = [d for d in all_dates if d >= start_date]
    if len(trade_dates) == 0:
        print(f"  错误: {start_date_str} 之后无任何ETF数据")
        return
    print(f"  共 {len(trade_dates)} 个交易日: {trade_dates[0].strftime('%Y-%m-%d')} → {trade_dates[-1].strftime('%Y-%m-%d')}")

    # 初始化(dict覆盖所有品种, 但仅活跃品种参与交易)
    cash = INITIAL_CAPITAL
    total_invested = INITIAL_CAPITAL
    shares = {n: 0 for n in ETF_NAMES}
    cost_basis = {n: 0.0 for n in ETF_NAMES}
    peak_nav = INITIAL_CAPITAL
    trade_log = []
    nav_log = []
    vol_history = []  # 沪深300 vol20 历史, 用于99%分位检测

    # 逐日模拟
    print(f"\n[2] 逐日模拟...")
    for di, d in enumerate(trade_dates):
        # DCA: 每月第一个交易日追加
        if dca > 0 and di > 0:
            prev = trade_dates[di-1]
            if (d.year, d.month) != (prev.year, prev.month):
                cash += dca
                total_invested += dca

        # 获取当日行(各ETF独立判断, 支持中途上市)
        rows = {}
        for name in ETF_NAMES:
            r = etf_data[name][etf_data[name]['date'] == d]
            if r.empty: continue
            rows[name] = r.iloc[0]
        if not rows:  # 无任何品种有数据则跳过
            continue

        # 评分(仅当日有数据的品种)
        scores = {}
        for name in ETF_NAMES:
            if name in rows:
                s, _ = score_etf(rows[name], name)
                scores[name] = s
        # 未上市的品种评分设为0, 不会产生买入
        for n in ETF_NAMES:
            if n not in scores:
                scores[n] = 0.0

        # 组合估值
        holdings = sum(shares[n] * rows[n]['close'] for n in rows if shares.get(n, 0) > 0)
        nav = cash + holdings
        if nav > peak_nav: peak_nav = nav
        dd = (nav - peak_nav) / peak_nav if peak_nav > 0 else 0
        # 层大小: (本金+净值)/2, 净值增长时适度放大但不翻倍
        layer_sz = (total_invested*0.7 + nav*0.3) / TOTAL_LAYERS

        # 利润乘数
        profit_mult = calc_profit_multiplier(nav, total_invested)
        layer_sz *= profit_mult

        # 极端波动过滤器: 固定阈值 + 99%分位检测
        vol_r = rows.get('沪深300', {}).get('vol_ma60', np.nan)
        if not np.isnan(vol_r) and vol_r > 0 and dd < -0.05:
            vol20 = rows['沪深300'].get('vol20', np.nan)
            if not np.isnan(vol20):
                vol_ratio = vol20 / vol_r
                if vol_ratio > 2.5:       vol_adj = 0.6
                elif vol_ratio > 2.0:     vol_adj = 0.75
                elif vol_ratio > 1.5:     vol_adj = 0.9
                else:                     vol_adj = 1.0
                layer_sz *= vol_adj

        # 99%极端行情逆向操作: 暴跌加仓, 暴涨减仓
        hs300 = rows.get('沪深300')
        v20_val = hs300.get('vol20', np.nan) if hs300 is not None else np.nan
        ret_val = hs300.get('ret', np.nan) if hs300 is not None else np.nan
        if not np.isnan(v20_val) and len(vol_history) > 252:
            p99 = np.percentile(vol_history, 99)
            if v20_val > p99 and not np.isnan(ret_val):
                if ret_val < -0.03:       layer_sz *= 1.3   # 暴跌3%+: 逆势加仓
                elif ret_val > 0.03:      layer_sz *= 0.7   # 暴涨3%+: 锁定利润

        target = allocate(scores, dd)

        # 当前层数(仅活跃品种)
        cur = {}
        for n in ETF_NAMES:
            if n in rows:
                cur[n] = round(shares[n] * rows[n]['close'] / layer_sz)
            else:
                cur[n] = 0

        # 模拟交易
        ranked = sorted(ETF_NAMES, key=lambda n: scores[n], reverse=True)
        remaining_cash = cash
        new_cur = dict(cur)

        # 卖出 — 回撤越深卖越快, 真牛市(价在年线上+接近新高)卖慢
        avg_score = np.mean(list(scores.values()))
        # 沪深300是否在年线之上(多头排列)
        above_ma200 = ('沪深300' in rows and
                       rows['沪深300']['close'] > rows['沪深300']['maL'])
        in_bull = (avg_score >= 6.0 and dd > -0.05 and above_ma200)

        if dd <= -0.25:      max_daily_sell = 999  # 深度回撤: 一天清到底
        elif dd <= -0.18:    max_daily_sell = 5    # 中度回撤: 加速减仓
        elif in_bull:        max_daily_sell = 1    # 真牛市: 少卖多拿
        else:                max_daily_sell = 2    # 震荡/熊市: 正常节奏

        for n in ETF_NAMES:
            diff = target[n] - new_cur[n]
            if diff >= 0: continue
            actual = max(-max_daily_sell, diff)
            price = rows[n]['close']
            sell_amount = abs(actual) * layer_sz
            sell_shares = sell_amount / price
            sell_s = min(sell_shares, shares[n])
            sell_val = sell_s * price
            cost = sell_val * (COMMISSION + SLIPPAGE)
            cost_val = sell_s * cost_basis[n] if cost_basis[n] > 0 else 0
            realized_pct = ((sell_val - cost_val - cost) / cost_val * 100) if cost_val > 0 else 0

            remaining_cash += sell_val - cost
            shares[n] -= sell_s
            new_cur[n] += actual
            if shares[n] <= 0: cost_basis[n] = 0.0

            # 交易后重算
            hld = sum(shares[n2] * rows[n2]['close'] for n2 in shares if n2 in rows)
            nv = remaining_cash + hld
            # 各品种持仓市值(未上市品种=0)
            _mkt = lambda nm: shares[nm] * rows[nm]['close'] if nm in rows else 0
            h300 = _mkt('沪深300'); hcyb = _mkt('创业板'); hkc = _mkt('科创50')
            trade_log.append({
                '日期': d.strftime('%Y-%m-%d'), '品种': n, '方向': '卖',
                '层数': abs(actual), '价格': round(price,4),
                '已实现盈亏%': round(realized_pct,2),
                '累计收益率%': round((nv/total_invested-1)*100,2),
                '市值_沪深300': round(h300,0), '市值_创业板': round(hcyb,0), '市值_科创50': round(hkc,0),
                '持仓市值': round(hld,0), '总净值': round(nv,0), '仓位占比%': round(hld/nv*100,1) if nv>0 else 0
            })

        # 买入(只买1个)
        for n in ranked:
            diff = target[n] - new_cur[n]
            if diff <= 0: continue
            desired = min(2, diff)
            # 每层实际成本含手续费, 避免负现金
            cost_per_layer = layer_sz * (1 + COMMISSION + SLIPPAGE)
            affordable = int(remaining_cash / cost_per_layer)
            actual = min(desired, affordable)
            if actual <= 0: continue

            price = rows[n]['close']
            amount = actual * layer_sz
            cost = amount * (COMMISSION + SLIPPAGE)
            trade_shares = amount / price

            old_val = shares[n] * cost_basis[n]
            remaining_cash -= (amount + cost)
            new_val = old_val + amount
            shares[n] += trade_shares
            cost_basis[n] = new_val / shares[n] if shares[n] > 0 else 0
            new_cur[n] += actual

            # 交易后重算
            hld = sum(shares[n2] * rows[n2]['close'] for n2 in shares if n2 in rows)
            nv = remaining_cash + hld
            # 各品种持仓市值(未上市品种=0)
            _mkt = lambda nm: shares[nm] * rows[nm]['close'] if nm in rows else 0
            h300 = _mkt('沪深300'); hcyb = _mkt('创业板'); hkc = _mkt('科创50')
            trade_log.append({
                '日期': d.strftime('%Y-%m-%d'), '品种': n, '方向': '买',
                '层数': actual, '价格': round(price,4),
                '已实现盈亏%': '',
                '累计收益率%': round((nv/total_invested-1)*100,2),
                '市值_沪深300': round(h300,0), '市值_创业板': round(hcyb,0), '市值_科创50': round(hkc,0),
                '持仓市值': round(hld,0), '总净值': round(nv,0), '仓位占比%': round(hld/nv*100,1) if nv>0 else 0
            })
            break  # 只买1个

        cash = remaining_cash
        nav_log.append({'date': d, 'nav': nav, 'holdings': holdings, 'dd': dd})
        # 记录波动率历史(500日滑动窗口)
        if not np.isnan(v20_val):
            vol_history.append(v20_val)
            if len(vol_history) > 500:
                vol_history.pop(0)

    # 最终结果(仅统计最后交易日有数据的品种)
    last_rows = {}
    for n in ETF_NAMES:
        r = etf_data[n][etf_data[n]['date'] == trade_dates[-1]]
        if not r.empty:
            last_rows[n] = r.iloc[0]
    final_nav = cash + sum(shares[n] * last_rows[n]['close'] for n in shares if n in last_rows)
    total_ret = (final_nav / total_invested - 1) * 100
    n_days = len(nav_log)
    ann_ret = ((1 + total_ret/100) ** (252 / n_days) - 1) * 100 if n_days > 0 else 0

    # ================================================================
    # 绩效指标
    # ================================================================
    nav_df = pd.DataFrame(nav_log)
    if len(nav_df) > 1:
        daily_rets = nav_df['nav'].pct_change().dropna()
        nd = len(daily_rets)
        ann_vol = daily_rets.std() * np.sqrt(252) * 100
        sharpe = (ann_ret/100 - 0.02) / (ann_vol/100) if ann_vol > 0 else 0
        cum = nav_df['nav'] / total_invested
        dd_series = (cum - cum.cummax()) / cum.cummax()
        mdd = dd_series.min() * 100
        # 最大回撤区间
        mdd_end_idx = dd_series.idxmin()
        mdd_peak_idx = cum[:mdd_end_idx].idxmax()
        mdd_days = (nav_df['date'].iloc[mdd_end_idx] - nav_df['date'].iloc[mdd_peak_idx]).days
        calmar = ann_ret / abs(mdd) if mdd != 0 else 0
        win_rate = (daily_rets > 0).sum() / nd * 100
        # 索提诺比率(下行波动)
        downside = daily_rets[daily_rets < 0]
        down_vol = downside.std() * np.sqrt(252) * 100 if len(downside) > 0 else ann_vol
        sortino = (ann_ret/100 - 0.02) / (down_vol/100) if down_vol > 0 else 0
        # 盈亏比
        avg_win = daily_rets[daily_rets > 0].mean() if (daily_rets > 0).any() else 0
        avg_loss = abs(daily_rets[daily_rets < 0].mean()) if (daily_rets < 0).any() else 1
        pnl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    else:
        ann_vol = sharpe = mdd = calmar = win_rate = sortino = pnl_ratio = 0
        mdd_days = 0; mdd_peak_idx = mdd_end_idx = 0

    # 交易统计
    trade_df = pd.DataFrame(trade_log)
    buy_count = len(trade_df[trade_df['方向']=='买']) if len(trade_df) > 0 else 0
    sell_count = len(trade_df[trade_df['方向']=='卖']) if len(trade_df) > 0 else 0
    if sell_count > 0:
        sell_pnl = trade_df[trade_df['已实现盈亏%'] != '']['已实现盈亏%'].astype(float)
        avg_sell_pnl = sell_pnl.mean() if len(sell_pnl) > 0 else 0
        sell_win_rate = (sell_pnl > 0).sum() / len(sell_pnl) * 100 if len(sell_pnl) > 0 else 0
    else:
        avg_sell_pnl = sell_win_rate = 0

    # 分年收益
    nav_df = pd.DataFrame(nav_log)
    if len(nav_df) > 1:
        nav_df['year'] = nav_df['date'].dt.year
        yearly = nav_df.groupby('year').agg(
            年末净值=('nav', 'last'), 年初净值=('nav', 'first')
        )
        yearly['年度收益%'] = (yearly['年末净值'] / yearly['年初净值'] - 1) * 100

    # ---- 输出 ----
    print(f"\n  {''}")
    print(f"  {'='*60}")
    print(f"  回测结果")
    print(f"  {'='*60}")
    print(f"  {''}")
    print(f"  ▸ 收益指标")
    print(f"    初始资金:       {INITIAL_CAPITAL:,.0f} 元")
    print(f"    最终净值:       {final_nav:,.0f} 元")
    print(f"    总投入本金:     {total_invested:,.0f} 元")
    print(f"    累计收益率:     {total_ret:+.2f}%")
    print(f"    年化收益率:     {ann_ret:+.2f}%")
    p()
    print(f"  ▸ 风险指标")
    print(f"    年化波动率:     {ann_vol:.2f}%")
    print(f"    下行波动率:     {down_vol:.2f}%")
    print(f"    最大回撤:       {mdd:+.2f}%")
    if mdd != 0 and len(nav_df) > 1:
        print(f"    回撤区间:       {nav_df['date'].iloc[mdd_peak_idx].strftime('%Y-%m-%d')} → {nav_df['date'].iloc[mdd_end_idx].strftime('%Y-%m-%d')} ({mdd_days}天)")
    p()
    print(f"  ▸ 风险调整收益")
    print(f"    夏普比率:       {sharpe:.3f}")
    print(f"    索提诺比率:     {sortino:.3f}")
    print(f"    Calmar比率:     {calmar:.2f}")
    print(f"    盈亏比:         {pnl_ratio:.2f}")
    p()
    print(f"  ▸ 交易统计")
    print(f"    总交易次数:     {len(trade_df)}  (买{buy_count} / 卖{sell_count})")
    print(f"    日均胜率:       {win_rate:.1f}%")
    if sell_count > 0:
        print(f"    卖出胜率:       {sell_win_rate:.1f}%")
        print(f"    均笔卖出盈亏:   {avg_sell_pnl:+.2f}%")
    # 各品种交易分布
    if len(trade_df) > 0:
        for n in ETF_NAMES:
            cnt = len(trade_df[trade_df['品种']==n])
            if cnt > 0:
                print(f"    {n}: {cnt}笔")
    p()
    if len(nav_df) > 1 and len(yearly) > 0:
        print(f"  ▸ 分年度收益")
        for yr, row in yearly.iterrows():
            print(f"    {int(yr)}: {row['年度收益%']:+.2f}%")
    p()
    print(f"  ▸ 最终持仓")
    for n in ETF_NAMES:
        if shares[n] > 0 and n in last_rows:
            val = shares[n] * last_rows[n]['close']
            print(f"    {n}: {shares[n]:,.0f}股 × {last_rows[n]['close']:.3f} = {val:,.0f}元")
    if holdings := sum(shares[n] * last_rows[n]['close'] for n in shares if n in last_rows):
        print(f"    持仓市值{holdings:,.0f}元 / 现金{cash:,.0f}元 / 仓位{holdings/final_nav*100:.1f}%")

    if len(trade_df) > 0:
        csv_path = 'd:\\策略\\YH-beta\\backtest.csv'
        trade_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n  完整交易明细: {csv_path}")
        print(f"\n  {'='*80}")
        print(f"  前10笔 + 后10笔:")
        print(f"  {'='*80}")
        cols = ['日期','品种','方向','层数','价格','已实现盈亏%','累计收益率%','市值_沪深300','市值_创业板','市值_科创50','持仓市值','总净值','仓位占比%']
        display_cols = [c for c in cols if c in trade_df.columns]
        print(trade_df.head(10)[display_cols].to_string(index=False))
        print(f"  ... ({len(trade_df)-20}笔省略) ...")
        print(trade_df.tail(10)[display_cols].to_string(index=False))

    # === K线图: 每日净值 + 日收益 ===
    nav_df = pd.DataFrame(nav_log)
    if len(nav_df) > 1:
        nav_df['daily_ret'] = nav_df['nav'].pct_change() * 100
        nav_df['cum_ret'] = (nav_df['nav'] / INITIAL_CAPITAL - 1) * 100

        # 计算月度收益
        nav_df['year_month'] = nav_df['date'].dt.to_period('M')
        monthly = nav_df.groupby('year_month').agg(
            月初净值=('nav', 'first'), 月末净值=('nav', 'last')
        )
        monthly['月度收益%'] = (monthly['月末净值'] / monthly['月初净值'] - 1) * 100
        months = [str(m) for m in monthly.index]
        month_vals = monthly['月度收益%'].tolist()

        fig, axes = plt.subplots(5, 1, figsize=(10, 16), gridspec_kw={'height_ratios': [3, 1, 1, 1.3, 1.2]})
        fig.suptitle(f'ETF轮动策略 回测 {start_date_str} ~ {trade_dates[-1].strftime("%Y-%m-%d")}', fontsize=18, fontweight='bold', y=0.995)

        # 1. 净值曲线
        ax = axes[0]
        ax.plot(nav_df['date'], nav_df['nav']/INITIAL_CAPITAL, color='#e74c3c', lw=2.5, label=f'累计 {total_ret:+.1f}%')
        ax.axhline(1.0, color='gray', lw=0.8, ls='--')
        ax.fill_between(nav_df['date'], 1.0, nav_df['nav']/INITIAL_CAPITAL,
                        where=nav_df['nav']/INITIAL_CAPITAL>=1.0, color='#e74c3c', alpha=0.1)
        ax.fill_between(nav_df['date'], 1.0, nav_df['nav']/INITIAL_CAPITAL,
                        where=nav_df['nav']/INITIAL_CAPITAL<1.0, color='#2ecc71', alpha=0.1)
        ax.set_ylabel('净值', fontsize=14)
        ax.legend(fontsize=13, loc='upper left')
        ax.grid(True, alpha=0.3)

        # 2. 每日收益率
        ax = axes[1]
        colors = ['#e74c3c' if r >= 0 else '#2ecc71' for r in nav_df['daily_ret'].iloc[1:]]
        ax.bar(nav_df['date'].iloc[1:], nav_df['daily_ret'].iloc[1:], color=colors, width=1.0, alpha=0.8)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_ylabel('日收益(%)', fontsize=14)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:+.1f}%'))
        ax.grid(True, alpha=0.3)

        # 3. 累计收益率
        ax = axes[2]
        ax.fill_between(nav_df['date'], 0, nav_df['cum_ret'], color='#e74c3c', alpha=0.15)
        ax.plot(nav_df['date'], nav_df['cum_ret'], color='#c0392b', lw=1.8)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_ylabel('累计收益(%)', fontsize=14)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:+.0f}%'))
        ax.grid(True, alpha=0.3)

        # 4. 月度收益柱状图(占上半) + 指标&持仓(占下半)
        ax = axes[3]
        n_months = len(months)
        bar_w = max(0.3, min(0.5, 8.0 / n_months))
        ax.set_xlim(-0.5, n_months + 0.5)
        max_abs = max(max(abs(v) for v in month_vals), 3)
        ax.set_ylim(-max_abs * 1.6, max_abs * 1.6)
        colors_m = ['#e74c3c' if v >= 0 else '#2ecc71' for v in month_vals]
        ax.bar(range(n_months), month_vals, color=colors_m, width=bar_w, alpha=0.85, edgecolor='white', lw=0.3)
        ax.axhline(0, color='#666666', lw=1)
        # 标注
        for i, v in enumerate(month_vals):
            if abs(v) > max_abs * 0.25:
                ax.text(i, v + (0.3 if v >= 0 else -0.3) * max_abs / 10, f'{v:+.0f}%',
                        ha='center', va='bottom' if v >= 0 else 'top', fontsize=7.5, color='#444444', fontweight='bold')
        tick_pos, tick_lbl = [], []
        for i, m in enumerate(months):
            if m.endswith('-01'): tick_pos.append(i); tick_lbl.append(m[:4])
        ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lbl, fontsize=10)
        ax.set_ylabel('月度(%)', fontsize=13)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:+.0f}%'))
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, axis='y')


        # 5. 指标 + 持仓 — 深色背景
        ax = axes[4]
        ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis('off')

        ret_color = '#e74c3c' if total_ret >= 0 else '#2ecc71'
        ann_color = '#e74c3c' if ann_ret >= 0 else '#2ecc71'
        kpi_data = [
            (1.2, f'{total_ret:+.1f}%', '累计收益', ret_color),
            (3.5, f'{ann_ret:+.1f}%', '年化收益', ann_color),
            (5.8, f'{mdd:+.1f}%', '最大回撤', '#2ecc71'),
            (8.0, f'{sharpe:.3f}', '夏普比率', '#00d2ff'),
        ]
        for x, val, label, color in kpi_data:
            ax.text(x, 2.2, val, fontsize=22, color=color, fontweight='bold', ha='center')
            ax.text(x, 1.7, label, fontsize=10, color='#2c3e50', ha='center')

        ax.axhline(y=1.4, xmin=0.03, xmax=0.97, color='#ecf0f1', lw=0.8)

        etf_colors = {'沪深300': '#e74c3c', '创业板': '#3498db', '科创50': '#f39c12', '现金': '#555577'}
        items = []
        for name in ETF_NAMES:
            if shares[name] > 0 and name in last_rows:
                items.append((name, shares[name] * last_rows[name]['close'], etf_colors[name]))
        if cash > 0: items.append(('现金', cash, etf_colors['现金']))
        if not items: items.append(('现金', final_nav, etf_colors['现金']))
        total_v = sum(v for _, v, _ in items)
        bar_y, bar_h, bar_l, bar_w = 1.1, 0.5, 0.3, 9.4
        bx = bar_l
        for label, val, color in items:
            w = val / total_v * bar_w if total_v > 0 else bar_w
            ax.barh(bar_y, w, bar_h, left=bx, color=color, edgecolor='white', lw=2.5)
            pct = val / total_v * 100 if total_v > 0 else 0
            if pct > 10:
                ax.text(bx + w/2, bar_y, f'{label} {pct:.0f}%', ha='center', va='center',
                        fontsize=13, color='white', fontweight='bold')
            bx += w
        pos_lines = []
        for n in ETF_NAMES:
            if shares[n] > 0 and n in last_rows:
                pos_lines.append(f'{n}: {shares[n]*last_rows[n]["close"]:,.0f}元')
        pos_lines.append(f'现金: {cash:,.0f}元    仓位: {holdings/final_nav*100:.1f}%')
        pos_lines.append(f'总资产: {final_nav:,.0f}元')
        for i, line in enumerate(pos_lines):
            ax.text(0.3, 0.3 - i * 0.5, line, fontsize=16, color='#2c3e50')

        plt.subplots_adjust(top=0.97, bottom=0.06, hspace=0.3)
        chart_path = 'd:\\策略\\YH-beta\\backtest_chart.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"\n  K线图已保存: {chart_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='ETF轮动策略每日监控')
    parser.add_argument('-r', '--reset', action='store_true', help='清空持仓记录, 从今天开始模拟')
    parser.add_argument('--from', dest='from_date', type=str, default=None, metavar='YYYY-MM-DD', help='从指定日期回测到今天')
    parser.add_argument('--days', type=int, default=0, metavar='N', help='模拟过去N天每日执行, 生成收益曲线')
    parser.add_argument('--push', action='store_true', help='回测/回顾模式也推送到手机')
    parser.add_argument('--dca', type=int, default=0, metavar='N', help='每月追加N元本金(如 --dca 20000)')
    parser.add_argument('--output', type=str, default=None, help='输出到文件(UTF-8)')
    args = parser.parse_args()

    if args.output:
        sys.stdout = open(args.output, 'w', encoding='utf-8')

    # === 回测模式 ===
    if args.from_date or args.days:
        if args.days:
            start = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
        else:
            start = args.from_date
        run_backtest(start, dca=args.dca)
        if args.push:
            _push_backtest_chart(start, args.days or 0)
        return

    if args.reset:
        if os.path.exists(POSITION_FILE):
            os.remove(POSITION_FILE)
            print(f"  已清空: {POSITION_FILE}")
        else:
            print(f"  无需清空(文件不存在)")
        if not args.from_date:
            print("  重置完成.")
            return

    p()
    print("  ETF轮动策略 — 每日监控")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    p()

    # ---- 获取数据 ----
    print("\n[1] 获取数据...")
    etf_data = fetch_data()
    compute_indicators(etf_data)

    # 最新交易日
    today_rows = {}
    for name in etf_data:
        today_rows[name] = etf_data[name].iloc[-1]
    # 优先用沪深300, 不可用时取第一个有数据的ETF
    ref_etf = '沪深300' if '沪深300' in today_rows else next(iter(today_rows))
    latest_trade_day = today_rows[ref_etf]['date'].date()
    today = datetime.now().date()
    print(f"  数据最新交易日: {latest_trade_day}")

    # === 非交易日检测 ===
    is_off_day = False
    if latest_trade_day < today:
        is_off_day = True
        days_gap = (today - latest_trade_day).days
        if today.weekday() >= 5:
            print(f"\n  *** 今天是周末({['周一','周二','周三','周四','周五','周六','周日'][today.weekday()]}), 非交易日 ***")
        else:
            print(f"\n  *** 今日({today})数据未更新, 最近交易日为{latest_trade_day}({days_gap}天前) ***")
        print(f"  以下为最近交易日({latest_trade_day})的信号, 仅供参考:")

    # === 防重复执行 ===
    pos = load_positions()
    last_run = pos.get('last_run_date', '')
    is_rerun = (str(latest_trade_day) == last_run)
    if is_rerun:
        print(f"\n  *** 今日({latest_trade_day})已执行过, 请勿重复运行 ***")

    # === 打印当前持仓(无论是否交易日/重复执行) ===
    display_positions(pos, etf_data)

    # 非交易日/重复执行: 继续输出信号和推送(仅供参看)
    # 只跳过一次信号计算, 推送照常

    # ---- 评分 ----
    print("\n[2] 计算评分...")
    scores = {}
    details = {}
    for name in ETF_NAMES:
        s, d = score_etf(today_rows[name], name)
        scores[name] = s
        details[name] = d

    # 评分结果
    for name in ETF_NAMES:
        d = details[name]
        print(f"  {name:<8} 评分: {scores[name]:.1f}  {'[放]' if d['amplifier']>1.0 else ''}")

    # ---- 信号解读 ----
    avg_s = np.mean(list(scores.values()))

    # === 牛熊指标 ===
    # 沪深300 MACD状态(代表大盘), 不可用时用第一个可用ETF
    benchmark = today_rows['沪深300'] if '沪深300' in today_rows else today_rows[ref_etf]
    dif300, dea300 = benchmark['macd_dif'], benchmark['macd_dea']
    close300, maL300 = benchmark['close'], benchmark['maL']

    macd_bull = (not np.isnan(dif300) and dif300 > 0 and dif300 > dea300)
    ma_bull   = (not np.isnan(maL300) and close300 > maL300)

    if macd_bull and ma_bull:
        regime = '🟢 牛市'
    elif macd_bull or ma_bull:
        regime = '🟡 震荡'
    else:
        regime = '🔴 熊市'

    # 策略内部档位
    if avg_s >= 6.5:   tier = '强牛 (满仓进攻)'
    elif avg_s >= 5.5: tier = '弱牛 (适度加仓)'
    elif avg_s >= 3.0: tier = '震荡 (平衡配置)'
    else:              tier = '弱势 (防御为主)'

    print(f"\n  {''}")
    print(f"  牛熊指标")
    print(f"  {''}")
    benchmark_name = '沪深300' if '沪深300' in today_rows else ref_etf
    print(f"  市场状态:  {regime}  ({benchmark_name}: MACD{'金叉' if macd_bull else '非金叉'} | {'站上' if ma_bull else '跌破'}年线)")
    print(f"  策略档位:  {tier}")
    print(f"  三ETF均分: {avg_s:.2f} (强牛≥6.5 / 弱牛≥5.5 / 震荡≥3.0)")
    print(f"  仓位上限:  {'14层' if avg_s>=6.0 else '12层' if avg_s>=5.0 else '10层'}")

    # === 顶部预警 ===
    warnings_list = []
    # 1. DIF斜率转负
    turning = []
    for name in ETF_NAMES:
        slp = today_rows[name]['macd_dif_slope']
        if not np.isnan(slp) and slp < 0:
            turning.append(name)
    if len(turning) >= 2:
        names_str = ' '.join(turning)
        warnings_list.append('⚠ DIF斜率转负: ' + names_str + ' → 动能衰减')
    # 2. 5日动量转负
    mom5_neg = []
    for name in ETF_NAMES:
        m5 = today_rows[name]['mom5']
        if not np.isnan(m5) and m5 < 0:
            mom5_neg.append(name)
    if len(mom5_neg) >= 2:
        names_str = ' '.join(mom5_neg)
        warnings_list.append('⚠ 5日动量转负: ' + names_str + ' → 短期走弱')
    # 3. 价格跌破MA25
    below_ma25 = []
    for name in ETF_NAMES:
        c = today_rows[name]['close']; m = today_rows[name]['maS']
        if not np.isnan(m) and c < m:
            below_ma25.append(name)
    if below_ma25:
        names_str = ' '.join(below_ma25)
        warnings_list.append('⚠ 跌破MA25: ' + names_str + ' → 短期破位')
    # 4. MACD背离
    diverge = []
    for name in ETF_NAMES:
        std_bull = today_rows[name]['std_dif'] > today_rows[name]['std_dea']
        our_bull = today_rows[name]['macd_dif'] > today_rows[name]['macd_dea']
        if our_bull and not std_bull:
            diverge.append(name)
    if diverge:
        names_str = ' '.join(diverge)
        warnings_list.append('⚠ MACD背离(策略金叉/标准死叉): ' + names_str + ' → 信号不一致')

    if warnings_list:
        p()
        print("  🔔 顶部预警")
        p()
        for w in warnings_list:
            print("  " + w)
    else:
        print("\n  ✅ 无顶部预警信号")

    # ---- 仓位计算 ----
    print("\n[3] 目标仓位...")
    cash = pos['cash']
    shares = pos['shares']
    cost_basis = pos['cost_basis']
    total_invested = pos['total_invested']

    # 当前持仓市值
    holdings_val = sum(shares[n] * today_rows[n]['close'] for n in shares)
    nav = cash + holdings_val
    peak = pos.get('peak_nav', INITIAL_CAPITAL)
    if nav > peak: peak = nav
    dd = (nav - peak) / peak if peak > 0 else 0
    layer_sz = ((total_invested + nav) / 2) / TOTAL_LAYERS

    # 利润乘数
    profit_ratio = nav / total_invested - 1
    profit_mult = calc_profit_multiplier(nav, total_invested)
    layer_sz *= profit_mult

    target = allocate(scores, dd)

    # 当前层数
    cur = {}
    for n in shares:
        price = today_rows[n]['close']
        cur[n] = round(shares[n] * price / layer_sz)

    # ---- 输出建议 ----
    print(f"\n  {''}")
    print(f"  当日交易建议 (数据日: {latest_trade_day})")
    print(f"  {''}")
    print(f"  组合净值:     {nav:,.0f} 元")
    print(f"  累计投入:     {total_invested:,.0f} 元")
    print(f"  累计收益:     {profit_ratio*100:+.2f}%")
    print(f"  当前回撤:     {dd*100:+.2f}%")
    print(f"  利润乘数:     {profit_mult:.2f}x (每层{layer_sz:,.0f}元)")
    print(f"  当前持仓:     {sum(cur.values())}层 / 市值{holdings_val:,.0f}元 / 占比{holdings_val/nav*100:.1f}%")
    p()

    # 按评分排序: 先卖(释放现金), 再买(评分高的优先, 现金用完即停)
    ranked = sorted(ETF_NAMES, key=lambda n: scores[n], reverse=True)
    remaining_cash = cash
    new_cur = dict(cur)  # 模拟交易后的仓位

    # Step 1: 卖出 (真牛市少卖, 回撤加速)
    above_ma200 = (today_rows['沪深300']['close'] > today_rows['沪深300']['maL'])
    in_bull = (avg_s >= 6.0 and dd > -0.05 and above_ma200)
    if dd <= -0.25:      max_daily_sell = 999
    elif dd <= -0.18:    max_daily_sell = 5
    elif in_bull:        max_daily_sell = 1
    else:                max_daily_sell = 2

    for n in ETF_NAMES:
        diff = target[n] - cur[n]
        if diff >= 0: continue
        actual = max(-max_daily_sell, diff)
        remaining_cash += abs(actual) * layer_sz
        new_cur[n] += actual

    # Step 2: 买入 (只买评分最高的1个, ±2层/天)
    for n in ranked:
        diff = target[n] - new_cur[n]
        if diff <= 0: continue
        desired = min(2, diff)
        # 每层实际成本含手续费, 避免负现金
        cost_per_layer = layer_sz * (1 + COMMISSION + SLIPPAGE)
        affordable = int(remaining_cash / cost_per_layer)
        actual = min(desired, affordable)
        if actual <= 0: continue
        remaining_cash -= actual * layer_sz
        new_cur[n] += actual
        break  # 只买1个标的

    # 输出 & 收集交易
    trades = []
    for n in ETF_NAMES:
        price = today_rows[n]['close']
        diff = new_cur[n] - cur[n]
        status = 'HOLD'
        if diff >= 1: status = 'BUY'
        elif diff <= -1: status = 'SELL'

        cost_val = shares[n] * cost_basis[n] if cost_basis[n] > 0 else 0
        mkt_val = shares[n] * price
        pnl = (mkt_val - cost_val) / cost_val * 100 if cost_val > 0 else 0

        remain = f"(目标{target[n]}层)" if new_cur[n] < target[n] else ""
        print(f"  {n:<8} | 评分{scores[n]:.1f} | {cur[n]}→{new_cur[n]}层 | {status:>4s} {abs(diff)}层 {abs(diff)*layer_sz:,.0f}元 {remain} | 浮动{pnl:+.1f}%")
        if status != 'HOLD':
            trades.append({'品种': n, '方向': status, '层数': abs(diff), '金额': abs(diff)*layer_sz, '价格': price})

    if trades:
        print(f"\n  >>> 执行交易 <<<")
        total_trade = sum(t['金额'] for t in trades if t['方向']=='BUY')
        print(f"  共{len(trades)}笔, 需资金约{total_trade:,.0f}元")
        for t in trades:
            print(f"    {t['方向']} {t['品种']} {t['层数']}层 @{t['价格']:.3f} ≈{t['金额']:,.0f}元")
    else:
        print(f"\n  >>> 无需调仓 <<<")

    # ---- 保存状态 ----
    pos['peak_nav'] = peak
    pos['last_run_date'] = str(latest_trade_day)
    save_positions(pos)

    # ---- Bark推送 ----
    try:
        bark_url = 'https://api.day.app/eoq8G58fJtDDFxHjhNueGH'
        # 构建富文本推送
        lines = []
        lines.append(f"{regime} {tier}")
        lines.append(f"净值{nav:,.0f}  收益{profit_ratio*100:+.1f}%  回撤{dd*100:+.1f}%  仓位{holdings_val/nav*100:.1f}%")
        lines.append(f"现金{cash:,.0f}  乘数{profit_mult:.2f}x  每层{layer_sz:,.0f}元")
        lines.append('')
        lines.append(f"沪深300 {scores['沪深300']:.1f}分 → {cur.get('沪深300',0)}→{new_cur.get('沪深300',0)}层")
        lines.append(f"创业板  {scores['创业板']:.1f}分 → {cur.get('创业板',0)}→{new_cur.get('创业板',0)}层")
        lines.append(f"科创50 {scores['科创50']:.1f}分 → {cur.get('科创50',0)}→{new_cur.get('科创50',0)}层")
        if warnings_list:
            lines.append('')
            lines.append(f"⚠ 预警: {len(warnings_list)}项")
            for w in warnings_list[:2]:  # 最多2条
                lines.append(f"  {w.replace('⚠ ','')[:40]}")
        if trades:
            lines.append('')
            for t in trades:
                lines.append(f">>> {t['方向']} {t['品种']} {t['层数']}层 @{t['价格']:.3f}")
        body = '\n'.join(lines)
        if len(body) > 3000:
            body = body[:3000] + '\n...'
        # 保存图表到本地+GitHub
        prices = {n: today_rows[n]['close'] for n in ETF_NAMES}
        _make_chart_and_save(nav, shares, prices, cash, latest_trade_day)
        chart_url = f'https://cdn.jsdelivr.net/gh/sunran1996/my_candle@main/YH-beta/summary_chart.png?t={latest_trade_day}'
        payload = {'title': f'YH-beta {latest_trade_day}', 'body': body,
                   'url': chart_url}
        data = json.dumps(payload).encode('utf-8')
        urllib.request.urlopen(urllib.request.Request(bark_url, data=data,
            headers={'Content-Type': 'application/json'}), timeout=10)
        print(f"  已推送到手机")
    except Exception:
        pass  # 推送失败不影响主流程

    # ---- 持仓图表 ----
    try:
        # 记录每日净值到历史
        nav_history = pos.get('nav_history', [])
        nav_history.append({'date': str(latest_trade_day), 'nav': round(nav, 0),
                            'ret': round(profit_ratio*100, 2)})
        # 只保留最近90天
        if len(nav_history) > 90:
            nav_history = nav_history[-90:]
        pos['nav_history'] = nav_history
        save_positions(pos)

        # 画图: 净值曲线 + 持仓分布
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'YH-beta 运行概览 ({latest_trade_day})', fontsize=13, fontweight='bold')

        # 左: 净值曲线
        ax = axes[0]
        if len(nav_history) >= 2:
            dates = [e['date'] for e in nav_history]
            nvs = [e['nav']/INITIAL_CAPITAL for e in nav_history]
            ax.plot(range(len(nvs)), nvs, color='#e74c3c', lw=2)
            ax.fill_between(range(len(nvs)), 1.0, nvs, alpha=0.1, color='#e74c3c')
            ax.axhline(1.0, color='gray', lw=0.5, ls='--')
            # X轴标签(只显示首尾和每月初)
            ticks = [0, len(nvs)-1]
            labels = [dates[0], dates[-1]]
            ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=30, fontsize=8)
        ax.set_title(f'净值走势 (累计{profit_ratio*100:+.1f}%)', fontsize=11)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
        ax.grid(True, alpha=0.3)

        # 右: 持仓饼图
        ax = axes[1]
        labels, sizes, colors = [], [], []
        etf_colors = {'沪深300': '#e74c3c', '创业板': '#3498db', '科创50': '#f39c12'}
        for name in ETF_NAMES:
            val = shares[name] * today_rows[name]['close']
            if val > 0:
                labels.append(f'{name}\n{val:,.0f}元')
                sizes.append(val)
                colors.append(etf_colors.get(name, '#95a5a6'))
        if cash > 0:
            labels.append(f'现金\n{cash:,.0f}元')
            sizes.append(cash)
            colors.append('#bdc3c7')
        if sizes:
            wedges, texts = ax.pie(sizes, labels=labels, colors=colors,
                                   startangle=90, textprops={'fontsize': 8})
            for w in wedges:
                w.set_edgecolor('white')
        ax.set_title(f'持仓分布 (总{nav:,.0f}元)', fontsize=11)

        plt.tight_layout()
        chart_path = 'd:\\策略\\YH-beta\\summary_chart.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"\n  持仓图表: {chart_path}")
    except Exception:
        pass  # 画图失败不影响主流程

if __name__ == '__main__':
    main()
