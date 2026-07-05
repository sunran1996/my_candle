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
import json, os, warnings
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
POSITION_FILE = 'd:\\策略\\YH02\\positions.json'

# ============================================================
# 0. 共享工具函数
# ============================================================
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

        # 极端波动过滤器: 只在回撤+高波时激活, 正常行情不干预
        vol_r = rows.get('沪深300', {}).get('vol_ma60', np.nan)
        if not np.isnan(vol_r) and vol_r > 0 and dd < -0.05:
            vol20 = rows['沪深300'].get('vol20', np.nan)
            if not np.isnan(vol20):
                vol_ratio = vol20 / vol_r
                if vol_ratio > 2.5:       vol_adj = 0.6   # 股灾级波动
                elif vol_ratio > 2.0:     vol_adj = 0.75  # 剧烈波动
                elif vol_ratio > 1.5:     vol_adj = 0.9   # 显著升高
                else:                     vol_adj = 1.0
                layer_sz *= vol_adj

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

        # 卖出 — YH02全量版: 不限每日卖速
        for n in ETF_NAMES:
            diff = target[n] - new_cur[n]
            if diff >= 0: continue
            actual = diff  # 不限量, 直接卖到目标仓位
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

        # 买入 — YH02全量版: 不限量不限标的
        for n in ranked:
            diff = target[n] - new_cur[n]
            if diff <= 0: continue
            desired = diff  # 不限每日买量
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

        cash = remaining_cash
        nav_log.append({'date': d, 'nav': nav, 'holdings': holdings, 'dd': dd})

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
        csv_path = 'd:\\策略\\YH02\\backtest.csv'
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

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.suptitle(f'ETF轮动策略 回测 ({start_date_str} → {trade_dates[-1].strftime("%Y-%m-%d")})', fontsize=14, fontweight='bold')

        # 1. 净值曲线
        ax = axes[0]
        ax.plot(nav_df['date'], nav_df['nav']/INITIAL_CAPITAL, color='#e74c3c', lw=1.5, label=f'净值 (累计{total_ret:+.1f}%)')
        ax.axhline(1.0, color='gray', lw=0.5, ls='--')
        ax.fill_between(nav_df['date'], 1.0, nav_df['nav']/INITIAL_CAPITAL,
                        where=nav_df['nav']/INITIAL_CAPITAL>=1.0, color='#e74c3c', alpha=0.1)
        ax.fill_between(nav_df['date'], 1.0, nav_df['nav']/INITIAL_CAPITAL,
                        where=nav_df['nav']/INITIAL_CAPITAL<1.0, color='#2ecc71', alpha=0.1)
        ax.set_ylabel('净值', fontsize=10)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)

        # 2. 每日收益率柱状图
        ax = axes[1]
        colors = ['#e74c3c' if r >= 0 else '#2ecc71' for r in nav_df['daily_ret'].iloc[1:]]
        ax.bar(nav_df['date'].iloc[1:], nav_df['daily_ret'].iloc[1:], color=colors, width=1.0, alpha=0.8)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_ylabel('日收益(%)', fontsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:+.1f}%'))
        ax.grid(True, alpha=0.3)

        # 3. 累计收益率
        ax = axes[2]
        ax.fill_between(nav_df['date'], 0, nav_df['cum_ret'], color='#e74c3c', alpha=0.15)
        ax.plot(nav_df['date'], nav_df['cum_ret'], color='#c0392b', lw=1.2)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_ylabel('累计收益(%)', fontsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:+.0f}%'))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = 'd:\\策略\\YH02\\backtest_chart.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"\n  K线图已保存: {chart_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='ETF轮动策略每日监控')
    parser.add_argument('-r', '--reset', action='store_true', help='清空持仓记录, 从今天开始模拟')
    parser.add_argument('--from', dest='from_date', type=str, default=None, metavar='YYYY-MM-DD', help='从指定日期回测到今天')
    parser.add_argument('--dca', type=int, default=0, metavar='N', help='每月追加N元本金(如 --dca 20000)')
    parser.add_argument('--output', type=str, default=None, help='输出到文件(UTF-8)')
    args = parser.parse_args()

    if args.output:
        sys.stdout = open(args.output, 'w', encoding='utf-8')

    # === 回测模式 ===
    if args.from_date:
        run_backtest(args.from_date, dca=args.dca)
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

    if is_rerun:
        return
    # 非交易日继续输出信号(仅供参看), 交易日正常执行

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

    # Step 1: 卖出 — YH02不限速
    for n in ETF_NAMES:
        diff = target[n] - cur[n]
        if diff >= 0: continue
        actual = diff
        remaining_cash += abs(actual) * layer_sz
        new_cur[n] += actual

    # Step 2: 买入 — YH02不限量不限标的
    for n in ranked:
        diff = target[n] - new_cur[n]
        if diff <= 0: continue
        desired = diff
        cost_per_layer = layer_sz * (1 + COMMISSION + SLIPPAGE)
        affordable = int(remaining_cash / cost_per_layer)
        actual = min(desired, affordable)
        if actual <= 0: continue
        remaining_cash -= actual * layer_sz
        new_cur[n] += actual

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

if __name__ == '__main__':
    main()
