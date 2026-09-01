# -*- coding: utf-8 -*-
"""
个股交易策略 YH-1.0: 基于v18 (TS=7% + per-stock TP + 止损冷却20天)
标的: 山东高速 渝农商行 皖通高速 江苏银行
调试版 — 从YH_ultra fork, +创业板(MACD强)补位
"""
import sys, io, os, json, ssl, time, base64, warnings
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

# 核心股票池 (最高优先级)
CORE_STOCKS = {'山东高速': 'sh600350', '渝农商行': 'sh601077', '皖通高速': 'sh600012', '江苏银行': 'sh600919'}
# 创业板ETF (低优先级, 股票持仓<2时启用)
ETF_STOCKS = {'创业板': 'sz159915'}
ALL_STOCKS = {**CORE_STOCKS, **ETF_STOCKS}

INIT = 1_000_000; COMM = 0.0003; SLIP = 0.0001; MAX_POS = 0.25
BARK_ENABLED = True  # 启用Bark(仅推接近信号/买卖信号/交易, 不推纯持仓状态)
BARK_KEYS = ['eoq8G58fJtDDFxHjhNueGH']  # 仅推送给第一个用户
REPO = 'sunran1996/my_candle'

# 每只股票独立买入+止盈阈值 (2019至今最优)
BUY_PARAMS = {
    '山东高速': {'rsi': 45, 'bb': 0.25, 'tp': 0.15, 'tp_hi': 0.20},
    '渝农商行': {'rsi': 30, 'bb': 0.10, 'tp': 0.20, 'tp_hi': 0.25},
    '皖通高速': {'rsi': 38, 'bb': 0.12, 'tp': 0.20, 'tp_hi': 0.25},
    '江苏银行': {'rsi': 35, 'bb': 0.10, 'tp': 0.15, 'tp_hi': 0.20},
    '创业板':  {'tp': 0.10, 'tp_hi': 0.15},  # MACD驱动, 无RSI/BB
}

# 卖出 (默认值)
TRAIL_STOP     = 0.07       # 移动止损7% (最优)
HARD_STOP      = 0.10       # 硬止损10%
COOLDOWN       = 20         # 硬止损后冷却天数
MAX_POS_BOOST  = 0.35       # 连亏≥2 + 有其他持仓 → 加仓35%
MAX_POS_DOUBLE = 0.50       # 连亏≥4 + 有其他持仓 → 翻倍50%
LOSS_STREAK_N  = 2          # 连续止损N次触发加仓
MONTHLY_INJECT = 20000      # 每月定投2w

SCRIPT = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT, '_positions.json')

def load_state():
    """加载持久化持仓状态, 无文件时返回None"""
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            s = json.load(f)
        s['last_date'] = pd.Timestamp(s['last_date'])
        for t in s.get('trades', []):
            t['date'] = pd.Timestamp(t['date'])
        return s
    except Exception as e:
        print(f'  加载持仓状态失败: {e}, 将重新回测')
        return None

def save_state(last_date, cash, total_injected, last_inject_month,
               shares, entry, high, accel, cooldown, loss_streak, all_trades):
    """保存持仓状态到本地JSON"""
    out = {
        'last_date': last_date.strftime('%Y-%m-%d'),
        'cash': cash,
        'total_injected': total_injected,
        'last_inject_month': last_inject_month,
        'positions': {},
        'trades': [],
    }
    for n in ALL_STOCKS:
        out['positions'][n] = {
            'shares': shares[n],
            'entry': entry[n],
            'high': high[n],
            'accel': accel[n],
            'cooldown': cooldown[n],
            'loss_streak': loss_streak[n],
        }
    # 只保留最近50笔交易
    for t in all_trades[-50:]:
        out['trades'].append({
            'date': t['date'].strftime('%Y-%m-%d'),
            'name': t['name'],
            'dir': t['dir'],
            'price': t['price'],
            'pnl': t['pnl'],
            'why': t['why'],
        })
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def push_state(token):
    """推送持仓状态到GitHub"""
    if not os.path.exists(STATE_FILE):
        return
    with open(STATE_FILE, 'rb') as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode('ascii')
    try:
        github_put(token, 'YH-1.0/_positions.json', b64, 'YH-1.0 position state')
        print(f'  持仓状态已推送')
    except Exception as e:
        print(f'  状态推送失败: {e}')

# =====================================================
def _retry(fn, *args, retries=3, delay=5, **kwargs):
    """网络重试: 失败后延迟递增, 最后一次仍失败才抛出"""
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if i == retries - 1:
                raise
            print(f'  网络重试 {i+1}/{retries} ({e.__class__.__name__})...')
            time.sleep(delay * (i + 1))

def fetch():
    dfs = {}
    for name, sym in ALL_STOCKS.items():
        if sym.startswith('sz159') or sym.startswith('sh510'):
            df = _retry(ak.fund_etf_hist_sina, symbol=sym)
            df = df.rename(columns={'prevclose':'pre_close'})
        else:
            df = _retry(ak.stock_zh_a_daily, symbol=sym, adjust='qfq')
        df['date'] = pd.to_datetime(df['date'])
        dfs[name] = df[['date','open','high','low','close','volume']].sort_values('date').reset_index(drop=True)
    return dfs

def add_indicators(df):
    df = df.copy(); c = df['close']
    df['ma20'] = c.rolling(20).mean(); df['ma60'] = c.rolling(60).mean()
    df['bb_ma'] = c.rolling(20).mean(); df['bb_std'] = c.rolling(20).std()
    df['bb_up'] = df['bb_ma'] + 2*df['bb_std']; df['bb_lo'] = df['bb_ma'] - 2*df['bb_std']
    df['bb_up_d2'] = df['bb_up'].diff().diff()  # 上轨二阶导: >0加速扩张(趋势延续)
    df['bb_lo_d2'] = df['bb_lo'].diff().diff()  # 下轨二阶导: <0加速下行(跌势加剧)
    # MACD
    ema12=c.ewm(span=12,adjust=False).mean();ema26=c.ewm(span=26,adjust=False).mean()
    df['macd_dif']=ema12-ema26;df['macd_dea']=df['macd_dif'].ewm(span=9,adjust=False).mean()
    df['macd_hist']=2*(df['macd_dif']-df['macd_dea'])
    d = c.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    df['rsi'] = 100 - 100/(1 + g.ewm(alpha=1/14,adjust=False).mean() /
                l.ewm(alpha=1/14,adjust=False).mean().replace(0, np.nan))
    return df

def check_buy(row, name):
    """每只股票独立阈值, score>=1; 无RSI/BB参数返回False(MACD驱动等)"""
    bp = BUY_PARAMS.get(name, {'rsi': 42, 'bb': 0.25})
    if 'rsi' not in bp or 'bb' not in bp: return False, 0
    if pd.isna(row['bb_lo']) or pd.isna(row['rsi']): return False, 0
    rsi = row['rsi']; c = row['close']; lo = row['bb_lo']; up = row['bb_up']
    if up <= lo: return False, 0
    dist = (c - lo) / (up - lo)
    rsi_th = bp['rsi']; bb_th = bp['bb']
    sc = (1 if rsi <= rsi_th else 0) + (1 if dist <= bb_th else 0)
    if rsi <= 30: sc += 1
    return sc >= 1, sc

NEAR_PCT = 0.03  # 接近买卖点提示阈值(3%)

def proximity_alert(row, name, holding, entry_px, high_px, accel_flag, cooldown_days):
    """接近买/卖点提示, 返回 (短标签, 详情) 或 None"""
    close = row['close']; rsi = row['rsi']
    bp = BUY_PARAMS.get(name, {})
    bb_up = row.get('bb_up'); bb_lo = row.get('bb_lo')

    if not holding:
        if cooldown_days > 0:
            return None
        if 'rsi' not in bp or 'bb' not in bp:
            return None  # 创业板等MACD驱动, 无RSI/BB
        if pd.isna(bb_lo) or pd.isna(bb_up) or bb_up <= bb_lo:
            return None
        buy_px = bb_lo + bp['bb'] * (bb_up - bb_lo)  # BB触发价
        rsi_th = bp['rsi']
        if buy_px > 0 and close <= buy_px * (1 + NEAR_PCT):
            return (f'近买{name}', f'接近买点 {name} 现{close:.2f} 建议≤{buy_px:.2f} RSI{rsi:.0f}(阈{rsi_th})')
        if pd.notna(rsi) and rsi <= rsi_th + 3:
            return (f'近买{name}', f'RSI临近 {name} RSI{rsi:.0f}(阈{rsi_th}) 现{close:.2f}')
        return None
    else:
        tp = bp.get('tp', 0.15); tp_hi = bp.get('tp_hi', 0.20)
        if accel_flag:
            floor = entry_px * (1 + tp)
            stop_px = max(high_px * (1 - TRAIL_STOP), floor)
            target_px = entry_px * (1 + tp_hi)
            stop_label = '保底'
        else:
            stop_px = high_px * (1 - TRAIL_STOP)
            target_px = entry_px * (1 + tp)
            stop_label = '移动止损'
        if stop_px > 0 and stop_px < close <= stop_px * (1 + NEAR_PCT):
            return (f'近卖{name}', f'接近{stop_label} {name} 现{close:.2f} {stop_label}≈{stop_px:.2f} 成本{entry_px:.2f}')
        if target_px > 0 and target_px * (1 - NEAR_PCT) <= close < target_px:
            return (f'近卖{name}', f'接近止盈 {name} 现{close:.2f} 止盈≈{target_px:.2f} 成本{entry_px:.2f}')
        return None

ALERT_FILE = os.path.join(SCRIPT, '_alerts_sent.json')
ALERT_COOLDOWN_MIN = 60  # 同一标的提醒冷却(分钟), 避免盘中每10分钟重复推送

def _load_sent_alerts():
    if not os.path.exists(ALERT_FILE): return {}
    try:
        with open(ALERT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except: return {}

def _save_sent_alerts(d):
    try:
        with open(ALERT_FILE, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False)
    except: pass

# =====================================================
def run_backtest(start_str=None):
    start = pd.Timestamp(start_str) if start_str else None
    print("获取数据...")
    raw = fetch(); dfs = {n: add_indicators(d) for n, d in raw.items()}
    dates = sorted(set.intersection(*[set(d['date']) for d in dfs.values()]))
    if start: dates = [d for d in dates if d >= start]
    if len(dates) < 60: print("数据不足"); return

    cash = INIT
    shares = {n: 0.0 for n in ALL_STOCKS}
    entry = {n: 0.0 for n in ALL_STOCKS}
    high = {n: 0.0 for n in ALL_STOCKS}
    accel = {n: False for n in ALL_STOCKS}    # BB加速模式(已锁定20%底线)
    cooldown = {n: 0 for n in ALL_STOCKS}     # 止损冷却剩余天数
    loss_streak = {n: 0 for n in ALL_STOCKS}  # 连续止损计数
    sold_today = {n: False for n in ALL_STOCKS}
    last_inject_month = None
    total_injected = INIT
    stock_pnl_dollar = {n: 0.0 for n in ALL_STOCKS}
    navs = []; trades = []

    for date in dates:
        # 每月定投
        if last_inject_month is not None and date.month != last_inject_month:
            cash += MONTHLY_INJECT
            total_injected += MONTHLY_INJECT
        last_inject_month = date.month

        # 新的一天, 重置卖出标记
        for n in ALL_STOCKS: sold_today[n] = False
        px = {n: raw[n][raw[n]['date']==date]['close'].iloc[0]
              for n in ALL_STOCKS if len(raw[n][raw[n]['date']==date])>0}

        # ── 卖出 ──
        for n in ALL_STOCKS:
            if shares[n] <= 0: continue
            cp = px.get(n, 0); r = dfs[n][dfs[n]['date']==date]
            if cp <= 0 or len(r)==0: continue
            if cp > high[n]: high[n] = cp
            pnl = cp / entry[n] - 1; dd = cp / high[n] - 1

            tp_params = BUY_PARAMS[n]
            tp = tp_params['tp']; tp_hi = tp_params['tp_hi']

            do = False; why = ''; sell_px = cp
            if pnl <= -HARD_STOP:
                do = True; why = f'硬止损{pnl*100:+.1f}%'
            elif accel[n]:
                if pnl >= tp_hi:
                    do = True; why = f'BB加速止盈{pnl*100:+.1f}%'
                elif dd <= -TRAIL_STOP:
                    floor_px = entry[n] * (1 + tp)
                    stop_px = max(high[n] * (1 - TRAIL_STOP), floor_px)
                    if cp <= stop_px:
                        do = True
                        sell_px = max(cp, floor_px)
                        why = f'BB加速止盈{(sell_px/entry[n]-1)*100:+.1f}%(保底{tp*100:.0f}%)'
            elif dd <= -TRAIL_STOP:
                do = True; why = f'移动止损{pnl*100:+.1f}%'
            elif pnl >= tp:
                d2 = r.iloc[0].get('bb_up_d2')
                if not pd.isna(d2) and d2 > 0:
                    accel[n] = True  # 加速→目标tp_hi+保底tp
                else:
                    do = True; why = f'止盈{pnl*100:+.1f}%'

            if do:
                stock_pnl_dollar[n] += shares[n] * (sell_px * (1-COMM-SLIP) - entry[n])
                cash += shares[n] * sell_px * (1-COMM-SLIP)
                pnl_real = sell_px / entry[n] - 1
                trades.append({'date':date,'name':n,'dir':'SELL','price':sell_px,
                               'pnl':pnl_real*100,'reason':why})
                shares[n] = 0; entry[n] = 0; high[n] = 0; accel[n] = False
                sold_today[n] = True
                if pnl_real > 0: loss_streak[n] = 0
                else: loss_streak[n] += 1
                if pnl_real <= -HARD_STOP:  # 仅硬止损触发冷却
                    cooldown[n] = COOLDOWN

        nav = cash + sum(shares[n]*px.get(n,0) for n in ALL_STOCKS)

        # 冷却递减
        for n in ALL_STOCKS:
            if cooldown[n] > 0: cooldown[n] -= 1

        # ── 买入 第一阶段: 核心4股 (最高优先级) ──
        for n in CORE_STOCKS:
            if sold_today[n]: continue
            if shares[n] > 0: continue
            if cooldown[n] > 0: continue
            cp = px.get(n, 0); r = dfs[n][dfs[n]['date']==date]
            if cp <= 0 or len(r)==0: continue
            row = r.iloc[0]

            ok, sc = check_buy(row, n)
            if not ok: continue

            # 连续止损+有其他持仓 → 阶梯加仓
            has_other = sum(1 for nn in CORE_STOCKS if nn != n and shares[nn] > 0) > 0
            if has_other:
                if loss_streak[n] >= 4: pos_limit = MAX_POS_DOUBLE
                elif loss_streak[n] >= LOSS_STREAK_N: pos_limit = MAX_POS_BOOST
                else: pos_limit = MAX_POS
            else:
                pos_limit = MAX_POS
            target_val = nav * pos_limit
            if cash < target_val and '创业板' in ETF_STOCKS:
                cy = '创业板'
                cy_px = px.get(cy, 0)
                if shares.get(cy, 0) > 0 and cy_px > 0:
                    cy_val = shares[cy] * cy_px
                    need = target_val - cash
                    if need >= 5000:  # 缺口够大才换仓
                        cy_qty_before = shares[cy]
                        sell_val = min(need, cy_val)
                        sell_qty = sell_val / cy_px
                        sell_cost = sell_val * (COMM + SLIP)
                        shares[cy] -= sell_qty
                        cash += sell_val - sell_cost
                        pnl_real = (cy_px / entry[cy] - 1) * 100
                        tag = '全换仓' if shares[cy] < 1e-8 else f'卖{int(sell_qty/cy_qty_before*100)}%'
                        trades.append({'date':date,'name':cy,'dir':'SELL','price':cy_px,
                                       'pnl':pnl_real,'reason':f'换仓→{n}({tag})'})
                        if shares[cy] < 1e-8:
                            shares[cy] = 0; entry[cy] = 0; high[cy] = 0; accel[cy] = False

            val = min(cash, target_val)
            if val > 5000:
                qty = val/cp*(1-COMM-SLIP)
                shares[n] = qty; cash -= val
                entry[n] = cp; high[n] = cp
                real_pct = val / nav * 100
                label = f'RSI{row["rsi"]:.0f} 评{sc}'
                if pos_limit >= MAX_POS_DOUBLE:
                    label += f' 翻倍(连{loss_streak[n]}亏,实{real_pct:.0f}%)'
                elif pos_limit > MAX_POS:
                    label += f' 加仓(连{loss_streak[n]}亏,实{real_pct:.0f}%)'
                trades.append({'date':date,'name':n,'dir':'BUY','price':cp,
                               'pnl':0,'reason':label})

        # ── 买入 第二阶段: 创业板 (持仓<2时启用, MACD驱动) ──
        core_held = sum(1 for n in CORE_STOCKS if shares[n] > 0)
        n = '创业板'
        if core_held < 4 and shares[n] <= 0 and cooldown[n] <= 0:
            cp = px.get(n, 0); r = dfs[n][dfs[n]['date']==date]
            if cp > 0 and len(r) > 0:
                row = r.iloc[0]
                dif = row.get('macd_dif'); dea = row.get('macd_dea'); hist = row.get('macd_hist')
                ma60 = row.get('ma60')
                if (not pd.isna(dif)) and (not pd.isna(dea)) and dif > dea:
                    # 下跌趋势不交易: 价格在MA60下方跳过
                    if not pd.isna(ma60) and cp < ma60: continue
                    hist_recent = dfs[n]['macd_hist'].iloc[-40:].dropna()
                    if len(hist_recent) > 10:
                        max_hist = hist_recent.abs().max()
                        strength = abs(hist) / max_hist if max_hist > 0 else 0
                    else:
                        strength = 0
                    # 强度过滤: 太弱跳过, 温和重仓(高胜率), 强轻仓(防反转)
                    if strength < 0.15: continue
                    if strength <= 0.5:
                        pos_frac = 0.50; tag = '温和'
                    else:
                        pos_frac = 0.25; tag = '强'
                    # 换仓保护: 核心股即将触发买入时不买
                    near_core = False
                    for cn in CORE_STOCKS:
                        if shares[cn] > 0 or cooldown[cn] > 0: continue
                        cr = dfs[cn][dfs[cn]['date']==date]
                        if len(cr) == 0: continue
                        ok, _ = check_buy(cr.iloc[0], cn)
                        if ok: near_core = True; break
                    if near_core: continue
                    label = f'MACD{tag}(s{strength:.1f})'
                    val = min(cash, nav * pos_frac)
                    if val > 5000:
                        qty = val/cp*(1-COMM-SLIP)
                        shares[n] = qty; cash -= val
                        entry[n] = cp; high[n] = cp
                        trades.append({'date':date,'name':n,'dir':'BUY','price':cp,
                                       'pnl':0,'reason':label})

        nav = cash + sum(shares[n]*px.get(n,0) for n in ALL_STOCKS)
        holding = [n for n in ALL_STOCKS if shares[n]>0]
        navs.append({'date':date,'nav':nav,'hold':','.join(holding) if holding else 'CASH'})

    # ── 统计 ──
    ndf = pd.DataFrame(navs); final = ndf['nav'].iloc[-1]
    ret = (final/INIT-1)*100; ann = ((1+ret/100)**(252/len(ndf))-1)*100
    dr = ndf['nav'].pct_change().dropna(); vol = dr.std()*np.sqrt(252)*100
    sr = (ann-2)/vol if vol>0 else 0
    mdd = ((ndf['nav']-ndf['nav'].cummax())/ndf['nav'].cummax()).min()*100

    td = pd.DataFrame(trades)
    buys = td[td['dir']=='BUY']; sells = td[td['dir']=='SELL']
    wr = (sells['pnl']>0).sum()/len(sells)*100 if len(sells)>0 else 0
    aw = sells[sells['pnl']>0]['pnl'].mean() if len(sells[sells['pnl']>0])>0 else 0
    al = sells[sells['pnl']<0]['pnl'].mean() if len(sells[sells['pnl']<0])>0 else 0
    cpct = (ndf['hold']=='CASH').sum()/len(ndf)*100

    total_ret_pct = (final/total_injected-1)*100
    print(f"\n{'='*60}")
    print(f"  YH-1.0: TS={TRAIL_STOP*100:.0f}%+止损冷却{COOLDOWN}天 | 月投{MONTHLY_INJECT/10000:.0f}w")
    print(f"  {'─'*40}")
    print(f"  累计: {ret:+.1f}%  总投{total_injected/10000:.1f}w  净回报{total_ret_pct:+.1f}%  夏普: {sr:.2f}  回撤: {mdd:+.1f}%")
    print(f"  交易: BUY{len(buys)} SELL{len(sells)}  胜率{wr:.0f}%  均盈{aw:+.1f}%  均亏{al:+.1f}%  空仓{cpct:.0f}%")

    # ── 每只股票独立收益 ──
    print(f"\n  {'标的':<8} {'交易':>5} {'胜率':>6} {'均盈':>7} {'均亏':>7} {'已实现盈亏':>12}")
    print(f"  {'─'*55}")
    for name in ALL_STOCKS:
        ss = sells[sells['name']==name]
        if len(ss)==0: continue
        sw = (ss['pnl']>0).sum()
        sr_wr = sw/len(ss)*100
        sr_aw = ss[ss['pnl']>0]['pnl'].mean() if sw>0 else 0
        sr_al = ss[ss['pnl']<0]['pnl'].mean() if sw<len(ss) else 0
        pnl_w = stock_pnl_dollar[name] / 10000
        print(f"  {name:<8} {len(ss):>4}笔 {sr_wr:>5.0f}% {sr_aw:>+6.1f}% {sr_al:>+6.1f}% {pnl_w:>+9.1f}w")
    print(f"  {'─'*55}")
    print(f"  总投入: {total_injected/10000:.0f}w  终值: {final/10000:.0f}w  净回报: {(final/total_injected-1)*100:+.1f}%")

    ndf['year'] = ndf['date'].dt.year
    print(f"\n  {'年份':<6} {'组合':>8} {'回撤':>7}  {'山东高速':>8} {'渝农商行':>8} {'皖通高速':>8} {'江苏银行':>8} {'创业板':>8}")
    for yr, grp in ndf.groupby('year'):
        if len(grp)<10: continue
        yr_ret = (grp['nav'].iloc[-1]/grp['nav'].iloc[0]-1)*100
        yr_mdd = ((grp['nav']-grp['nav'].cummax())/grp['nav'].cummax()).min()*100
        yr_sells = td[(td['dir']=='SELL') & (td['date'].dt.year==yr)]
        parts = []
        for name in ALL_STOCKS:
            ss = yr_sells[yr_sells['name']==name]
            if len(ss) > 0:
                w = ss['pnl'].sum(); n = len(ss)
                parts.append(f'{n}笔 {w:+.1f}%')
            else:
                parts.append('—')
        print(f"  {yr:<6} {yr_ret:>+7.1f}% {yr_mdd:>+6.1f}%  {parts[0]:>8}  {parts[1]:>8}  {parts[2]:>8}  {parts[3]:>8}  {parts[4]:>8}")

    # ── 图表 ──
    RED = '#CC0000'; GREEN = '#008800'; PURPLE = '#9B59B6'; BLUE = '#3498DB'
    ORANGE = '#E67E22'; GRAY = '#888888'; CYAN = '#2ECC71'; DBLUE = '#2980B9'
    colors5 = [RED, ORANGE, CYAN, DBLUE, PURPLE]
    days = (ndf['date'].iloc[-1] - ndf['date'].iloc[0]).days
    plot_start = ndf['date'].iloc[0] if days <= 365 else ndf['date'].iloc[-1] - pd.DateOffset(years=3)

    fig = plt.figure(figsize=(20, 25), facecolor='white')
    gs = fig.add_gridspec(7, 1, height_ratios=[1.2, 2.2, 2.2, 2.2, 2.2, 2.2, 1.0],
                          hspace=0.22, top=0.97, bottom=0.03, left=0.05, right=0.97)

    nav_s = ndf['nav']/INIT; dd_s = (nav_s-nav_s.cummax())/nav_s.cummax()

    # P1: 净值+回撤
    ax = fig.add_subplot(gs[0])
    ax.plot(ndf['date'], nav_s, color=RED, lw=2.0, label='策略净值', zorder=3)
    ax.fill_between(ndf['date'], 1, nav_s, alpha=0.06, color=RED)
    ax.axhline(y=1.0, color=GRAY, lw=0.8, ls='--')
    in_dd, dd_srt = False, None
    for i, (d, dv) in enumerate(zip(ndf['date'], dd_s)):
        if dv<-0.05 and not in_dd: dd_srt=d; in_dd=True
        elif dv>-0.02 and in_dd and dd_srt:
            ax.axvspan(dd_srt, d, alpha=0.06, color='red'); in_dd=False; dd_srt=None
    if in_dd and dd_srt: ax.axvspan(dd_srt, ndf['date'].iloc[-1], alpha=0.06, color='red')
    for _, t in td.iterrows():
        m = ndf['date']==t['date']
        if not m.any(): continue
        yv = nav_s[m].iloc[0]
        ci = list(ALL_STOCKS.keys()).index(t['name']) if t['name'] in ALL_STOCKS else 0
        c = colors5[ci % 5]
        if t['dir']=='BUY': ax.scatter(t['date'],yv,color=c,s=50,marker='^',zorder=6,edgecolors='white',lw=1)
        else: ax.scatter(t['date'],yv,color=GREEN,s=50,marker='v',zorder=6,edgecolors='white',lw=1)
    ax.set_ylabel('净值',fontsize=10); ax.legend(fontsize=8,loc='upper left'); ax.grid(True,alpha=0.12)
    ax.set_title(f'YH-1.0 均值回归+创业板补位 | +{ret:.1f}% | 年化{ann:.1f}% | 夏普{sr:.2f} | 回撤{mdd:.1f}% | {len(td)}笔 | 胜率{wr:.0f}%',
                 fontsize=13, fontweight='bold')

    cn_c = mpf.make_marketcolors(up=RED, down=GREEN, edge='inherit', wick='inherit', volume='inherit')
    cn_s = mpf.make_mpf_style(marketcolors=cn_c, gridstyle='',
                               rc={'font.sans-serif':[CN],'axes.unicode_minus':False})

    for idx, (name, color) in enumerate(zip(ALL_STOCKS.keys(), colors5)):
        ax = fig.add_subplot(gs[idx+1])
        ohlc = raw[name][raw[name]['date']>=plot_start].copy()
        if len(ohlc)<20: ohlc = raw[name].tail(500).copy()
        ohlc = ohlc.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
        ohlc = ohlc.set_index('date')[['Open','High','Low','Close','Volume']]
        mpf.plot(ohlc, type='candle', ax=ax, volume=False, style=cn_s)
        bb = dfs[name][dfs[name]['date']>=plot_start]
        if len(bb)<20: bb = dfs[name].tail(500)
        x = range(len(ohlc))
        ax.plot(x, bb['bb_up'].values[-len(ohlc):], color=PURPLE, lw=0.6, ls='--', alpha=0.5)
        ax.plot(x, bb['bb_lo'].values[-len(ohlc):], color=PURPLE, lw=0.6, ls='--', alpha=0.5)
        ax.plot(x, bb['bb_ma'].values[-len(ohlc):], color=GRAY, lw=0.7, ls='--', alpha=0.4)
        ax.plot(x, bb['ma60'].values[-len(ohlc):], color=BLUE, lw=1.0, alpha=0.6, label='MA60')
        ohlc_dates = ohlc.index
        for _, t in td.iterrows():
            if t['name']!=name: continue
            td_d = pd.Timestamp(t['date'])
            for j, od in enumerate(ohlc_dates):
                if pd.Timestamp(od).date()==td_d.date():
                    if t['dir']=='BUY':
                        ax.scatter(j, ohlc['Low'].iloc[j], color='#FF0000', s=120, marker='^',
                                  zorder=10, edgecolors='white', lw=2.0)
                        ax.annotate(f"买\n{t['price']:.2f}", (j, ohlc['Low'].iloc[j]),
                                   textcoords='offset points', xytext=(0,-25),
                                   fontsize=7, color='#CC0000', fontweight='bold', ha='center')
                    else:
                        ax.scatter(j, ohlc['High'].iloc[j], color='#008800', s=120, marker='v',
                                  zorder=10, edgecolors='white', lw=2.0)
                        ax.annotate(f"卖\n{t['pnl']:+.1f}%", (j, ohlc['High'].iloc[j]),
                                   textcoords='offset points', xytext=(0,15),
                                   fontsize=7, color='#008800', fontweight='bold', ha='center')
                    break
        lp = ohlc['Close'].iloc[-1]; lr = dfs[name]['rsi'].iloc[-1]
        if name == '创业板':
            # 创业板用MACD策略, 画DIF/DEA替代BB
            dif_vals = bb['macd_dif'].values[-len(ohlc):]
            dea_vals = bb['macd_dea'].values[-len(ohlc):]
            ax.plot(x, dif_vals, color=ORANGE, lw=1.0, alpha=0.8, label='DIF')
            ax.plot(x, dea_vals, color=BLUE, lw=0.8, alpha=0.7, label='DEA')
            ax.axhline(y=0, color=GRAY, lw=0.5, ls='-', alpha=0.3)
        ax.set_title(f'{name}  {lp:.2f}  {ohlc_dates[-1].strftime("%Y-%m-%d")}  RSI{lr:.0f}',
                    fontsize=12, fontweight='bold', color=color)
        ax.legend(fontsize=7, loc='upper left'); ax.tick_params(labelsize=7); ax.grid(True, alpha=0.1)

    ax = fig.add_subplot(gs[6])
    ax.fill_between(ndf['date'], 0, dd_s*100, color='#E74C3C', alpha=0.35, step='post')
    ax.plot(ndf['date'], dd_s*100, color='#C0392B', lw=0.8)
    ax.axhline(y=-5, color=GRAY, lw=0.5, ls='--', alpha=0.5)
    ax.axhline(y=-10, color=GRAY, lw=0.5, ls='--', alpha=0.5)
    ax.set_ylabel('回撤 %', fontsize=10); ax.set_ylim(None, 2)
    ax.tick_params(labelsize=8); ax.grid(True, alpha=0.12)
    ax.set_title('组合回撤', fontsize=11, fontweight='bold')

    plt.savefig(os.path.join(SCRIPT,'backtest_chart.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\n  图表: {SCRIPT}\\backtest_chart.png')

    # 交易明细
    if len(td)>0:
        print(f"\n  ── 交易明细 ──")
        print(f"  {'日期':<12} {'标的':<8} {'操作':<5} {'价格':>7} {'盈亏':>7}  {'说明'}")
        print(f"  {'─'*60}")
        for _, t in td.iterrows():
            d = t['date'].strftime('%Y-%m-%d')
            pnl_s = f"{t['pnl']:+.1f}%" if t['dir']=='SELL' else '—'
            print(f"  {d:<12} {t['name']:<8} {t['dir']:<5} {t['price']:>7.2f} {pnl_s:>7}  {t['reason']}")

    return ndf, td

def fetch_realtime(all_stocks_dict):
    """获取实时价格: Sina直连 → akshare → 日线收盘价"""
    results = {}
    # 构建symbol→name映射
    code_to_name = {}
    sina_codes = []
    for name, sym in all_stocks_dict.items():
        code = sym[2:]  # 去掉sh/sz前缀
        code_to_name[code] = name
        sina_codes.append(sym)

    # 方法1: Sina JS API (最可靠, 轻量)
    try:
        ctx = ssl._create_unverified_context()
        url = 'http://hq.sinajs.cn/list=' + ','.join(sina_codes)
        req = ur.Request(url, headers={'Referer':'https://finance.sina.com.cn'})
        data = ur.urlopen(req, timeout=8, context=ctx).read().decode('gbk')
        for line in data.strip().split('\n'):
            if not line.strip() or '=' not in line: continue
            parts = line.split('"')
            if len(parts) < 2: continue
            hq = parts[1].split(',')
            if len(hq) < 4 or hq[0] == '': continue
            code = line.split('_str_')[1].split('=')[0] if '_str_' in line else ''
            if not code: continue
            name = code_to_name.get(code[2:], code)
            price = float(hq[3]) if hq[3] else 0  # 当前价=第4字段
            if price > 0:
                results[name] = price
    except Exception:
        pass

    # 方法2: akshare spot_em (备用)
    missing = {n: s for n, s in all_stocks_dict.items() if n not in results}
    if missing:
        try:
            spot = ak.stock_zh_a_spot_em()
            for sym, name in {v: k for k, v in code_to_name.items()}.items():
                if code_to_name.get(sym) in results: continue
                code = sym  # 纯数字代码
                s = spot[spot['代码'] == code]
                if len(s) > 0:
                    price = float(s['最新价'].iloc[0])
                    if price > 0:
                        results[code_to_name[sym]] = price
        except Exception:
            pass

    return results  # {name: price}, 可能不完整


def send_bark(title, body, url=''):
    if not BARK_ENABLED:
        return
    data = json.dumps({'title':title,'body':body,'url':url}).encode()
    for bk in BARK_KEYS:
        try:
            ur.urlopen(ur.Request(f'https://api.day.app/{bk}', data=data,
                       headers={'Content-Type':'application/json'}), timeout=10)
        except: pass

def github_put(token, path, content_b64, msg):
    ctx = ssl._create_unverified_context()
    h = {'Authorization':'Bearer '+token, 'User-Agent':'YH-1.0'}
    api = f'https://api.github.com/repos/{REPO}/contents/{path}'
    try:
        r = json.loads(ur.urlopen(ur.Request(api, headers=h), timeout=10, context=ctx).read())
        sha = r.get('sha')
    except: sha = None
    body = json.dumps({'message':msg,'content':content_b64,'branch':'main',
                       **({'sha':sha} if sha else {})}).encode()
    ur.urlopen(ur.Request(api, data=body, headers={**h, 'Content-Type':'application/json'}, method='PUT'),
               timeout=15, context=ctx)
    return sha is not None  # True=update, False=create

def upload_chart(token, img_bytes):
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    fn = f'chart_{ts}.png'
    try:
        github_put(token, f'YH-1.0/{fn}', base64.b64encode(img_bytes).decode('ascii'), 'YH-1.0 v1.0 chart')
    except Exception as e:
        print(f'  图表上传失败: {e}')
        return ''
    return f'https://cdn.jsdelivr.net/gh/{REPO}@main/YH-1.0/{fn}'

def push_code(token):
    self_path = os.path.abspath(__file__)
    with open(self_path, 'rb') as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode('ascii')
    # push to YH-1.0/
    try:
        existed = github_put(token, 'YH-1.0/daily_signal.py', b64, 'YH-1.0 v1.0 daily update')
        print(f"  代码已推送 YH-1.0 v1.0 ({'更新' if existed else '新建'})")
    except Exception as e:
        print(f'  代码推送失败: {e}')

def _quick_positions(raw, dfs):
    """快速回测返回 (当前持仓, 近期交易)"""
    dates=sorted(set.intersection(*[set(d['date'])for d in dfs.values()]))
    cash=INIT;shares={n:0.0 for n in ALL_STOCKS};entry={n:0.0 for n in ALL_STOCKS};high={n:0.0 for n in ALL_STOCKS}
    accel={n:False for n in ALL_STOCKS};cooldown={n:0 for n in ALL_STOCKS}
    loss_streak={n:0 for n in ALL_STOCKS}
    last_inject_month=None
    all_trades=[]
    for date in dates:
        # 每月定投
        if last_inject_month is not None and date.month!=last_inject_month:
            cash+=MONTHLY_INJECT
        last_inject_month=date.month
        sold_today={n:False for n in ALL_STOCKS}
        px={n:raw[n][raw[n]['date']==date]['close'].iloc[0]for n in ALL_STOCKS if len(raw[n][raw[n]['date']==date])>0}
        for n in ALL_STOCKS:
            if shares[n]<=0:continue
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            if cp>high[n]:high[n]=cp
            pnl=cp/entry[n]-1;dd=cp/high[n]-1
            tp_params = BUY_PARAMS[n]
            tp = tp_params['tp']; tp_hi = tp_params['tp_hi']
            do=False;sell_px=cp;why=''
            if pnl<=-HARD_STOP:do=True;why='hard'
            elif accel[n]:
                if pnl>=tp_hi:do=True;why='accel_tp25'
                elif dd<=-TRAIL_STOP:
                    floor=entry[n]*(1+tp);stop_px=max(high[n]*(1-TRAIL_STOP),floor)
                    if cp<=stop_px:do=True;sell_px=max(cp,floor);why='accel_floor'
            elif dd<=-TRAIL_STOP:do=True;why='trail'
            elif pnl>=tp:
                d2=r.iloc[0].get('bb_up_d2')
                if not pd.isna(d2) and d2>0:accel[n]=True
                else:do=True;why='tp20'
            if do:
                pnl_real=sell_px/entry[n]-1
                all_trades.append({'date':date,'name':n,'dir':'SELL','price':sell_px,'pnl':pnl_real*100,'why':why})
                cash+=shares[n]*sell_px*(1-COMM-SLIP)
                shares[n]=0;entry[n]=0;high[n]=0;accel[n]=False;sold_today[n]=True
                if pnl_real>0:loss_streak[n]=0
                else:loss_streak[n]+=1
                if pnl_real<=-HARD_STOP:cooldown[n]=COOLDOWN
        nav=cash+sum(shares[n]*px.get(n,0)for n in ALL_STOCKS)
        for n in ALL_STOCKS:
            if cooldown[n]>0:cooldown[n]-=1
        # 核心4股买入
        for n in CORE_STOCKS:
            if sold_today[n]:continue
            if shares[n]>0:continue
            if cooldown[n]>0:continue
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            ok,sc=check_buy(r.iloc[0],n)
            if not ok:continue
            has_other=sum(1 for nn in CORE_STOCKS if nn!=n and shares[nn]>0)>0
            if has_other:
                if loss_streak[n]>=4:pos_limit=MAX_POS_DOUBLE
                elif loss_streak[n]>=LOSS_STREAK_N:pos_limit=MAX_POS_BOOST
                else:pos_limit=MAX_POS
            else:pos_limit=MAX_POS
            target_val=nav*pos_limit
            if cash<target_val and '创业板' in ETF_STOCKS:
                cy='创业板';cy_px=px.get(cy,0)
                if shares.get(cy,0)>0 and cy_px>0:
                    need=target_val-cash
                    if need>=5000:
                        cy_val=shares[cy]*cy_px
                        cy_qty_before=shares[cy]
                        sell_val=min(need,cy_val);sell_qty=sell_val/cy_px
                        sell_cost=sell_val*(COMM+SLIP)
                        shares[cy]-=sell_qty;cash+=sell_val-sell_cost
                        pnl_real=(cy_px/entry[cy]-1)*100
                        tag='全换仓' if shares[cy]<1e-8 else f'卖{int(sell_qty/cy_qty_before*100)}%'
                        all_trades.append({'date':date,'name':cy,'dir':'SELL','price':cy_px,'pnl':pnl_real,'why':f'换仓→{n}({tag})'})
                        if shares[cy]<1e-8:shares[cy]=0;entry[cy]=0;high[cy]=0;accel[cy]=False
            val=min(cash,target_val)
            if val>5000:
                qty=val/cp*(1-COMM-SLIP);shares[n]=qty;cash-=val
                entry[n]=cp;high[n]=cp
                real_pct=val/nav*100
                label=f'RSI{r.iloc[0]["rsi"]:.0f} 评{sc}'
                if pos_limit>=MAX_POS_DOUBLE:label+=f' 翻倍(连{loss_streak[n]}亏,实{real_pct:.0f}%)'
                elif pos_limit>MAX_POS:label+=f' 加仓(连{loss_streak[n]}亏,实{real_pct:.0f}%)'
                all_trades.append({'date':date,'name':n,'dir':'BUY','price':cp,'pnl':0,'why':label})
        # 创业板 fallback
        n='创业板'
        core_held=sum(1 for nn in CORE_STOCKS if shares[nn]>0)
        if core_held<4 and shares[n]<=0 and cooldown[n]<=0:
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp>0 and len(r)>0:
                row=r.iloc[0]
                dif=row.get('macd_dif');dea=row.get('macd_dea');hist=row.get('macd_hist')
                if (not pd.isna(dif)) and (not pd.isna(dea)) and dif>dea:
                    ma60=row.get('ma60')
                    if not pd.isna(ma60) and cp<ma60:continue  # 下跌趋势不交易
                    hist_recent=dfs[n]['macd_hist'].iloc[-40:].dropna()
                    max_hist=hist_recent.abs().max()if len(hist_recent)>10 else 0
                    strength=abs(hist)/max_hist if max_hist>0 else 0
                    # 强度过滤: 太弱跳过, 温和重仓(高胜率), 强轻仓(防反转)
                    if strength<0.15:continue
                    if strength<=0.5:pos_frac=0.50;tag='温和'
                    else:pos_frac=0.25;tag='强'
                    # 换仓保护
                    near_core=False
                    for cn in CORE_STOCKS:
                        if shares[cn]>0 or cooldown[cn]>0:continue
                        cr=dfs[cn][dfs[cn]['date']==date]
                        if len(cr)==0:continue
                        ok,_=check_buy(cr.iloc[0],cn)
                        if ok:near_core=True;break
                    if near_core:continue
                    val=min(cash,nav*pos_frac)
                    if val>5000:
                        qty=val/cp*(1-COMM-SLIP);shares[n]=qty;cash-=val
                        entry[n]=cp;high[n]=cp
                        all_trades.append({'date':date,'name':n,'dir':'BUY','price':cp,'pnl':0,'why':f'MACD{tag}(s{strength:.1f})'})
    recent=[t for t in all_trades if (pd.Timestamp.now()-t['date']).days<365][-20:]
    return {n:entry[n] for n in ALL_STOCKS}, {n:shares[n] for n in ALL_STOCKS}, cash, recent, loss_streak

def live_signal():
    print("获取数据...")
    raw = fetch(); dfs = {n: add_indicators(d) for n, d in raw.items()}

    # ── 增量回测: 从上次状态推进, 避免全量重放 ──
    state = load_state()
    if state is not None:
        # 从持久化状态恢复
        print(f"  加载持仓状态: {state['last_date'].strftime('%Y-%m-%d')}, "
              f"现金{state['cash']/10000:.1f}w, "
              f"持仓{sum(1 for p in state['positions'].values() if p['shares']>0)}只")
        cash = state['cash']
        total_injected = state['total_injected']
        last_inject_month = state['last_inject_month']
        shares = {}; entry = {}; high = {}; accel = {}; cooldown = {}; loss_streak = {}
        for n in ALL_STOCKS:
            ps = state['positions'][n]
            shares[n] = ps['shares']; entry[n] = ps['entry']; high[n] = ps['high']
            accel[n] = ps['accel']; cooldown[n] = ps['cooldown']; loss_streak[n] = ps['loss_streak']
        all_trades = state.get('trades', [])
        # 从last_date的下一天开始
        start_date = state['last_date'] + pd.Timedelta(days=1)
        dates = sorted(set.union(*[set(d['date']) for d in dfs.values()]))
        dates = [d for d in dates if d >= start_date]
    else:
        print("  无持仓状态, 全量回测")
        cash = INIT; total_injected = INIT; last_inject_month = None
        shares = {n: 0.0 for n in ALL_STOCKS}; entry = {n: 0.0 for n in ALL_STOCKS}
        high = {n: 0.0 for n in ALL_STOCKS}; accel = {n: False for n in ALL_STOCKS}
        cooldown = {n: 0 for n in ALL_STOCKS}; loss_streak = {n: 0 for n in ALL_STOCKS}
        all_trades = []
        dates = sorted(set.union(*[set(d['date']) for d in dfs.values()]))

    # ── 推进到最新 ──
    prev_trade_count = len(all_trades)
    for date in dates:
        # 每月定投
        if last_inject_month is not None and date.month != last_inject_month:
            cash += MONTHLY_INJECT; total_injected += MONTHLY_INJECT
        last_inject_month = date.month

        px = {n: raw[n][raw[n]['date']==date]['close'].iloc[0]
              for n in ALL_STOCKS if len(raw[n][raw[n]['date']==date])>0}
        sold_today = {n: False for n in ALL_STOCKS}

        # 卖出
        for n in ALL_STOCKS:
            if shares[n] <= 0: continue
            cp = px.get(n, 0); r = dfs[n][dfs[n]['date']==date]
            if cp <= 0 or len(r)==0: continue
            if cp > high[n]: high[n] = cp
            pnl = cp/entry[n] - 1; dd = cp/high[n] - 1
            tp_params = BUY_PARAMS[n]; tp = tp_params['tp']; tp_hi = tp_params['tp_hi']
            do = False; sell_px = cp; why = ''
            if pnl <= -HARD_STOP: do = True; why = 'hard'
            elif accel[n]:
                if pnl >= tp_hi: do = True; why = 'accel_tp25'
                elif dd <= -TRAIL_STOP:
                    floor = entry[n]*(1+tp); stop_px = max(high[n]*(1-TRAIL_STOP), floor)
                    if cp <= stop_px: do = True; sell_px = max(cp, floor); why = 'accel_floor'
            elif dd <= -TRAIL_STOP: do = True; why = 'trail'
            elif pnl >= tp:
                d2 = r.iloc[0].get('bb_up_d2')
                if not pd.isna(d2) and d2 > 0: accel[n] = True
                else: do = True; why = 'tp20'
            if do:
                pnl_real = sell_px/entry[n] - 1
                all_trades.append({'date':date,'name':n,'dir':'SELL','price':sell_px,'pnl':pnl_real*100,'why':why})
                cash += shares[n]*sell_px*(1-COMM-SLIP)
                shares[n] = 0; entry[n] = 0; high[n] = 0; accel[n] = False; sold_today[n] = True
                if pnl_real > 0: loss_streak[n] = 0
                else: loss_streak[n] += 1
                if pnl_real <= -HARD_STOP: cooldown[n] = COOLDOWN

        nav = cash + sum(shares[n]*px.get(n, 0) for n in ALL_STOCKS)
        for n in ALL_STOCKS:
            if cooldown[n] > 0: cooldown[n] -= 1

        # 核心股买入
        for n in CORE_STOCKS:
            if sold_today[n]: continue
            if shares[n] > 0: continue
            if cooldown[n] > 0: continue
            cp = px.get(n, 0); r = dfs[n][dfs[n]['date']==date]
            if cp <= 0 or len(r)==0: continue
            ok, sc = check_buy(r.iloc[0], n)
            if not ok: continue
            has_other = sum(1 for nn in CORE_STOCKS if nn != n and shares[nn] > 0) > 0
            if has_other:
                if loss_streak[n] >= 4: pos_limit = MAX_POS_DOUBLE
                elif loss_streak[n] >= LOSS_STREAK_N: pos_limit = MAX_POS_BOOST
                else: pos_limit = MAX_POS
            else: pos_limit = MAX_POS
            target_val = nav * pos_limit
            if cash < target_val and '创业板' in ETF_STOCKS:
                cy = '创业板'; cy_px = px.get(cy, 0)
                if shares.get(cy, 0) > 0 and cy_px > 0:
                    need = target_val - cash
                    if need >= 5000:
                        cy_val = shares[cy]*cy_px; cy_qty_before = shares[cy]
                        sell_val = min(need, cy_val); sell_qty = sell_val/cy_px
                        shares[cy] -= sell_qty; cash += sell_val*(1-COMM-SLIP)
                        pnl_real = (cy_px/entry[cy]-1)*100
                        tag = '全换仓' if shares[cy] < 1e-8 else f'卖{int(sell_qty/cy_qty_before*100)}%'
                        all_trades.append({'date':date,'name':cy,'dir':'SELL','price':cy_px,'pnl':pnl_real,'why':f'换仓→{n}({tag})'})
                        if shares[cy] < 1e-8:
                            shares[cy] = 0; entry[cy] = 0; high[cy] = 0; accel[cy] = False
            val = min(cash, target_val)
            if val > 5000:
                qty = val/cp*(1-COMM-SLIP); shares[n] = qty; cash -= val
                entry[n] = cp; high[n] = cp
                real_pct = val/nav*100
                label = f'RSI{r.iloc[0]["rsi"]:.0f} 评{sc}'
                if pos_limit >= MAX_POS_DOUBLE: label += f' 翻倍(连{loss_streak[n]}亏,实{real_pct:.0f}%)'
                elif pos_limit > MAX_POS: label += f' 加仓(连{loss_streak[n]}亏,实{real_pct:.0f}%)'
                all_trades.append({'date':date,'name':n,'dir':'BUY','price':cp,'pnl':0,'why':label})

        # 创业板 fallback
        n = '创业板'
        core_held = sum(1 for nn in CORE_STOCKS if shares[nn] > 0)
        if core_held < 4 and shares[n] <= 0 and cooldown[n] <= 0:
            cp = px.get(n, 0); r = dfs[n][dfs[n]['date']==date]
            if cp > 0 and len(r) > 0:
                row = r.iloc[0]
                dif = row.get('macd_dif'); dea = row.get('macd_dea'); hist = row.get('macd_hist')
                if (not pd.isna(dif)) and (not pd.isna(dea)) and dif > dea:
                    ma60 = row.get('ma60')
                    if not pd.isna(ma60) and cp < ma60: pass
                    else:
                        hist_recent = dfs[n]['macd_hist'].iloc[-40:].dropna()
                        max_hist = hist_recent.abs().max() if len(hist_recent) > 10 else 0
                        strength = abs(hist)/max_hist if max_hist > 0 else 0
                        if strength >= 0.15:
                            if strength <= 0.5: pos_frac = 0.50; tag = '温和'
                            else: pos_frac = 0.25; tag = '强'
                            near_core = False
                            for cn in CORE_STOCKS:
                                if shares[cn] > 0 or cooldown[cn] > 0: continue
                                cr = dfs[cn][dfs[cn]['date']==date]
                                if len(cr) == 0: continue
                                ok, _ = check_buy(cr.iloc[0], cn)
                                if ok: near_core = True; break
                            if near_core: continue
                            val = min(cash, nav*pos_frac)
                            if val > 5000:
                                qty = val/cp*(1-COMM-SLIP); shares[n] = qty; cash -= val
                                entry[n] = cp; high[n] = cp
                                all_trades.append({'date':date,'name':n,'dir':'BUY','price':cp,'pnl':0,'why':f'MACD{tag}(s{strength:.1f})'})

    # ── 保存状态 (持久化) ──
    latest_date = max(dates) if dates else (state['last_date'] if state else raw['山东高速']['date'].iloc[-1])
    save_state(latest_date, cash, total_injected, last_inject_month,
               shares, entry, high, accel, cooldown, loss_streak, all_trades)

    positions = {n: entry[n] for n in ALL_STOCKS}
    holdings = {n: shares[n] for n in ALL_STOCKS}
    cash_end = cash
    recent_trades = [t for t in all_trades if (pd.Timestamp.now()-t['date']).days < 365][-20:]
    new_trades = all_trades[prev_trade_count:]
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    new_trades = [t for t in new_trades if t['date'] == today_str]

    held = [(n, (dfs[n]['close'].iloc[-1]/pos-1)*100) for n, pos in positions.items() if pos > 0]
    if held:
        info = ', '.join(f'{n}({pnl:+.1f}%)' for n, pnl in held)
        print(f"  当前持仓: {info}")
    else:
        print(f"  当前持仓: 全部空仓")

    # 实时行情: Sina直连 → akshare → 日线收盘价
    rt_prices = fetch_realtime(ALL_STOCKS)
    updated = 0
    for name, rt in rt_prices.items():
        if name in raw and rt > 0:
            raw[name].loc[raw[name].index[-1], 'close'] = rt
            raw[name].loc[raw[name].index[-1], 'date'] = pd.Timestamp.now().floor('s')
            updated += 1
    if updated > 0:
        dfs = {n: add_indicators(d) for n, d in raw.items()}
        print(f"  实时行情已更新 ({updated}/{len(ALL_STOCKS)}只)")
    else:
        print("  实时行情失败,用日线收盘价")

    lines = []
    buy_list = []
    alerts = []
    for name in ALL_STOCKS:
        row = dfs[name].iloc[-1]
        close = row['close']; rsi = row['rsi']
        bb_up = row['bb_up']; bb_lo = row['bb_lo']; bb_ma = row['bb_ma']
        bb_range = bb_up - bb_lo
        bb_pos = (close-bb_lo)/(bb_up-bb_lo)*100 if bb_range>0 else 50

        buy_ok, sc = check_buy(row, name)
        holding = positions.get(name, 0) > 0

        # 接近买卖点提示
        pa = proximity_alert(row, name, holding, positions.get(name, 0),
                             high.get(name, 0), accel.get(name, False), cooldown.get(name, 0))
        if pa:
            alerts.append((name, pa[0], pa[1]))

        if buy_ok and not holding:
            sig = '买入'
            buy_list.append((name, sc))
        elif holding:
            sig = '持仓'
        else:
            sig = '空仓'

        # 附加持仓盈亏 + 连续亏损标记
        extra = ''
        if holding:
            pnl_h = (close / positions[name] - 1) * 100
            extra = f' ({pnl_h:+.1f}%)'
        if loss_streak.get(name, 0) >= 4:
            extra += f' ⚡连亏{loss_streak[name]}'
        elif loss_streak.get(name, 0) >= 2:
            extra += f' 连亏{loss_streak[name]}'
        lines.append(f'{sig} | {name} {close:.2f} RSI{rsi:.0f} BB{bb_pos:.0f}%{extra}')

    # 去重: 同标的在冷却时间内只提醒一次, 避免盘中每10分钟重复推送
    sent_map = _load_sent_alerts()
    now_ts = pd.Timestamp.now()
    fresh = []
    for nm, short, detail in alerts:
        last = sent_map.get(nm)
        if last:
            try:
                if (now_ts - pd.Timestamp(last)).total_seconds() < ALERT_COOLDOWN_MIN * 60:
                    continue
            except Exception:
                pass
        fresh.append((nm, short, detail))
        sent_map[nm] = now_ts.strftime('%Y-%m-%d %H:%M:%S')
    if fresh:
        _save_sent_alerts(sent_map)
    alerts = fresh

    print(f"\n{'='*50}")
    print(f"  {' '.join(lines)}")
    print(f"{'='*50}")
    if alerts:
        print("  ⚠️ 接近买卖点:")
        for _n, _s, _d in alerts:
            print(f"    {_d}")

    # 简版K线图 + 买卖点 + 统计面板
    fig, axes = plt.subplots(len(ALL_STOCKS)+1, 1, figsize=(9, 2.9*(len(ALL_STOCKS)+1)), facecolor='white',
        gridspec_kw={'height_ratios':[1]*len(ALL_STOCKS)+[0.8], 'hspace': 0.5})
    cn_c = mpf.make_marketcolors(up='#CC0000', down='#008800', edge='inherit', wick='inherit', volume='inherit')
    cn_s = mpf.make_mpf_style(marketcolors=cn_c, gridstyle='',
                               rc={'font.sans-serif':[CN],'axes.unicode_minus':False})

    # 每只股票K线+买卖点
    from matplotlib.dates import DateFormatter, DayLocator
    for idx, name in enumerate(ALL_STOCKS):
        ax = axes[idx]
        ohlc = raw[name].tail(90).copy()
        ohlc = ohlc.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
        ohlc = ohlc.set_index('date')[['Open','High','Low','Close','Volume']]
        mpf.plot(ohlc, type='candle', ax=ax, volume=False, style=cn_s, datetime_format='%m-%d', xrotation=0)
        bb = dfs[name].tail(90)
        x = range(len(ohlc))
        ax.plot(x, bb['bb_up'].values[-len(ohlc):], color='#9B59B6', lw=0.5, ls='--', alpha=0.5)
        ax.plot(x, bb['bb_lo'].values[-len(ohlc):], color='#9B59B6', lw=0.5, ls='--', alpha=0.5)
        row = dfs[name].iloc[-1]; px = row['close']; rsi = row['rsi']
        chg = (raw[name]['close'].iloc[-1]/raw[name]['close'].iloc[-2]-1)*100 if len(raw[name])>1 else 0

        # 标注近期买卖点 (只标日期, 不标价格/收益)
        ohlc_dates = ohlc.index
        for t in recent_trades:
            if t['name'] != name: continue
            td = pd.Timestamp(t['date'])
            for j, d in enumerate(ohlc_dates):
                if pd.Timestamp(d).date() == td.date():
                    if t['dir'] == 'BUY':
                        ax.scatter(j, ohlc['Low'].iloc[j], color='red', s=80, marker='^',
                                  zorder=10, edgecolors='white', lw=1.5)
                        ax.annotate(td.strftime('%m-%d'),
                                   (j, ohlc['Low'].iloc[j]),
                                   textcoords='offset points', xytext=(0,-18),
                                   fontsize=5.5, color='#CC0000', ha='center')
                    else:
                        ax.scatter(j, ohlc['High'].iloc[j], color='green', s=80, marker='v',
                                  zorder=10, edgecolors='white', lw=1.5)
                        ax.annotate(td.strftime('%m-%d'),
                                   (j, ohlc['High'].iloc[j]),
                                   textcoords='offset points', xytext=(0,10),
                                   fontsize=5.5, color='#008800', ha='center')
                    break

        holding = positions.get(name, 0) > 0
        status = f'持仓 +{(px/positions[name]-1)*100:+.1f}%' if holding else '空仓'
        ax.set_title(f'{name} {px:.2f} {chg:+.2f}% RSI{rsi:.0f} | {status}', fontsize=14, fontweight='bold',
                    color='#CC0000' if holding else '#333333')
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.1)

    # 第五面板: 统计 (加权总浮动盈亏)
    ax5 = axes[-1]
    ax5.axis('off')
    from matplotlib.patches import FancyBboxPatch
    # 先算总资产
    total_mv = 0; total_cost = 0
    for name in ALL_STOCKS:
        cp = dfs[name].iloc[-1]['close']
        if positions.get(name, 0) > 0:
            sh = holdings[name]
            total_mv += sh * cp
            total_cost += sh * positions[name]
    total_asset = total_mv + cash_end

    rows_data = []
    for name in ALL_STOCKS:
        row = dfs[name].iloc[-1]; cp = row['close']
        holding = positions.get(name, 0) > 0
        if holding:
            entry_px = positions[name]; sh = holdings[name]
            pnl_pct = (cp/entry_px-1)*100
            mv = sh * cp
            weight = mv / total_asset * 100 if total_asset > 0 else 0
            stock_trades = [t for t in recent_trades if t['name']==name and t['dir']=='SELL']
            hist_wins = sum(1 for t in stock_trades if t['pnl']>0)
            hist_total = len(stock_trades)
            hist_wr = f'{hist_wins}/{hist_total}' if hist_total>0 else '-'
            rows_data.append([name, f'{cp:.2f}', f'{entry_px:.2f}', f'{pnl_pct:+.1f}%',
                            f'{hist_wr}', f'{weight:.1f}%'])
        else:
            stock_trades = [t for t in recent_trades if t['name']==name and t['dir']=='SELL']
            hist_wins = sum(1 for t in stock_trades if t['pnl']>0)
            hist_total = len(stock_trades)
            hist_wr = f'{hist_wins}/{hist_total}' if hist_total>0 else '-'
            rows_data.append([name, f'{cp:.2f}', '-', '-', f'{hist_wr}', '空仓'])

    total_pnl_pct = (total_mv/total_cost-1)*100 if total_cost>0 else 0

    col_labels = ['标的', '现价', '成本', '盈亏%', '历史胜率', '占比']
    table = ax5.table(cellText=rows_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#DDDDDD')
        if key[0] == 0:
            cell.set_facecolor('#F5F5F5')
            cell.set_text_props(fontweight='bold')
        elif rows_data[key[0]-1][-1] == '空仓':
            pass
        else:
            cell.set_facecolor('#FFF3F3')

    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    stock_pct = total_mv/total_asset*100 if total_asset>0 else 0
    cash_pct = cash_end/total_asset*100 if total_asset>0 else 0
    status_line = f'{today_str} | 浮动盈亏: {total_pnl_pct:+.1f}% | 持仓{sum(1 for p in positions.values() if p>0)}只 | 股票{stock_pct:.0f}%/现金{cash_pct:.0f}%' if total_mv>0 else f'{today_str} | 全部空仓'
    ax5.set_title(status_line, fontsize=11, fontweight='bold', loc='left', pad=10)

    buf = io.BytesIO()
    fig.savefig(buf, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    img_bytes = buf.getvalue()

    # 上传GitHub
    token = os.environ.get('GH_TOKEN','')
    if not token:
        for p in ['../github_token.txt','github_token.txt','d:/策略/github_token.txt']:
            try: token = open(p).read().strip(); break
            except: pass
    chart_url = ''
    if token:
        chart_url = upload_chart(token, img_bytes)
        push_code(token)
        push_state(token)

    # 推送
    # 非交易日检测: 周末=非交易日, 工作日=交易日(不推断假日)
    now = pd.Timestamp.now()
    is_weekend = now.dayofweek >= 5  # 周六=5 周日=6
    is_trading_day = not is_weekend

    buy_names = [b for b,_ in buy_list]
    buy_count = len(buy_names)
    holding_count = sum(1 for p in positions.values() if p > 0)

    # 接近买卖点提示
    near_short = ' '.join(a[1] for a in alerts)
    near_body = '\n'.join(f'  ⚠️ {a[2]}' for a in alerts)

    # 交易变动提醒
    alert_body = ''
    parts = []
    if new_trades:
        for t in new_trades:
            d = t['date'].strftime('%m-%d')
            if t['dir'] == 'BUY':
                parts.append(f"买{t['name']}")
                alert_body += f"🔴 {d} 买入 {t['name']} @{t['price']:.2f} {t['why']}\n"
            else:
                pnl_s = f"{t['pnl']:+.1f}%"
                why_cn = {'hard':'硬止损','trail':'移动止损','tp20':'止盈','accel_tp25':'BB加速止盈',
                          'accel_floor':'BB加速保底'}.get(t['why'], t['why'])
                if '换仓' in t['why']:
                    parts.append(f"换仓{t['name']}")
                    alert_body += f"🔄 {d} {t['why']} {pnl_s}\n"
                else:
                    parts.append(f"卖{t['name']}{pnl_s}")
                    alert_body += f"🟢 {d} 卖出 {t['name']} {pnl_s} ({why_cn})\n"

    # 是否推送: 有接近信号/买入信号/交易变动才推, 纯持仓/空仓状态不推
    actionable = len(alerts) > 0 or buy_count > 0 or len(new_trades) > 0

    if not is_trading_day:
        day_type = '周末' if is_weekend else '假日'
        if alerts:
            title = f'YH1.0 [{day_type}] ⚠️ {near_short}'
        elif buy_count >= 1:
            title = f'YH1.0 [{day_type}] 买入: ' + ' '.join(buy_names)
        elif holding_count > 0:
            title = f'YH1.0 [{day_type}] 持仓中 ({holding_count}只)'
        else:
            title = f'YH1.0 [{day_type}] 空仓 (非交易日)'
    else:
        if alerts:
            title = 'YH1.0 ⚠️ ' + near_short
        elif new_trades:
            title = 'YH1.0 ' + ' '.join(parts)
        elif buy_count >= 3: title = 'YH1.0 多只买入! ' + ' '.join(buy_names)
        elif buy_count >= 1: title = 'YH1.0 买入: ' + ' '.join(buy_names)
        elif holding_count > 0: title = f'YH1.0 持仓中 ({holding_count}只)'
        else: title = 'YH1.0 空仓观望'

    body = '\n'.join(lines)
    if near_body:
        body = '⚠️ 接近买卖点:\n' + near_body + '\n' + body
    if alert_body:
        body = alert_body + '\n' + body
    if not is_trading_day:
        body = f'⚠️ 今日{day_type}, 以下为最近交易日信号:\n' + body

    if actionable:
        send_bark(title, body, chart_url)
        print("已推送")
    else:
        print("无动作信号, 跳过推送")

    with open('_preview.png','wb') as f: f.write(img_bytes)
    print(f"完成! _preview.png")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--from', dest='fr', type=str, default=None)
    p.add_argument('--live', action='store_true', default=False)
    a = p.parse_args()
    if a.live: live_signal()
    else: run_backtest(a.fr or '2016-01-01')

if __name__ == '__main__':
    main()
