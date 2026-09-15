# -*- coding: utf-8 -*-
"""夏普优化变体对比: V0基准 + V1~V7, 2019年起"""
import sys,io,os,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak,pandas as pd,numpy as np
warnings.filterwarnings('ignore')

STOCKS={'山东高速':'sh600350','渝农商行':'sh601077','皖通高速':'sh600012','江苏银行':'sh600919'}
INIT=1_000_000;COMM=0.0003;SLIP=0.0001;MAX_POS=0.25

def fetch():
    dfs={}
    for n,s in STOCKS.items():
        df=ak.stock_zh_a_daily(symbol=s,adjust='qfq');df['date']=pd.to_datetime(df['date'])
        dfs[n]=df[['date','open','high','low','close','volume']].sort_values('date').reset_index(drop=True)
    return dfs

def add_indicators(df):
    df=df.copy();c=df['close']
    df['ma20']=c.rolling(20).mean();df['ma60']=c.rolling(60).mean()
    df['bb_ma']=c.rolling(20).mean();df['bb_std']=c.rolling(20).std()
    df['bb_up']=df['bb_ma']+2*df['bb_std'];df['bb_lo']=df['bb_ma']-2*df['bb_std']
    df['bb_bw']=(df['bb_up']-df['bb_lo'])/df['bb_ma']
    df['vol_ma20']=df['volume'].rolling(20).mean()  # 新增: 成交量均线
    d=c.diff();g=d.clip(lower=0);l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/14,adjust=False).mean()/l.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan))
    return df

def run_one(name, buy_check, sell_check, max_pos, use_scaling=True, time_stop=0):
    """use_scaling=False → 亏损后不缩放补仓; time_stop>0 → 持仓超N天强制平仓"""
    raw=fetch();dfs={n:add_indicators(d)for n,d in raw.items()}
    dates=sorted(set.intersection(*[set(d['date'])for d in dfs.values()]))
    dates=[d for d in dates if d>=pd.Timestamp('2019-01-01')]
    cash=INIT;shares={n:0.0 for n in STOCKS};entry={n:0.0 for n in STOCKS};high={n:0.0 for n in STOCKS}
    loss={n:False for n in STOCKS};scale_step={n:0 for n in STOCKS}
    hold_start={n:None for n in STOCKS}  # 持仓起始日期
    SCALE_PCTS=[0.30,0.30,0.40]
    navs=[];trades=[];pbw={n:None for n in STOCKS}

    for date in dates:
        # 同日卖后禁买重置
        for n in STOCKS:
            if scale_step[n]==-1: scale_step[n]=0
        px={n:raw[n][raw[n]['date']==date]['close'].iloc[0]for n in STOCKS if len(raw[n][raw[n]['date']==date])>0}

        # ── 卖出 ──
        for n in STOCKS:
            if shares[n]<=0:continue
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            if cp>high[n]:high[n]=cp
            pnl=cp/entry[n]-1;dd=cp/high[n]-1
            hold_days = (date-hold_start[n]).days if hold_start[n] else 0

            do=False;why=''
            # 时间止损(优先级最高)
            if time_stop>0 and hold_days>=time_stop:
                do=True;why=f'时间止损{hold_days}天'
            if not do:
                do,why=sell_check(r.iloc[0],pnl,dd,pbw.get(n))
            if do:
                cash+=shares[n]*cp*(1-COMM-SLIP)
                trades.append({'pnl':pnl*100})
                if pnl>=0:
                    scale_step[n]=0;loss[n]=False
                else:
                    if use_scaling:loss[n]=True
                    else:loss[n]=False  # V3: 亏损也不缩放
                    scale_step[n]=0
                shares[n]=0;entry[n]=0;high[n]=0;hold_start[n]=None
                scale_step[n]=-1
            bw=r.iloc[0].get('bb_bw')
            if not pd.isna(bw):pbw[n]=bw

        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)

        # ── 买入 ──
        for n in STOCKS:
            if scale_step[n]==-1:continue
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue

            in_scale=use_scaling and loss[n] and scale_step[n]<3
            if shares[n]>0 and not in_scale:continue

            ok,sc=buy_check(r.iloc[0],loss[n] if use_scaling else False)
            if not ok:continue

            if in_scale:
                step=scale_step[n]+1;pct=SCALE_PCTS[step-1]
                val=min(cash,nav*max_pos*pct)
                if val>5000:
                    qty=val/cp*(1-COMM-SLIP)
                    old_cost=entry[n]*shares[n]if shares[n]>0 else 0
                    shares[n]+=qty;cash-=val
                    entry[n]=(old_cost+cp*qty)/shares[n]if shares[n]>0 else cp
                    high[n]=max(high[n],cp)if shares[n]-qty>0 else cp
                    scale_step[n]=step
            else:
                if shares[n]>0:continue
                val=min(cash,nav*max_pos)
                if val>5000:
                    qty=val/cp*(1-COMM-SLIP)
                    shares[n]=qty;cash-=val
                    entry[n]=cp;high[n]=cp;hold_start[n]=date

        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)
        navs.append(nav)

    ndf=pd.DataFrame(navs,columns=['nav']);final=ndf['nav'].iloc[-1]
    ret=(final/INIT-1)*100;ann=((1+ret/100)**(252/len(ndf))-1)*100
    dr=ndf['nav'].pct_change().dropna();vol=dr.std()*np.sqrt(252)*100
    sr=(ann-2)/vol if vol>0 else 0
    mdd=((ndf['nav']-ndf['nav'].cummax())/ndf['nav'].cummax()).min()*100
    sells=np.array([t['pnl']for t in trades if t['pnl']!=0])
    wr=(sells>0).sum()/len(sells)*100 if len(sells)>0 else 0
    aw=sells[sells>0].mean() if (sells>0).sum()>0 else 0
    al=sells[sells<0].mean() if (sells<0).sum()>0 else 0
    return ann,ret,sr,mdd,wr,len(trades),aw,al

# ============ 买入函数 ============

# V0/V5 基准 — RSI<42 或 BB<25%, score>=1
def buy_baseline(row,strict):
    if pd.isna(row['bb_lo'])or pd.isna(row['rsi']):return False,0
    rsi=row['rsi'];c=row['close'];lo=row['bb_lo'];up=row['bb_up']
    if up<=lo:return False,0
    dist=(c-lo)/(up-lo)
    rsi_th=35 if strict else 42;bb_th=0.18 if strict else 0.25
    sc=(1 if rsi<=rsi_th else 0)+(1 if dist<=bb_th else 0)
    if strict and rsi<=25:sc+=1
    elif not strict and rsi<=30:sc+=1
    return sc>=1,sc

# V1 — 积分>=2 (RSI AND BB 双触发)
def buy_score2(row,strict):
    if pd.isna(row['bb_lo'])or pd.isna(row['rsi']):return False,0
    rsi=row['rsi'];c=row['close'];lo=row['bb_lo'];up=row['bb_up']
    if up<=lo:return False,0
    dist=(c-lo)/(up-lo)
    rsi_th=35 if strict else 42;bb_th=0.18 if strict else 0.25
    sc=(1 if rsi<=rsi_th else 0)+(1 if dist<=bb_th else 0)
    if strict and rsi<=25:sc+=1
    elif not strict and rsi<=30:sc+=1
    return sc>=2,sc

# V2 — 成交量收缩过滤 (volume < vol_ma20)
def buy_vol(row,strict):
    ok,sc=buy_baseline(row,strict)
    if not ok:return False,0
    vol=row.get('volume');vma=row.get('vol_ma20')
    if pd.isna(vol)or pd.isna(vma)or vma<=0:return False,0
    if vol>=vma:return False,0  # 放量不买
    return True,sc

# V5 — MA20趋势过滤 (price < MA20)
def buy_ma20(row,strict):
    ok,sc=buy_baseline(row,strict)
    if not ok:return False,0
    c=row['close'];ma20=row.get('ma20')
    if pd.isna(ma20):return False,0
    if c>=ma20:return False,0  # 价格在均线上方不买
    return True,sc

# V7 — 组合: score>=2 + 缩量, 无严格模式
def buy_combo(row,strict):
    # 忽略strict, 始终用正常模式但score>=2
    if pd.isna(row['bb_lo'])or pd.isna(row['rsi']):return False,0
    rsi=row['rsi'];c=row['close'];lo=row['bb_lo'];up=row['bb_up']
    if up<=lo:return False,0
    dist=(c-lo)/(up-lo)
    sc=(1 if rsi<=42 else 0)+(1 if dist<=0.25 else 0)
    if rsi<=30:sc+=1
    if sc<2:return False,0
    # 成交量过滤
    vol=row.get('volume');vma=row.get('vol_ma20')
    if pd.isna(vol)or pd.isna(vma)or vma<=0:return False,0
    if vol>=vma:return False,0
    return True,sc

# ============ 卖出函数 ============

# V0 基准 — HS-10% TS-8% TP+20%
def sell_baseline(row,pnl,dd,pbw):
    if pnl<=-0.10:return True,'硬止损'
    if dd<=-0.08:return True,'移动止损'
    if pnl>=0.20:return True,'止盈'
    return False,''

# V4 — 阶梯止盈: 浮盈>10%收紧TS到5%, >15%收紧到3%
def sell_tiered(row,pnl,dd,pbw):
    if pnl<=-0.10:return True,'硬止损'
    if pnl>0.15:ts=0.03
    elif pnl>0.10:ts=0.05
    else:ts=0.08
    if dd<=-ts:return True,f'阶梯止损(TS{ts*100:.0f}%)'
    if pnl>=0.20:return True,'止盈'
    return False,''

# V6 — 时间止损 + 基准卖出 (时间止损在run_one中处理)
def sell_time(row,pnl,dd,pbw):
    return sell_baseline(row,pnl,dd,pbw)

# V9 — 硬止损收紧到-8%
def sell_hs8(row,pnl,dd,pbw):
    if pnl<=-0.08:return True,'硬止损-8%'
    if dd<=-0.08:return True,'移动止损'
    if pnl>=0.20:return True,'止盈'
    return False,''

# V10 — 止盈放宽到25%
def sell_tp25(row,pnl,dd,pbw):
    if pnl<=-0.10:return True,'硬止损'
    if dd<=-0.08:return True,'移动止损'
    if pnl>=0.25:return True,'止盈25%'
    return False,''

# ============ 运行 ============
versions=[
    # (名称, buy_fn, sell_fn, use_scaling, time_stop)
    ('V0 基准(当前)',         buy_baseline, sell_baseline, True,  0),
    ('V1 积分>=2',           buy_score2,   sell_baseline, True,  0),
    ('V2 缩量买入',           buy_vol,      sell_baseline, True,  0),
    ('V3 取消缩放补仓',       buy_baseline, sell_baseline, False, 0),
    ('V4 阶梯止盈',           buy_baseline, sell_tiered,   True,  0),
    ('V5 MA20趋势过滤',       buy_ma20,     sell_baseline, True,  0),
    ('V6 时间止损60天',       buy_baseline, sell_time,     True,  60),
    ('V7 组合(>=2+缩量+阶梯)', buy_combo,    sell_tiered,   False, 0),
    ('V8 V3+V6(无缩放+时停)',  buy_baseline, sell_baseline, False, 60),
    ('V9 硬止损收紧-8%',       buy_baseline, sell_hs8,      True,  0),
    ('V10 止盈放宽25%',        buy_baseline, sell_tp25,     True,  0),
    ('V11 V3+止盈25%',        buy_baseline, sell_tp25,     False, 0),
]

print(f"{'版本':<24} {'年化':>7} {'累计':>7} {'夏普':>6} {'回撤':>7} {'胜率':>6} {'交易':>5} {'均盈':>7} {'均亏':>7}")
print('─'*80)
best_sr=0;best_name=''
for nm, buy_fn, sell_fn, us, ts in versions:
    ann,ret,sr,mdd,wr,nt,aw,al=run_one(nm,buy_fn,sell_fn,MAX_POS,use_scaling=us,time_stop=ts)
    print(f'{nm:<24} {ann:>+6.1f}% {ret:>+6.0f}% {sr:>5.2f} {mdd:>+6.1f}% {wr:>5.0f}% {nt:>5} {aw:>+6.1f}% {al:>+6.1f}%')
    if sr>best_sr:best_sr=sr;best_name=nm

print(f'\n最佳夏普: {best_name} (SR={best_sr:.2f})')
