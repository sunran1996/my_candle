# -*- coding: utf-8 -*-
"""查看创业板交易明细"""
import sys,io,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak,pandas as pd,numpy as np
from collections import Counter
warnings.filterwarnings('ignore')

CORE_STOCKS={'山东高速':'sh600350','渝农商行':'sh601077','皖通高速':'sh600012','江苏银行':'sh600919'}
ETF_STOCKS={'创业板':'sz159915'}
ALL_STOCKS={**CORE_STOCKS,**ETF_STOCKS}
INIT=1_000_000;COMM=0.0003;SLIP=0.0001;MAX_POS=0.25
TRAIL_STOP=0.07;HARD_STOP=0.10;COOLDOWN=20
BUY_PARAMS={
    '山东高速':{'rsi':45,'bb':0.25,'tp':0.15,'tp_hi':0.20},
    '渝农商行':{'rsi':30,'bb':0.10,'tp':0.20,'tp_hi':0.25},
    '皖通高速':{'rsi':38,'bb':0.12,'tp':0.20,'tp_hi':0.25},
    '江苏银行':{'rsi':35,'bb':0.10,'tp':0.15,'tp_hi':0.20},
    '创业板':{'tp':0.15,'tp_hi':0.20},
}

def fetch():
    dfs={}
    for n,sym in ALL_STOCKS.items():
        if sym.startswith('sz159')or sym.startswith('sh510'):
            df=ak.fund_etf_hist_sina(symbol=sym)
        else:
            df=ak.stock_zh_a_daily(symbol=sym,adjust='qfq')
        df['date']=pd.to_datetime(df['date'])
        dfs[n]=df[['date','open','high','low','close','volume']].sort_values('date').reset_index(drop=True)
    return dfs

def add_indicators(df):
    df=df.copy();c=df['close']
    df['bb_ma']=c.rolling(20).mean();df['bb_std']=c.rolling(20).std()
    df['bb_up']=df['bb_ma']+2*df['bb_std'];df['bb_lo']=df['bb_ma']-2*df['bb_std']
    df['bb_up_d2']=df['bb_up'].diff().diff()
    ema12=c.ewm(span=12,adjust=False).mean();ema26=c.ewm(span=26,adjust=False).mean()
    df['macd_dif']=ema12-ema26;df['macd_dea']=df['macd_dif'].ewm(span=9,adjust=False).mean()
    df['macd_hist']=2*(df['macd_dif']-df['macd_dea'])
    d=c.diff();g=d.clip(lower=0);l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/14,adjust=False).mean()/l.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan))
    return df

def check_buy(row,name):
    if pd.isna(row['bb_lo'])or pd.isna(row['rsi']):return False,0
    rsi=row['rsi'];c=row['close'];lo=row['bb_lo'];up=row['bb_up']
    if up<=lo:return False,0
    dist=(c-lo)/(up-lo)
    bp=BUY_PARAMS.get(name,{'rsi':42,'bb':0.25})
    sc=(1 if rsi<=bp['rsi'] else 0)+(1 if dist<=bp['bb'] else 0)
    if rsi<=30:sc+=1
    return sc>=1,sc

print("获取数据...")
raw=fetch();dfs={n:add_indicators(d)for n,d in raw.items()}
dates=sorted(set.intersection(*[set(d['date'])for d in dfs.values()]))
dates=[d for d in dates if d>=pd.Timestamp('2019-01-01')]

cash=INIT;shares={n:0.0 for n in ALL_STOCKS};entry={n:0.0 for n in ALL_STOCKS};high={n:0.0 for n in ALL_STOCKS}
accel={n:False for n in ALL_STOCKS};cooldown={n:0 for n in ALL_STOCKS}
sold_today={n:False for n in ALL_STOCKS};trades=[]

for date in dates:
    for n in ALL_STOCKS:sold_today[n]=False
    px={n:raw[n][raw[n]['date']==date]['close'].iloc[0]for n in ALL_STOCKS if len(raw[n][raw[n]['date']==date])>0}
    for n in ALL_STOCKS:
        if shares[n]<=0:continue
        cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
        if cp<=0 or len(r)==0:continue
        if cp>high[n]:high[n]=cp
        pnl=cp/entry[n]-1;dd=cp/high[n]-1
        tp_params=BUY_PARAMS[n];tp=tp_params['tp'];tp_hi=tp_params['tp_hi']
        do=False;sell_px=cp;why=''
        if n in ETF_STOCKS:
            if pnl<=-HARD_STOP:do=True;why='hard'
            else:
                dif=r.iloc[0].get('macd_dif');dea=r.iloc[0].get('macd_dea')
                if (not pd.isna(dif)) and (not pd.isna(dea)) and dif<dea:do=True;why='死叉'
        else:
            if pnl<=-HARD_STOP:do=True;why='hard'
            elif accel[n]:
                if pnl>=tp_hi:do=True;why='accel_tp'
                elif dd<=-TRAIL_STOP:
                    floor=entry[n]*(1+tp);stop_px=max(high[n]*(1-TRAIL_STOP),floor)
                    if cp<=stop_px:do=True;sell_px=max(cp,floor);why='accel_floor'
            elif dd<=-TRAIL_STOP:do=True;why='trail'
            elif pnl>=tp:
                d2=r.iloc[0].get('bb_up_d2')
                if not pd.isna(d2) and d2>0:accel[n]=True
                else:do=True;why='tp'
        if do:
            pnl_real=(sell_px/entry[n]-1)*100
            trades.append({'date':date,'name':n,'dir':'SELL','price':sell_px,'pnl':pnl_real,'why':why,'entry':entry[n]})
            cash+=shares[n]*sell_px*(1-COMM-SLIP)
            shares[n]=0;entry[n]=0;high[n]=0;accel[n]=False;sold_today[n]=True
            if sell_px/entry[n]-1<=-HARD_STOP:cooldown[n]=COOLDOWN
    nav=cash+sum(shares[n]*px.get(n,0)for n in ALL_STOCKS)
    for n in ALL_STOCKS:
        if cooldown[n]>0:cooldown[n]-=1
    for n in CORE_STOCKS:
        if sold_today[n]:continue
        if shares[n]>0:continue
        if cooldown[n]>0:continue
        cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
        if cp<=0 or len(r)==0:continue
        ok,sc=check_buy(r.iloc[0],n)
        if not ok:continue
        val=min(cash,nav*MAX_POS)
        if val>5000:
            qty=val/cp*(1-COMM-SLIP);shares[n]=qty;cash-=val
            entry[n]=cp;high[n]=cp
            trades.append({'date':date,'name':n,'dir':'BUY','price':cp,'pnl':0,'why':f'RSI{dfs[n].iloc[-1]["rsi"]:.0f}'})
    nn='创业板'
    core_held=sum(1 for nn2 in CORE_STOCKS if shares[nn2]>0)
    if core_held<2 and shares[nn]<=0 and cooldown[nn]<=0 and not sold_today.get(nn,False):
        cp=px.get(nn,0);r=dfs[nn][dfs[nn]['date']==date]
        if cp>0 and len(r)>0:
            row=r.iloc[0]
            dif=row.get('macd_dif');dea=row.get('macd_dea');hist=row.get('macd_hist')
            if (not pd.isna(dif))and(not pd.isna(dea))and dif>dea:
                hist_recent=dfs[nn]['macd_hist'].iloc[-40:].dropna()
                max_hist=hist_recent.abs().max()if len(hist_recent)>10 else 0
                strength=abs(hist)/max_hist if max_hist>0 else 0
                pos_frac=0.50 if strength>0.3 else 0.25
                val=min(cash,nav*pos_frac)
                if val>5000:
                    qty=val/cp*(1-COMM-SLIP);shares[nn]=qty;cash-=val
                    entry[nn]=cp;high[nn]=cp
                    trades.append({'date':date,'name':nn,'dir':'BUY','price':cp,'pnl':0,'why':f'MACD{"强" if pos_frac>0.4 else "弱"}'})

# Extract 创业板 trades
cy=[t for t in trades if t['name']=='创业板']
print(f"\n{'='*70}")
print(f"  创业板交易明细 ({len(cy)}笔)")
print(f"{'='*70}")
if cy:
    print(f"{'日期':<12} {'操作':<5} {'价格':>7} {'盈亏':>8} {'持仓天数':>6} {'说明'}")
    print('-'*60)
    last_buy=None
    for t in cy:
        d=t['date'].strftime('%Y-%m-%d')
        pnl_s=f"{t['pnl']:+.1f}%" if t['dir']=='SELL' else '-'
        hd=(t['date']-last_buy['date']).days if t['dir']=='SELL' and last_buy else ''
        print(f"  {d:<12} {t['dir']:<5} {t['price']:>7.3f} {pnl_s:>8} {str(hd):>6}  {t['why']}")
        if t['dir']=='BUY':last_buy=t

sells=[t for t in cy if t['dir']=='SELL']
buys=[t for t in cy if t['dir']=='BUY']
if sells:
    pnls=[t['pnl']for t in sells]
    wr=sum(1 for p in pnls if p>0)/len(pnls)*100
    reasons=Counter(t['why']for t in sells)
    print(f"\n  买入{len(buys)}笔 卖出{len(sells)}笔 胜率{wr:.0f}% 均盈{np.mean(pnls):+.1f}%")
    print(f"  卖出分布: {dict(reasons)}")
    print(f"\n  按年盈亏:")
    for yr in range(2019,2027):
        yt=[t for t in sells if t['date'].year==yr]
        if yt:
            ypnl=sum(t['pnl']for t in yt)
            print(f"    {yr}: {len(yt)}笔 累计{ypnl:+.1f}%")
