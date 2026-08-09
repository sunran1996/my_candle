# -*- coding: utf-8 -*-
"""分析v13 BB加速保底止盈 (含sell_px修复)"""
import sys,io,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak,pandas as pd,numpy as np
from collections import Counter
warnings.filterwarnings('ignore')

STOCKS={'山东高速':'sh600350','渝农商行':'sh601077','皖通高速':'sh600012','江苏银行':'sh600919'}
INIT=1_000_000;COMM=0.0003;SLIP=0.0001;MAX_POS=0.25
TAKE_PROFIT=0.20;TAKE_PROFIT_HI=0.25;TRAIL_STOP=0.08;HARD_STOP=0.10
BUY_PARAMS={
    '山东高速':{'rsi':45,'bb':0.25},'渝农商行':{'rsi':30,'bb':0.10},
    '皖通高速':{'rsi':38,'bb':0.12},'江苏银行':{'rsi':35,'bb':0.10},
}

def fetch():
    dfs={}
    for n,s in STOCKS.items():
        df=ak.stock_zh_a_daily(symbol=s,adjust='qfq');df['date']=pd.to_datetime(df['date'])
        dfs[n]=df[['date','open','high','low','close','volume']].sort_values('date').reset_index(drop=True)
    return dfs

def add_indicators(df):
    df=df.copy();c=df['close']
    df['ma20']=c.rolling(20).mean()
    df['bb_ma']=c.rolling(20).mean();df['bb_std']=c.rolling(20).std()
    df['bb_up']=df['bb_ma']+2*df['bb_std'];df['bb_lo']=df['bb_ma']-2*df['bb_std']
    df['bb_up_d2']=df['bb_up'].diff().diff()
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

raw=fetch();dfs={n:add_indicators(d)for n,d in raw.items()}
dates=sorted(set.intersection(*[set(d['date'])for d in dfs.values()]))
dates=[d for d in dates if d>=pd.Timestamp('2019-01-01')]
cash=INIT;shares={n:0.0 for n in STOCKS};entry={n:0.0 for n in STOCKS};high={n:0.0 for n in STOCKS}
accel={n:False for n in STOCKS};sold_today={n:False for n in STOCKS}
trades=[]

for date in dates:
    for n in STOCKS:sold_today[n]=False
    px={n:raw[n][raw[n]['date']==date]['close'].iloc[0]for n in STOCKS if len(raw[n][raw[n]['date']==date])>0}
    for n in STOCKS:
        if shares[n]<=0:continue
        cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
        if cp<=0 or len(r)==0:continue
        if cp>high[n]:high[n]=cp
        pnl=cp/entry[n]-1;dd=cp/high[n]-1
        do=False;why='';sell_px=cp
        if pnl<=-HARD_STOP:do=True;why='hard'
        elif accel[n]:
            if pnl>=TAKE_PROFIT_HI:do=True;why='accel_tp25'
            elif dd<=-TRAIL_STOP:
                floor_px=entry[n]*(1+TAKE_PROFIT);stop_px=max(high[n]*(1-TRAIL_STOP),floor_px)
                if cp<=stop_px:
                    do=True;sell_px=max(cp,floor_px);why='accel_floor'
        elif dd<=-TRAIL_STOP:do=True;why='trail'
        elif pnl>=TAKE_PROFIT:
            d2=r.iloc[0].get('bb_up_d2')
            if not pd.isna(d2) and d2>0:accel[n]=True
            else:do=True;why='tp20'
        if do:
            cash+=shares[n]*sell_px*(1-COMM-SLIP)
            trades.append({'date':date,'name':n,'pnl':(sell_px/entry[n]-1)*100,'why':why,'cp':cp,'sell_px':sell_px})
            shares[n]=0;entry[n]=0;high[n]=0;accel[n]=False
            sold_today[n]=True
    nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)
    for n in STOCKS:
        if sold_today[n]:continue
        if shares[n]>0:continue
        cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
        if cp<=0 or len(r)==0:continue
        ok,sc=check_buy(r.iloc[0],n)
        if not ok:continue
        val=min(cash,INIT*MAX_POS)
        if val>5000:
            qty=val/cp*(1-COMM-SLIP);shares[n]=qty;cash-=val
            entry[n]=cp;high[n]=cp

print('='*60)
print('v13 BB加速保底止盈 - 卖出分类分析')
print('='*60)
reasons=Counter(t['why']for t in trades)
for r,c in reasons.most_common():
    print(f'  {r}: {c}笔')

print()
for why in ['tp20','accel_tp25','accel_floor','trail','hard']:
    ts=[t for t in trades if t['why']==why]
    if ts:
        pnls=[t['pnl']for t in ts]
        avg=np.mean(pnls)
        wr=(sum(1 for p in pnls if p>0))/len(pnls)*100
        print(f'  {why:<13}: {len(ts):>2}笔 均{avg:>+5.1f}%  胜率{wr:>.0f}%  范围{min(pnls):+.1f}~{max(pnls):+.1f}')

# BB加速相关统计
print(f'\n{"="*60}')
print('BB加速效果分析')
print(f'{"="*60}')
accel_enter=len([t for t in trades if t['why'] in ('accel_tp25','accel_floor')])
tp25=len([t for t in trades if t['why']=='accel_tp25'])
floor=len([t for t in trades if t['why']=='accel_floor'])
tp20n=len([t for t in trades if t['why']=='tp20'])
print(f'  触发BB加速: {accel_enter}次 (其中{tp25}次到25%, {floor}次回保底20%)')
print(f'  正常20%止盈: {tp20n}次')
if accel_enter>0:
    avg_accel=np.mean([t['pnl']for t in trades if t['why']in('accel_tp25','accel_floor')])
    print(f'  BB加速交易均盈: {avg_accel:+.1f}% (vs 正常止盈20%)')

# 分股票
print(f'\n分股票BB加速统计:')
for name in STOCKS:
    at=[t for t in trades if t['name']==name and t['why']in('accel_tp25','accel_floor','tp20')]
    if at:
        reasons_stock=Counter(t['why']for t in at)
        info=', '.join(f'{r}({c})' for r,c in reasons_stock.items())
        avg=np.mean([t['pnl']for t in at])
        print(f'  {name}: {info} 均{avg:+.1f}%')
