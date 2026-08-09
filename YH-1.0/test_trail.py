# -*- coding: utf-8 -*-
"""测试不同移动止损阈值: 5% 6% 7% 8%"""
import sys,io,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak,pandas as pd,numpy as np
warnings.filterwarnings('ignore')

STOCKS={'山东高速':'sh600350','渝农商行':'sh601077','皖通高速':'sh600012','江苏银行':'sh600919'}
INIT=1_000_000;COMM=0.0003;SLIP=0.0001;MAX_POS=0.25
HARD_STOP=0.10;COOLDOWN=20
TAKE_PROFIT=0.20;TAKE_PROFIT_HI=0.25
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

def run(ts, label):
    """用给定的移动止损百分比回测"""
    raw=fetch();dfs={n:add_indicators(d)for n,d in raw.items()}
    dates=sorted(set.intersection(*[set(d['date'])for d in dfs.values()]))
    dates=[d for d in dates if d>=pd.Timestamp('2019-01-01')]
    cash=INIT;shares={n:0.0 for n in STOCKS};entry={n:0.0 for n in STOCKS};high={n:0.0 for n in STOCKS}
    accel={n:False for n in STOCKS};cooldown={n:0 for n in STOCKS}
    sold_today={n:False for n in STOCKS};trades=[]

    for date in dates:
        for n in STOCKS:sold_today[n]=False
        px={n:raw[n][raw[n]['date']==date]['close'].iloc[0]for n in STOCKS if len(raw[n][raw[n]['date']==date])>0}

        for n in STOCKS:
            if shares[n]<=0:continue
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            if cp>high[n]:high[n]=cp
            pnl=cp/entry[n]-1;dd=cp/high[n]-1
            do=False;sell_px=cp;why=''

            if pnl<=-HARD_STOP:
                do=True;why='hard'
            elif accel[n]:
                if pnl>=TAKE_PROFIT_HI:
                    do=True;why='accel_tp25'
                elif dd<=-ts:
                    floor=entry[n]*(1+TAKE_PROFIT);stop_px=max(high[n]*(1-ts),floor)
                    if cp<=stop_px:do=True;sell_px=max(cp,floor);why='accel_floor'
            elif dd<=-ts:
                do=True;why='trail'
            elif pnl>=TAKE_PROFIT:
                d2=r.iloc[0].get('bb_up_d2')
                if not pd.isna(d2) and d2>0:accel[n]=True
                else:do=True;why='tp20'

            if do:
                cash+=shares[n]*sell_px*(1-COMM-SLIP)
                trades.append({'pnl':(sell_px/entry[n]-1)*100,'why':why})
                shares[n]=0;entry[n]=0;high[n]=0;accel[n]=False
                sold_today[n]=True
                pnl_real=sell_px/entry[n]-1
                if pnl_real<=-HARD_STOP:cooldown[n]=COOLDOWN

        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)
        for n in STOCKS:
            if cooldown[n]>0:cooldown[n]-=1

        for n in STOCKS:
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

        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)

    ndf=pd.DataFrame([nav]);final=nav  # Just store final value
    # Actually need full nav series for stats
    return trades, nav

# We need full nav tracking - let me redo properly
def run_full(ts):
    raw=fetch();dfs={n:add_indicators(d)for n,d in raw.items()}
    dates=sorted(set.intersection(*[set(d['date'])for d in dfs.values()]))
    dates=[d for d in dates if d>=pd.Timestamp('2019-01-01')]
    cash=INIT;shares={n:0.0 for n in STOCKS};entry={n:0.0 for n in STOCKS};high={n:0.0 for n in STOCKS}
    accel={n:False for n in STOCKS};cooldown={n:0 for n in STOCKS}
    sold_today={n:False for n in STOCKS};navs=[];trades=[]

    for date in dates:
        for n in STOCKS:sold_today[n]=False
        px={n:raw[n][raw[n]['date']==date]['close'].iloc[0]for n in STOCKS if len(raw[n][raw[n]['date']==date])>0}

        for n in STOCKS:
            if shares[n]<=0:continue
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            if cp>high[n]:high[n]=cp
            pnl=cp/entry[n]-1;dd=cp/high[n]-1
            do=False;sell_px=cp;why=''

            if pnl<=-HARD_STOP:
                do=True;why='hard'
            elif accel[n]:
                if pnl>=TAKE_PROFIT_HI:
                    do=True;why='accel_tp25'
                elif dd<=-ts:
                    floor=entry[n]*(1+TAKE_PROFIT);stop_px=max(high[n]*(1-ts),floor)
                    if cp<=stop_px:do=True;sell_px=max(cp,floor);why='accel_floor'
            elif dd<=-ts:
                do=True;why='trail'
            elif pnl>=TAKE_PROFIT:
                d2=r.iloc[0].get('bb_up_d2')
                if not pd.isna(d2) and d2>0:accel[n]=True
                else:do=True;why='tp20'

            if do:
                cash+=shares[n]*sell_px*(1-COMM-SLIP)
                trades.append({'pnl':(sell_px/entry[n]-1)*100,'why':why})
                shares[n]=0;entry[n]=0;high[n]=0;accel[n]=False
                sold_today[n]=True
                pnl_real=sell_px/entry[n]-1
                if pnl_real<=-HARD_STOP:cooldown[n]=COOLDOWN

        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)
        navs.append(nav)

        for n in STOCKS:
            if cooldown[n]>0:cooldown[n]-=1

        for n in STOCKS:
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

        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)

    ndf=pd.DataFrame(navs,columns=['nav']);final=ndf['nav'].iloc[-1]
    ret=(final/INIT-1)*100;ann=((1+ret/100)**(252/len(ndf))-1)*100
    dr=ndf['nav'].pct_change().dropna();vol=dr.std()*np.sqrt(252)*100
    sr=(ann-2)/vol if vol>0 else 0
    mdd=((ndf['nav']-ndf['nav'].cummax())/ndf['nav'].cummax()).min()*100

    pnls=np.array([t['pnl']for t in trades])
    wr=(pnls>0).sum()/len(pnls)*100 if len(pnls)>0 else 0
    aw=pnls[pnls>0].mean()if(pnls>0).sum()>0 else 0
    al=pnls[pnls<0].mean()if(pnls<0).sum()>0 else 0

    # Reason breakdown
    from collections import Counter
    reasons=Counter(t['why']for t in trades)

    return {'ts':ts,'ret':ret,'ann':ann,'sr':sr,'mdd':mdd,
            'n_trades':len(trades),'wr':wr,'aw':aw,'al':al,
            'reasons':reasons,'trades':trades}

print("获取数据...")
raw=fetch()  # Cache fetch - but run_full re-fetches... simpler to just call independently

results=[]
for ts_pct in [5,6,7,8]:
    ts=ts_pct/100.0
    print(f"回测 TS={ts_pct}%...")
    r=run_full(ts)
    results.append(r)

print(f"\n{'='*90}")
print(f"  移动止损阈值对策略表现的影响")
print(f"{'='*90}")
print(f"{'TS':>5} {'年化':>8} {'夏普':>6} {'回撤':>7} {'累计':>7} {'交易':>5} {'胜率':>5} {'均盈':>7} {'均亏':>7}")
print(f"{'─'*90}")

best_ann=-999;best_ts=0
for r in results:
    if r['ann']>best_ann:best_ann=r['ann'];best_ts=r['ts']
    print(f"{r['ts']*100:4.0f}% {r['ann']:>+7.1f}% {r['sr']:>5.2f} {r['mdd']:>+6.1f}% {r['ret']:>+6.0f}% {r['n_trades']:>5} {r['wr']:>4.0f}% {r['aw']:>+6.1f}% {r['al']:>+6.1f}%")

print(f"\n  → 最优: TS={best_ts*100:.0f}% 年化{best_ann:+.1f}%")

# 卖出理由分布
print(f"\n{'='*90}")
print(f"  卖出理由分布")
print(f"{'='*90}")
print(f"{'TS':>5} {'hard':>6} {'trail':>6} {'tp20':>6} {'accel_tp25':>10} {'accel_floor':>12} {'total':>6}")
print(f"{'─'*90}")
for r in results:
    rc=r['reasons']
    total=sum(rc.values())
    print(f"{r['ts']*100:4.0f}% {rc.get('hard',0):>6} {rc.get('trail',0):>6} {rc.get('tp20',0):>6} {rc.get('accel_tp25',0):>10} {rc.get('accel_floor',0):>12} {total:>6}")

# trail trades detail
print(f"\n{'='*90}")
print(f"  trail (移动止损) 交易均盈亏")
print(f"{'='*90}")
for r in results:
    trail_trades=[t for t in r['trades'] if t['why']=='trail']
    if trail_trades:
        pnls=[t['pnl']for t in trail_trades]
        print(f"  TS={r['ts']*100:.0f}%: {len(trail_trades):>2}笔 均{np.mean(pnls):>+5.1f}%  范围{min(pnls):+.1f}~{max(pnls):+.1f}")
