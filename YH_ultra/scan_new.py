# -*- coding: utf-8 -*-
"""扫描建设银行、中国平安的最优买入参数"""
import sys,io,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak,pandas as pd,numpy as np
warnings.filterwarnings('ignore')

NEW_STOCKS={'建设银行':'sh601939','中国平安':'sh601318'}
INIT=1_000_000;COMM=0.0003;SLIP=0.0001;MAX_POS=0.25
TRAIL_STOP=0.07;HARD_STOP=0.10;COOLDOWN=20

def fetch():
    dfs={}
    for n,s in NEW_STOCKS.items():
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

def check_buy(row,name,params):
    rsi_th=params['rsi'];bb_th=params['bb']
    if pd.isna(row['bb_lo'])or pd.isna(row['rsi']):return False,0
    rsi=row['rsi'];c=row['close'];lo=row['bb_lo'];up=row['bb_up']
    if up<=lo:return False,0
    dist=(c-lo)/(up-lo)
    sc=(1 if rsi<=rsi_th else 0)+(1 if dist<=bb_th else 0)
    if rsi<=30:sc+=1
    return sc>=1,sc

def run_stock(name, rsi_th, bb_th, tp_val):
    tp_hi=tp_val+0.05
    raw=fetch();dfs={n:add_indicators(d)for n,d in raw.items()}
    dates=sorted(set.intersection(*[set(d['date'])for d in dfs.values()]))
    dates=[d for d in dates if d>=pd.Timestamp('2019-01-01')]
    # Only the target stock
    single={name:NEW_STOCKS[name]}
    cash=INIT;shares={name:0.0};entry={name:0.0};high={name:0.0}
    accel={name:False};cooldown={name:0};navs=[];trades=[]

    for date in dates:
        n=name
        cp=raw[n][raw[n]['date']==date]['close'].iloc[0]if len(raw[n][raw[n]['date']==date])>0 else 0
        sold=False
        if shares[n]>0 and cp>0:
            r=dfs[n][dfs[n]['date']==date]
            if len(r)>0:
                if cp>high[n]:high[n]=cp
                pnl=cp/entry[n]-1;dd=cp/high[n]-1
                do=False;sell_px=cp
                if pnl<=-HARD_STOP:do=True
                elif accel[n]:
                    if pnl>=tp_hi:do=True
                    elif dd<=-TRAIL_STOP:
                        floor=entry[n]*(1+tp_val);stop_px=max(high[n]*(1-TRAIL_STOP),floor)
                        if cp<=stop_px:do=True;sell_px=max(cp,floor)
                elif dd<=-TRAIL_STOP:do=True
                elif pnl>=tp_val:
                    d2=r.iloc[0].get('bb_up_d2')
                    if not pd.isna(d2) and d2>0:accel[n]=True
                    else:do=True
                if do:
                    cash+=shares[n]*sell_px*(1-COMM-SLIP)
                    trades.append((sell_px/entry[n]-1)*100)
                    shares[n]=0;entry[n]=0;high[n]=0;accel[n]=False;sold=True
                    if sell_px/entry[n]-1<=-HARD_STOP:cooldown[n]=COOLDOWN
        nav=cash+shares[n]*cp
        if cooldown[n]>0:cooldown[n]-=1
        if not sold and shares[n]==0 and cooldown[n]==0 and cp>0:
            r=dfs[n][dfs[n]['date']==date]
            if len(r)>0:
                ok,sc=check_buy(r.iloc[0],n,{'rsi':rsi_th,'bb':bb_th})
                if ok:
                    val=min(cash,nav*MAX_POS)
                    if val>5000:
                        qty=val/cp*(1-COMM-SLIP);shares[n]=qty;cash-=val
                        entry[n]=cp;high[n]=cp
        nav=cash+shares[n]*cp
        navs.append(nav)

    ndf=pd.DataFrame(navs,columns=['nav']);final=ndf['nav'].iloc[-1]
    ret=(final/INIT-1)*100;ann=((1+ret/100)**(252/len(ndf))-1)*100
    dr=ndf['nav'].pct_change().dropna();vol=dr.std()*np.sqrt(252)*100
    sr=(ann-2)/vol if vol>0 else 0
    mdd=((ndf['nav']-ndf['nav'].cummax())/ndf['nav'].cummax()).min()*100
    pnls=np.array(trades)
    wr=(pnls>0).sum()/len(pnls)*100 if len(pnls)>0 else 0
    return ann,sr,mdd,wr,len(trades),ret

print("获取数据...")
print("\n===== 建设银行 (601939) =====")
print(f"{'RSI':>5} {'BB':>6} {'TP':>5} {'年化':>8} {'夏普':>6} {'胜率':>5} {'交易':>4}")
print('─'*55)
best_ann=-999;best_cfg=None
for rsi_th in [30,32,35,38,40]:
    for bb_th in [0.08,0.10,0.12,0.15]:
        for tp in [15,18,20,22,25]:
            tp_val=tp/100.0
            ann,sr,mdd,wr,nt,ret=run_stock('建设银行',rsi_th,bb_th,tp_val)
            mark=''
            if ann>best_ann:best_ann=ann;best_cfg=(rsi_th,bb_th,tp,ann,sr,nt);mark=' ←'
            if ann>25:
                print(f" RSI{rsi_th:>2} BB{bb_th:>.2f} TP{tp:>3}% {ann:>+7.1f}% {sr:>5.2f} {wr:>4.0f}% {nt:>4}{mark}")

print(f"\n→ 建设银行最优: RSI≤{best_cfg[0]} BB≤{best_cfg[1]:.2f} TP={best_cfg[2]}% 年化{best_cfg[3]:+.1f}% 夏普{best_cfg[4]:.2f} 交易{best_cfg[5]}笔")

print("\n===== 中国平安 (601318) =====")
print(f"{'RSI':>5} {'BB':>6} {'TP':>5} {'年化':>8} {'夏普':>6} {'胜率':>5} {'交易':>4}")
print('─'*55)
best_ann=-999;best_cfg=None
for rsi_th in [30,32,35,38,40,42]:
    for bb_th in [0.05,0.08,0.10,0.12,0.15,0.18]:
        for tp in [15,18,20,22,25]:
            tp_val=tp/100.0
            ann,sr,mdd,wr,nt,ret=run_stock('中国平安',rsi_th,bb_th,tp_val)
            mark=''
            if ann>best_ann:best_ann=ann;best_cfg=(rsi_th,bb_th,tp,ann,sr,nt);mark=' ←'
            if ann>18:
                print(f" RSI{rsi_th:>2} BB{bb_th:>.2f} TP{tp:>3}% {ann:>+7.1f}% {sr:>5.2f} {wr:>4.0f}% {nt:>4}{mark}")

print(f"\n→ 中国平安最优: RSI≤{best_cfg[0]} BB≤{best_cfg[1]:.2f} TP={best_cfg[2]}% 年化{best_cfg[3]:+.1f}% 夏普{best_cfg[4]:.2f} 交易{best_cfg[5]}笔")
