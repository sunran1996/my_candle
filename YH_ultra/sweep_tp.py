# -*- coding: utf-8 -*-
"""每只股票独立止盈阈值扫描"""
import sys,io,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak,pandas as pd,numpy as np
from collections import Counter
warnings.filterwarnings('ignore')

STOCKS={'山东高速':'sh600350','渝农商行':'sh601077','皖通高速':'sh600012','江苏银行':'sh600919'}
INIT=1_000_000;COMM=0.0003;SLIP=0.0001;MAX_POS=0.25
TRAIL_STOP=0.07;HARD_STOP=0.10;COOLDOWN=20
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

def run(tp_dict):
    """tp_dict = {name: (take_profit, take_profit_hi)}"""
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
            tp,tp_hi=tp_dict[n]
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
                cash+=shares[n]*sell_px*(1-COMM-SLIP)
                trades.append({'pnl':(sell_px/entry[n]-1)*100,'why':why,'name':n})
                shares[n]=0;entry[n]=0;high[n]=0;accel[n]=False;sold_today[n]=True
                if sell_px/entry[n]-1<=-HARD_STOP:cooldown[n]=COOLDOWN

        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS);navs.append(nav)
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
    return ann,sr,mdd,wr,len(trades),aw,al,ret

# ====== 扫描 ======
print("获取数据...")
# 基准: 全部20%
print("\n基准: 全部 TP=20% HI=25%")
base_ann,base_sr,base_mdd,base_wr,base_nt,base_aw,base_al,base_ret = run({n:(0.20,0.25)for n in STOCKS})
print(f"年化{base_ann:+.1f}% 夏普{base_sr:.2f} 回撤{base_mdd:+.1f}% 胜率{base_wr:.0f}% 交易{base_nt}笔 累计{base_ret:+.0f}%")

print(f"\n{'='*85}")
print(f"  每只股票独立止盈扫描 (保持HI=TP+5%)")
print(f"{'='*85}")

for name in STOCKS:
    print(f"\n── {name} ──")
    best_ann=-999;best_tp=0
    for tp in [15,16,17,18,19,20,21,22,23,24,25]:
        tp_val=tp/100.0;tp_hi_val=tp_val+0.05
        tp_dict={n:(0.20,0.25)for n in STOCKS}
        tp_dict[name]=(tp_val,tp_hi_val)
        ann,sr,mdd,wr,nt,aw,al,ret=run(tp_dict)
        mark=' ←' if ann>best_ann else ''
        print(f"  TP={tp:>2}%: 年化{ann:>+6.1f}% 夏普{sr:>5.2f} 回撤{mdd:>+5.1f}% 胜率{wr:>4.0f}% 交易{nt:>4}笔 均盈{aw:>+5.1f}%{mark}")
        if ann>best_ann:best_ann=ann;best_tp=tp
    print(f"  → {name}最优: TP={best_tp}% ({best_tp+5}%加速)")

# 组合最优
print(f"\n{'='*85}")
print(f"  验证组合最优")
print(f"{'='*85}")
# 先找到每只最优
best_per_stock={}
for name in STOCKS:
    best_ann=-999;best_tp=20
    for tp in [15,16,17,18,19,20,21,22,23,24,25]:
        tp_val=tp/100.0;tp_hi_val=tp_val+0.05
        tp_dict={n:(0.20,0.25)for n in STOCKS}
        tp_dict[name]=(tp_val,tp_hi_val)
        ann,_,_,_,_,_,_,_=run(tp_dict)
        if ann>best_ann:best_ann=ann;best_tp=tp
    best_per_stock[name]=(best_tp/100.0,best_tp/100.0+0.05)
    print(f"{name}: TP={best_tp}% HI={best_tp+5}%")

ann,sr,mdd,wr,nt,aw,al,ret=run(best_per_stock)
print(f"\n组合效果: 年化{ann:+.1f}% 夏普{sr:.2f} 回撤{mdd:+.1f}% 胜率{wr:.0f}% 交易{nt}笔 累计{ret:+.0f}%")
print(f"vs 基准: 年化{ann-base_ann:+.1f}% Δ 夏普{sr-base_sr:+.2f} Δ")
