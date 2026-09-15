# -*- coding: utf-8 -*-
"""v13 vs v16 三年滚动窗口回测对比"""
import sys,io,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak,pandas as pd,numpy as np
warnings.filterwarnings('ignore')

STOCKS={'山东高速':'sh600350','渝农商行':'sh601077','皖通高速':'sh600012','江苏银行':'sh600919'}
INIT=1_000_000;COMM=0.0003;SLIP=0.0001;MAX_POS=0.25
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
    rsi_th=bp['rsi'];bb_th=bp['bb']
    sc=(1 if rsi<=rsi_th else 0)+(1 if dist<=bb_th else 0)
    if rsi<=30:sc+=1
    return sc>=1,sc

# ============ v13: BB加速+保底20%, 正常买入 ============
def run_v13(start,end):
    raw=fetch();dfs={n:add_indicators(d)for n,d in raw.items()}
    dates=sorted(set.intersection(*[set(d['date'])for d in dfs.values()]))
    dates=[d for d in dates if start<=d<=end]
    cash=INIT;shares={n:0.0 for n in STOCKS};entry={n:0.0 for n in STOCKS};high={n:0.0 for n in STOCKS}
    accel={n:False for n in STOCKS};sold_today={n:False for n in STOCKS}
    navs=[];trades=[]

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
            if pnl<=-0.10:do=True;why='hard'
            elif accel[n]:
                if pnl>=0.25:do=True;why='accel25'
                elif dd<=-0.08:
                    floor=entry[n]*1.20;stop_px=max(high[n]*0.92,floor)
                    if cp<=stop_px:do=True;sell_px=max(cp,floor);why='accel_floor'
            elif dd<=-0.08:do=True;why='trail'
            elif pnl>=0.20:
                d2=r.iloc[0].get('bb_up_d2')
                if not pd.isna(d2) and d2>0:accel[n]=True
                else:do=True;why='tp20'
            if do:
                cash+=shares[n]*sell_px*(1-COMM-SLIP)
                trades.append((sell_px/entry[n]-1)*100)
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
            val=min(cash,nav*MAX_POS)
            if val>5000:
                qty=val/cp*(1-COMM-SLIP);shares[n]=qty;cash-=val
                entry[n]=cp;high[n]=cp
        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)
        navs.append(nav)

    ndf=pd.DataFrame(navs,columns=['nav']);final=ndf['nav'].iloc[-1]
    ret=(final/INIT-1)*100;ann=((1+ret/100)**(252/len(ndf))-1)*100 if len(ndf)>0 else 0
    dr=ndf['nav'].pct_change().dropna();vol=dr.std()*np.sqrt(252)*100
    sr=(ann-2)/vol if vol>0 else 0
    mdd=((ndf['nav']-ndf['nav'].cummax())/ndf['nav'].cummax()).min()*100
    pnls=np.array(trades)
    wr=(pnls>0).sum()/len(pnls)*100 if len(pnls)>0 else 0
    aw=pnls[pnls>0].mean()if(pnls>0).sum()>0 else 0
    al=pnls[pnls<0].mean()if(pnls<0).sum()>0 else 0
    return ann,ret,sr,mdd,wr,len(trades),aw,al

# ============ v16: 三阶止盈(减速15%/正常20%/加速25%) ============
def run_v16(start,end):
    raw=fetch();dfs={n:add_indicators(d)for n,d in raw.items()}
    dates=sorted(set.intersection(*[set(d['date'])for d in dfs.values()]))
    dates=[d for d in dates if start<=d<=end]
    cash=INIT;shares={n:0.0 for n in STOCKS};entry={n:0.0 for n in STOCKS};high={n:0.0 for n in STOCKS}
    accel={n:False for n in STOCKS};sold_today={n:False for n in STOCKS}
    navs=[];trades=[]

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
            if pnl<=-0.10:do=True;why='hard'
            elif accel[n]:
                if pnl>=0.25:do=True;why='accel25'
                elif dd<=-0.08:
                    floor=entry[n]*1.20;stop_px=max(high[n]*0.92,floor)
                    if cp<=stop_px:do=True;sell_px=max(cp,floor);why='accel_floor'
            elif pnl>=0.15:
                d2=r.iloc[0].get('bb_up_d2')
                if not pd.isna(d2) and d2<0:do=True;why='tp15'
                elif pnl>=0.20:
                    if not pd.isna(d2) and d2>0:accel[n]=True
                    else:do=True;why='tp20'
            elif dd<=-0.08:do=True;why='trail'
            if do:
                cash+=shares[n]*sell_px*(1-COMM-SLIP)
                trades.append((sell_px/entry[n]-1)*100)
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
            val=min(cash,nav*MAX_POS)
            if val>5000:
                qty=val/cp*(1-COMM-SLIP);shares[n]=qty;cash-=val
                entry[n]=cp;high[n]=cp
        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)
        navs.append(nav)

    ndf=pd.DataFrame(navs,columns=['nav']);final=ndf['nav'].iloc[-1]
    ret=(final/INIT-1)*100;ann=((1+ret/100)**(252/len(ndf))-1)*100 if len(ndf)>0 else 0
    dr=ndf['nav'].pct_change().dropna();vol=dr.std()*np.sqrt(252)*100
    sr=(ann-2)/vol if vol>0 else 0
    mdd=((ndf['nav']-ndf['nav'].cummax())/ndf['nav'].cummax()).min()*100
    pnls=np.array(trades)
    wr=(pnls>0).sum()/len(pnls)*100 if len(pnls)>0 else 0
    aw=pnls[pnls>0].mean()if(pnls>0).sum()>0 else 0
    al=pnls[pnls<0].mean()if(pnls<0).sum()>0 else 0
    return ann,ret,sr,mdd,wr,len(trades),aw,al

# ============ 滚动窗口 ============
windows=[
    ('2019-2021', pd.Timestamp('2019-01-01'), pd.Timestamp('2021-12-31')),
    ('2020-2022', pd.Timestamp('2020-01-01'), pd.Timestamp('2022-12-31')),
    ('2021-2023', pd.Timestamp('2021-01-01'), pd.Timestamp('2023-12-31')),
    ('2022-2024', pd.Timestamp('2022-01-01'), pd.Timestamp('2024-12-31')),
    ('2023-2025', pd.Timestamp('2023-01-01'), pd.Timestamp('2025-12-31')),
    ('2024-now',  pd.Timestamp('2024-01-01'), pd.Timestamp.now()),
    ('2019-now',  pd.Timestamp('2019-01-01'), pd.Timestamp.now()),
]

print("="*100)
print("  v13 (BB加速25%+保底20%) vs v16 (三阶: BB减速15% / 正常20% / BB加速25%保底)")
print("  三年滚动窗口回测, 初始资金100万")
print("="*100)

print(f"\n{'窗口':<14} {'策略':<6} {'年化':>8} {'夏普':>6} {'回撤':>7} {'胜率':>6} {'交易':>5} {'均盈':>7} {'均亏':>7} {'累计':>8}")
print('─'*100)

for label,start,end in windows:
    a1,r1,s1,m1,w1,n1,aw1,al1=run_v13(start,end)
    a2,r2,s2,m2,w2,n2,aw2,al2=run_v16(start,end)

    # 比较标记
    best=''
    if a1>a2+1.0:best=' ←v13'
    elif a2>a1+1.0:best=' ←v16'

    print(f'{label:<14} {"v13":<6} {a1:>+7.1f}% {s1:>5.2f} {m1:>+6.1f}% {w1:>5.0f}% {n1:>5} {aw1:>+6.1f}% {al1:>+6.1f}% {r1:>+7.1f}%{best if best==" ←v13" else ""}')
    print(f'{"":14} {"v16":<6} {a2:>+7.1f}% {s2:>5.2f} {m2:>+6.1f}% {w2:>5.0f}% {n2:>5} {aw2:>+6.1f}% {al2:>+6.1f}% {r2:>+7.1f}%{best if best==" ←v16" else ""}')

    # 差值
    if a1 and a2:
        da=a2-a1;ds=s2-s1;dm=m2-m1
        print(f'{"":14} {"Δ":<6} {da:>+7.1f}% {ds:>+5.2f} {dm:>+6.1f}%')
    print()

# ============ 逐年对比 ============
print("="*100)
print("  逐年对比 (v13 vs v16)")
print("="*100)
print(f"{'年份':<8} {'v13年化':>9} {'v16年化':>9} {'差值':>8} {'v13夏普':>8} {'v16夏普':>8} {'v13回撤':>8} {'v16回撤':>8}")
print('─'*80)

for yr in range(2019,2027):
    s=pd.Timestamp(f'{yr}-01-01')
    e=pd.Timestamp(f'{yr}-12-31') if yr<2026 else pd.Timestamp.now()
    a1,_,s1,m1,_,_,_,_=run_v13(s,e)
    a2,_,s2,m2,_,_,_,_=run_v16(s,e)
    if a1 or a2:
        da=a2-a1
        print(f'{yr:<8} {a1:>+8.1f}% {a2:>+8.1f}% {da:>+7.1f}% {s1:>7.2f} {s2:>7.2f} {m1:>+7.1f}% {m2:>+7.1f}%')
