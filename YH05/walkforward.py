# -*- coding: utf-8 -*-
"""YH05 Walk-Forward 滚动回测"""
import sys,io,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak,pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif']=['SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False

INIT=1_000_000;COMM=0.0003;SLIP=0.0001
BB_P=45;BB_S=2.0;RSI_P=14;RSI_L=30;RSI_H=70;ERS=65;BA=0.001
HARD_STOP=0.12;NAV_STOP=0.13;TRAIL=0.10;REBAL=5;MOM=10

MAIN_SYM='sh512890';MAIN_NAME='hl'
GROWTH={'cy':'sz159915'}
GN='cy'

def fetch():
    dfs={}
    for n,s in {**GROWTH,MAIN_NAME:MAIN_SYM}.items():
        df=ak.fund_etf_hist_sina(symbol=s);df['date']=pd.to_datetime(df['date'])
        dfs[n]=df[['date','close']].sort_values('date').reset_index(drop=True)
    return dfs

def add_main(df):
    df=df.copy();r=df['close'].pct_change().fillna(0);r[abs(r)>0.1]=0
    df['adj']=(1+r).cumprod()
    df['ma']=df['adj'].rolling(BB_P).mean();df['std']=df['adj'].rolling(BB_P).std()
    df['up']=df['ma']+BB_S*df['std'];df['lo']=df['ma']-BB_S*df['std']
    df['ua']=df['up'].diff().diff().rolling(3,min_periods=1).mean()
    df['pa']=df['adj'].diff().diff().rolling(3,min_periods=1).mean()
    d=df['adj'].diff();g=d.clip(lower=0);l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/RSI_P,adjust=False).mean()/l.ewm(alpha=1/RSI_P,adjust=False).mean().replace(0,np.nan))
    return df

def main_signal(row,pbw):
    adj,rsi,up,lo=row['adj'],row['rsi'],row['up'],row['lo']
    if pd.isna(lo)or pd.isna(rsi):return'HOLD',pbw
    bw=(up-lo)/row['ma']if row['ma']>0 else 0.1
    exp=(pbw is not None and bw>pbw);nbw=bw
    bb_buy=(adj<=lo);bb_sell=(adj>=up);rsi_buy=(rsi<=RSI_L)
    if exp:
        raw_sell=(bb_sell and rsi>=ERS)
        ua=row['ua']if not pd.isna(row['ua'])else 0
        pa=row['pa']if not pd.isna(row['pa'])else 0
        sell_sig=raw_sell and not((ua>BA)and(pa>0))
        buy_sig=(bb_buy or rsi_buy)
    else:
        buy_sig=(bb_buy or rsi_buy);sell_sig=(bb_sell or rsi>=RSI_H)
    if buy_sig:return'BUY',nbw
    elif sell_sig:return'SELL',nbw
    else:return'HOLD',nbw

def add_growth(df):
    df=df.copy();df['mom']=df['close']/df['close'].shift(MOM)-1
    e10=df['close'].ewm(span=10,adjust=False).mean();e20=df['close'].ewm(span=20,adjust=False).mean()
    df['macd']=e10-e20;df['macd_s']=df['macd'].ewm(span=7,adjust=False).mean()
    df['macd_h']=df['macd']-df['macd_s'];df['ma20']=df['close'].rolling(20).mean()
    df['macd_line']=df['macd']
    return df

def backtest_on_dates(dates,raw,df_main,dfs_growth):
    ALL=[GN,MAIN_NAME]
    start_px=raw[MAIN_NAME]['close'].iloc[0]
    cash=0.0
    shares={n:0.0 for n in ALL};shares[MAIN_NAME]=INIT/start_px*(1-COMM-SLIP)
    position=MAIN_NAME;peak=INIT;navs=[]
    sc=2;pbw=None;ep=start_px;hse=0;last_rb=None;stopped=False
    for date in dates:
        px={}
        for n in ALL:
            r=raw[n][raw[n]['date']==date]
            if len(r):px[n]=r['close'].values[0]
        nav=cash+sum(shares[n]*px.get(n,0)for n in ALL)
        if nav>peak:peak=nav
        dd_nav=(nav-peak)/peak if peak>0 else 0
        if dd_nav<-NAV_STOP and any(shares[n]>0 for n in ALL):
            for n in ALL:
                if shares[n]>0 and n in px:cash+=shares[n]*px[n]*(1-COMM-SLIP);shares[n]=0.0
            peak=nav;position=None;stopped=True
            navs.append({'date':date,'nav':nav});continue
        mr=df_main[df_main['date']==date]
        if len(mr)==0:navs.append({'date':date,'nav':nav});continue
        sig,pbw=main_signal(mr.iloc[0],pbw)
        mp=px.get(MAIN_NAME,0)
        if position==MAIN_NAME and mp>0 and ep>0 and mp<ep*(1-HARD_STOP):
            cash+=shares[MAIN_NAME]*mp*(1-COMM-SLIP);shares[MAIN_NAME]=0.0;position=None
        if sig=='BUY'and position!=MAIN_NAME:
            if position:
                for n in GROWTH:
                    if shares[n]>0 and n in px:cash+=shares[n]*px[n]*(1-COMM-SLIP);shares[n]=0.0
            if mp>0:
                val=min(cash,nav);shares[MAIN_NAME]+=val/mp*(1-COMM-SLIP);cash-=val
                position=MAIN_NAME;ep=mp;stopped=False
        elif sig=='SELL'and position==MAIN_NAME:
            if shares[MAIN_NAME]>0 and mp>0:cash+=shares[MAIN_NAME]*mp*(1-COMM-SLIP);shares[MAIN_NAME]=0.0;position=None
        elif sig=='HOLD'and position!=MAIN_NAME:
            if position and position in px:
                if px[position]>hse:hse=px[position]
                if px[position]<hse*(1-TRAIL):
                    cash+=shares[position]*px[position]*(1-COMM-SLIP);shares[position]=0.0
                    position=None;stopped=True
            if stopped or shares.get(MAIN_NAME,0)>0:navs.append({'date':date,'nav':nav});continue
            days=(date-last_rb).days if last_rb else 999
            if days>=REBAL and position is None:
                gdf=dfs_growth[GN]
                gr=gdf[gdf['date']==date]
                if len(gr)==0:navs.append({'date':date,'nav':nav});continue
                macd_h=gr['macd_h'].values[0];macd_line=gr['macd_line'].values[0]
                ma20=gr['ma20'].values[0];close_px=gr['close'].values[0]
                if macd_h>0 and close_px>ma20:
                    if GN in px:
                        pos_pct=1.0 if macd_line>0 else 0.3
                        val=min(cash,nav*pos_pct)
                        if val>100:shares[GN]=val/px[GN]*(1-COMM-SLIP);cash-=val;position=GN;hse=px[GN]
                last_rb=date
        nav=cash+sum(shares[n]*px.get(n,0)for n in ALL)
        navs.append({'date':date,'nav':nav})
    return navs

print("获取数据...")
raw=fetch()
df_main=add_main(raw[MAIN_NAME])
dfs_growth={n:add_growth(d)for n,d in raw.items()if n!=MAIN_NAME}

all_dates=sorted(set.intersection(*[set(d['date'])for d in raw.values()]))
all_dates=[d for d in all_dates if d>=pd.Timestamp('2019-01-01')]

TRAIN_YRS=2;TEST_YRS=1;STEP=1
windows=[]
start_yr=2019
while start_yr+TRAIN_YRS+TEST_YRS<=2027:
    train_start=pd.Timestamp(f'{start_yr}-01-01')
    test_start=pd.Timestamp(f'{start_yr+TRAIN_YRS}-01-01')
    test_end=pd.Timestamp(f'{start_yr+TRAIN_YRS+TEST_YRS}-01-01')
    train_dates=[d for d in all_dates if d>=train_start and d<test_start]
    test_dates=[d for d in all_dates if d>=test_start and d<test_end]
    if len(train_dates)>200 and len(test_dates)>100:
        windows.append((f'{start_yr}-{start_yr+TRAIN_YRS}',train_start,test_start,test_end,train_dates,test_dates))
    start_yr+=STEP

print(f"\n  YH05 滚动回测 (训练{TRAIN_YRS}年 -> 测试{TEST_YRS}年, 步进{STEP}年)")
print(f"  {'─'*72}")
print(f"  {'窗口':<12} {'训练期':<14} {'测试期':<14} {'OOS收益':>10} {'年化':>8} {'夏普':>7} {'MaxDD':>8}")
print(f"  {'─'*72}")

all_oos_navs=[];results=[]
for label,train_start,test_start,test_end,train_dates,test_dates in windows:
    df_m=df_main[(df_main['date']>=train_start)&(df_main['date']<test_end)].reset_index(drop=True)
    raw_sub={}
    for n in raw:
        raw_sub[n]=raw[n][(raw[n]['date']>=train_start)&(raw[n]['date']<test_end)].reset_index(drop=True)
    dfs_g_sub={}
    for n in dfs_growth:
        dfs_g_sub[n]=dfs_growth[n][(dfs_growth[n]['date']>=train_start)&(dfs_growth[n]['date']<test_end)].reset_index(drop=True)

    oos=backtest_on_dates(test_dates,raw_sub,df_m,dfs_g_sub)
    ndf=pd.DataFrame(oos)
    if len(ndf)<10:continue
    final=ndf['nav'].iloc[-1];ret=(final/INIT-1)*100
    ann=((1+ret/100)**(252/len(ndf))-1)*100
    dr=ndf['nav'].pct_change().dropna()
    sr=(ann/100-0.02)/(dr.std()*np.sqrt(252))if dr.std()>0 else 0
    mdd=((ndf['nav']/INIT-(ndf['nav']/INIT).cummax())/(ndf['nav']/INIT).cummax()).min()*100
    tr=train_dates[0].strftime('%Y-%m');te=f'{test_start.year}-{test_end.year}'
    print(f"  {label:<12} {tr:<14} {te:<14} {ret:>+9.1f}% {ann:>+7.1f}% {sr:>+6.2f} {mdd:>+7.1f}%")
    results.append({'window':label,'ret':ret,'ann':ann,'sr':sr,'mdd':mdd})
    ndf['window']=label;all_oos_navs.append(ndf)

rets=[r['ret']for r in results];anns=[r['ann']for r in results]
srs=[r['sr']for r in results];mdds=[r['mdd']for r in results]
print(f"  {'─'*72}")
print(f"  {'平均':<12} {'':14} {'':14} {np.mean(rets):>+9.1f}% {np.mean(anns):>+7.1f}% {np.mean(srs):>+6.2f} {np.mean(mdds):>+7.1f}%")
print(f"  {'最差':<12} {'':14} {'':14} {np.min(rets):>+9.1f}% {np.min(anns):>+7.1f}% {np.min(srs):>+6.2f} {np.max(mdds):>+7.1f}%")
win_rate=sum(1 for r in rets if r>0)/len(rets)*100
print(f"  胜率: {win_rate:.0f}% ({sum(1 for r in rets if r>0)}/{len(rets)}窗口盈利)")

if len(all_oos_navs)>0:
    fig,axes=plt.subplots(1,2,figsize=(18,6),facecolor='white')
    ax=axes[0]
    cols=plt.cm.tab10(np.linspace(0,1,len(all_oos_navs)))
    for i,ndf in enumerate(all_oos_navs):
        n2=ndf['nav']/ndf['nav'].iloc[0]
        ax.plot(range(len(n2)),n2,color=cols[i],lw=1.8,label=f"{results[i]['window']} ({results[i]['ret']:+.0f}%)")
    ax.axhline(y=1,color='#888',lw=0.8,ls='--');ax.legend(fontsize=8,loc='upper left');ax.grid(True,alpha=0.12)
    ax.set_title(f'YH05 Walk-Forward OOS净值 (均值{np.mean(rets):+.1f}%)',fontsize=13,fontweight='bold')
    ax=axes[1]
    colors2=['#CC2222'if r>=0 else'#228B22'for r in rets]
    bars=ax.bar(range(len(rets)),rets,color=colors2,alpha=0.85,edgecolor='white',lw=1)
    for bar,val in zip(bars,rets):
        off=1.5 if val>=0 else-3.5
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+off,f'{val:+.1f}%',ha='center',fontsize=12,fontweight='bold',color='#CC2222'if val>=0 else'#228B22')
    ax.axhline(y=0,color='black',lw=1)
    ax.set_xticks(range(len(rets)));ax.set_xticklabels([r['window']for r in results],fontsize=10)
    ax.set_title('各窗口OOS收益',fontsize=13,fontweight='bold');ax.grid(True,alpha=0.12,axis='y')
    plt.savefig('walkforward_chart.png',dpi=150,bbox_inches='tight',facecolor='white');plt.close()
    print(f'  图表: walkforward_chart.png')
