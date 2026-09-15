# -*- coding: utf-8 -*-
"""滑动窗口回测: 4年窗口, 每2个月滚动一次"""
import sys,io,os,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak,pandas as pd,numpy as np
warnings.filterwarnings('ignore')

INIT=1_000_000;COMM=0.0003;SLIP=0.0001
BB_P=45;BB_S=2.0;RSI_P=14;RSI_L=30;RSI_H=70;ERS=65;BA=0.001
HARD_STOP=0.12;NAV_STOP=0.13;TRAIL=0.10;MOM=10

MAIN_SYM='sh512890';MAIN_NAME='红利低波'
GROWTH={'创业板':'sz159915','科创50':'sh588000'}
ALL_NAMES=[MAIN_NAME]+list(GROWTH.keys());ALL_SYMS={MAIN_NAME:MAIN_SYM,**GROWTH}

def fetch():
    dfs={}
    for n,s in ALL_SYMS.items():
        df=ak.fund_etf_hist_sina(symbol=s);df['date']=pd.to_datetime(df['date'])
        dfs[n]=df[['date','close']].sort_values('date').reset_index(drop=True)
    return dfs

def add_main(df):
    df=df.copy();r=df['close'].pct_change().fillna(0);r[abs(r)>0.1]=0;df['adj']=(1+r).cumprod()
    df['ma']=df['adj'].rolling(BB_P).mean();df['std']=df['adj'].rolling(BB_P).std()
    df['up']=df['ma']+BB_S*df['std'];df['lo']=df['ma']-BB_S*df['std']
    df['ua']=df['up'].diff().diff().rolling(3,min_periods=1).mean()
    df['pa']=df['adj'].diff().diff().rolling(3,min_periods=1).mean()
    d=df['adj'].diff();g=d.clip(lower=0);l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/RSI_P,adjust=False).mean()/l.ewm(alpha=1/RSI_P,adjust=False).mean().replace(0,np.nan))
    df['mom20']=df['close']/df['close'].shift(20)-1;return df

def add_growth(df):
    df=df.copy();df['mom']=df['close']/df['close'].shift(MOM)-1
    e10=df['close'].ewm(span=10,adjust=False).mean();e20=df['close'].ewm(span=20,adjust=False).mean()
    df['macd']=e10-e20;df['macd_s']=df['macd'].ewm(span=7,adjust=False).mean()
    df['macd_h']=df['macd']-df['macd_s'];df['ma20']=df['close'].rolling(20).mean();df['macd_line']=df['macd']
    df['ma120']=df['close'].rolling(120).mean();df['ma10']=df['close'].rolling(10).mean();df['mom20']=df['close']/df['close'].shift(20)-1
    d=df['close'].diff();g=d.clip(lower=0);l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/14,adjust=False).mean()/l.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan))
    return df

def rank_momentum(dfs,date):
    scores=[]
    for n in GROWTH:
        idxs=dfs[n][dfs[n]['date']==date].index
        if len(idxs)==0:continue
        v=dfs[n]['mom'].iloc[idxs[0]]
        if not pd.isna(v):scores.append((n,v))
    scores.sort(key=lambda x:x[1],reverse=True)
    return scores

def run_window(raw,df_main,dfs_g,start,end):
    dates=sorted(set.intersection(*[set(d['date'])for d in raw.values()]))
    dates=[d for d in dates if d>=start and d<=end]
    if len(dates)<60:return None

    cash=INIT;shares={n:0.0 for n in ALL_NAMES}
    pos=None;peak=INIT;navs=[];trades=[]
    sc=2;pbw=None;ep=0;hse={};lrb=None;stp=False;gb=0;started=False

    for date in dates:
        px={}
        for n in ALL_NAMES:r=raw[n][raw[n]['date']==date];px[n]=r['close'].iloc[0]if len(r)>0 else 0

        nav=cash+sum(shares[n]*px.get(n,0)for n in ALL_NAMES)
        if nav>peak:peak=nav
        dd_nav=(nav-peak)/peak if peak>0 else 0

        mr=df_main[df_main['date']==date]
        if len(mr)==0:navs.append({'date':date,'nav':nav,'pos':pos});continue
        row=mr.iloc[0]
        adj,rsi,up,lo=row['adj'],row['rsi'],row['up'],row['lo']
        if pd.isna(lo)or pd.isna(rsi):navs.append({'date':date,'nav':nav,'pos':pos});continue
        bw=(up-lo)/row['ma']if row['ma']>0 else 0.1
        ex=(pbw is not None and bw>pbw);nbw=bw
        bb_buy=(adj<=lo);bb_sell=(adj>=up);rsi_buy=(rsi<=RSI_L)
        if ex:raw_sell=(bb_sell and rsi>=ERS);ua=row['ua']if not pd.isna(row['ua'])else 0;pa=row['pa']if not pd.isna(row['pa'])else 0;sell_sig=raw_sell and not((ua>BA)and(pa>0));buy_sig=(bb_buy or rsi_buy)
        else:buy_sig=(bb_buy or rsi_buy);sell_sig=(bb_sell or rsi>=RSI_H)
        sig='BUY'if buy_sig else('SELL'if sell_sig else'HOLD');pbw=nbw;mp=px[MAIN_NAME]

        if not started:
            if buy_sig:shares[MAIN_NAME]=cash/mp*(1-COMM-SLIP);cash=0;pos=MAIN_NAME;ep=mp;started=True;stp=False;gb=max(0,gb-1);trades.append({'date':date,'dir':'BUY','name':MAIN_NAME,'price':mp})
            elif sig=='HOLD':
                mmm=df_main[df_main['date']==date]['mom20'].values[0]if len(df_main[df_main['date']==date])>0 else 0
                ranking=rank_momentum(dfs_g,date)
                for tg,_ in ranking:
                    if tg not in px or not px[tg]:continue
                    gdf=dfs_g[tg];gr=gdf[gdf['date']==date]
                    if len(gr)==0:continue
                    gc=gr['close'].values[0];gma=gr['ma120'].values[0];grs=gr['rsi'].values[0];gm10=gr['ma10'].values[0];gmm=gr['mom20'].values[0]
                    mh=gr['macd_h'].values[0];ml=gr['macd_line'].values[0];m20=gr['ma20'].values[0]
                    ok=mh>0
                    if ok and not pd.isna(gma)and gc<gma:ok=False
                    if ok and not pd.isna(grs):
                        if grs>55:ok=False
                        if grs<25:ok=False
                    if ok and not pd.isna(gm10)and gm10>0:
                        if gc>gm10*1.05:ok=False
                    if ok:
                        az=ml>0;pp=1.0 if az else 0.3;val=min(cash,nav*pp)
                        if val>100:shares[tg]=val/px[tg]*(1-COMM-SLIP);cash-=val;pos=tg;hse[tg]=px[tg];started=True;lrb=date;trades.append({'date':date,'dir':'BUY','name':tg,'price':px[tg]})
                    break
            navs.append({'date':date,'nav':nav,'pos':pos});continue

        # === Normal ===
        if dd_nav<-NAV_STOP and any(shares[n]>0 for n in ALL_NAMES):
            for n in ALL_NAMES:
                if shares[n]>0 and px.get(n,0)>0:cash+=shares[n]*px[n]*(1-COMM-SLIP);shares[n]=0.0
            peak=nav;pos=None;stp=True;gb=2;hse.clear()
            trades.append({'date':date,'dir':'PANIC','name':'ALL','price':0})
            for gn in GROWTH:
                if px.get(gn,0)>0:trades.append({'date':date,'dir':'PANIC','name':gn,'price':px[gn]})
            navs.append({'date':date,'nav':nav,'pos':pos});continue
        if pos==MAIN_NAME and mp>0 and ep>0 and mp<ep*(1-HARD_STOP):cash+=shares[MAIN_NAME]*mp*(1-COMM-SLIP);shares[MAIN_NAME]=0.0;pos=None;trades.append({'date':date,'dir':'STOP','name':MAIN_NAME,'price':mp})
        if buy_sig and pos!=MAIN_NAME:
            for gn in GROWTH:
                if shares.get(gn,0)>0 and px.get(gn,0)>0:cash+=shares[gn]*px[gn]*(1-COMM-SLIP);shares[gn]=0.0;trades.append({'date':date,'dir':'SELL','name':gn,'price':px[gn]})
            hse.clear()
            if mp>0:val=min(cash,nav);shares[MAIN_NAME]+=val/mp*(1-COMM-SLIP);cash-=val;pos=MAIN_NAME;ep=mp;stp=False;gb=max(0,gb-1);trades.append({'date':date,'dir':'BUY','name':MAIN_NAME,'price':mp})
            navs.append({'date':date,'nav':cash+sum(shares[n]*px.get(n,0)for n in ALL_NAMES),'pos':pos});continue
        if buy_sig or sell_sig:sc+=1
        if sc<2:navs.append({'date':date,'nav':nav,'pos':pos});continue
        if sell_sig and pos==MAIN_NAME:
            if shares[MAIN_NAME]>0 and mp>0:cash+=shares[MAIN_NAME]*mp*(1-COMM-SLIP);shares[MAIN_NAME]=0.0;pos=None;trades.append({'date':date,'dir':'SELL','name':MAIN_NAME,'price':mp})
        elif sig=='HOLD'and pos!=MAIN_NAME:
            for gn in GROWTH:
                if shares.get(gn,0)<=0:continue
                p=px.get(gn,0)
                if p<=0:continue
                if p>hse.get(gn,0):hse[gn]=p
                if p<hse.get(gn,0)*(1-TRAIL):
                    cash+=shares[gn]*p*(1-COMM-SLIP);shares[gn]=0.0
                    trades.append({'date':date,'dir':'STOP','name':gn,'price':p})
                    if gn in hse:del hse[gn];stp=True;gb=2
            if not any(shares.get(gn,0)>0 for gn in GROWTH):pos=None
            if shares.get(MAIN_NAME,0)>0:navs.append({'date':date,'nav':nav,'pos':pos});continue
            if gb>0:navs.append({'date':date,'nav':nav,'pos':pos});continue
            if pos is None or pos in GROWTH:
                mmm=df_main[df_main['date']==date]['mom20'].values[0]if len(df_main[df_main['date']==date])>0 else 0
                ranking=rank_momentum(dfs_g,date)
                for tg,_ in ranking:
                    if tg not in px or not px[tg]:continue
                    gdf=dfs_g[tg];gr=gdf[gdf['date']==date]
                    if len(gr)==0:continue
                    gc=gr['close'].values[0];gma=gr['ma120'].values[0];grs=gr['rsi'].values[0];gm10=gr['ma10'].values[0];gmm=gr['mom20'].values[0]
                    mh=gr['macd_h'].values[0];ml=gr['macd_line'].values[0];m20=gr['ma20'].values[0]
                    ok=mh>0
                    if ok and not pd.isna(gma)and gc<gma:ok=False
                    if ok and not pd.isna(grs):
                        if grs>55:ok=False
                        if grs<25:ok=False
                    if ok and not pd.isna(gm10)and gm10>0:
                        if gc>gm10*1.05:ok=False
                    if ok:
                        if pos is None:
                            az=ml>0;pp=1.0 if az else 0.3;val=min(cash,nav*pp)
                            if val>100:shares[tg]=val/px[tg]*(1-COMM-SLIP);cash-=val;pos=tg;hse[tg]=px[tg];lrb=date;trades.append({'date':date,'dir':'BUY','name':tg,'price':px[tg]})
                        elif shares.get(tg,0)>0:
                            pass
                        else:
                            held=[gn for gn in GROWTH if shares.get(gn,0)>0]
                            if held:
                                src=max(held,key=lambda gn:shares[gn]*px.get(gn,0))
                                sv=shares[src]*px[src]*0.5
                                cash+=sv*(1-COMM-SLIP);shares[src]*=0.5
                                trades.append({'date':date,'dir':'SELL','name':src,'price':px[src],'note':'半仓轮换'})
                                az=ml>0;pp=1.0 if az else 0.3;val=min(cash,nav*pp)
                                if val>100:
                                    shares[tg]=shares.get(tg,0)+val/px[tg]*(1-COMM-SLIP)
                                    cash-=val;hse[tg]=px[tg]
                                    trades.append({'date':date,'dir':'BUY','name':tg,'price':px[tg]})
                                if shares.get(pos,0)<=0:
                                    alive=[gn for gn in GROWTH if shares.get(gn,0)>0]
                                    pos=alive[0]if alive else None
                        break
        nav=cash+sum(shares[n]*px.get(n,0)for n in ALL_NAMES);navs.append({'date':date,'nav':nav,'pos':pos})

    ndf=pd.DataFrame(navs)
    if len(ndf)<60:return None
    final=ndf['nav'].iloc[-1]
    ret=(final/INIT-1)*100
    ann=((1+ret/100)**(252/len(ndf))-1)*100
    dr=ndf['nav'].pct_change().dropna()
    sr=(ann/100-0.02)/(dr.std()*np.sqrt(252))if dr.std()>0 else 0
    mdd=((ndf['nav']/INIT-(ndf['nav']/INIT).cummax())/(ndf['nav']/INIT).cummax()).min()*100
    td=pd.DataFrame(trades)
    return {'start':start.strftime('%Y-%m-%d'),'end':end.strftime('%Y-%m-%d'),
            'ret':ret,'ann':ann,'sharpe':sr,'mdd':mdd,'trades':len(td),
            'n_hl':len(td[td['name']==MAIN_NAME])if len(td)>0 else 0,
            'n_cy':len(td[td['name']=='创业板'])if len(td)>0 else 0,
            'n_kc':len(td[td['name']=='科创50'])if len(td)>0 else 0}

def main():
    print("获取数据...")
    raw=fetch()
    df_main=add_main(raw[MAIN_NAME])
    dfs_g={n:add_growth(d)for n,d in raw.items()if n!=MAIN_NAME}

    all_dates=sorted(set.intersection(*[set(d['date'])for d in raw.values()]))
    t0=all_dates[0];tn=all_dates[-1]
    print(f"数据范围: {t0.strftime('%Y-%m-%d')} ~ {tn.strftime('%Y-%m-%d')}")

    results=[]
    start=t0
    while True:
        end=start+pd.DateOffset(years=4)
        if end>tn:break
        r=run_window(raw,df_main,dfs_g,start,end)
        if r:results.append(r)
        start+=pd.DateOffset(months=2)

    print(f"\n{'='*90}")
    print(f"  4年滑动窗口回测 (每2个月滚动)  共{len(results)}个窗口")
    print(f"{'='*90}")
    print(f"  {'窗口':<24} {'收益':>8} {'年化':>7} {'夏普':>6} {'回撤':>7} {'交易':>5} {'红利':>5} {'创业':>5} {'科创':>5}")
    print(f"  {'-'*80}")
    for r in results:
        print(f"  {r['start']}~{r['end']}  {r['ret']:>+7.1f}% {r['ann']:>+6.1f}% {r['sharpe']:>5.2f} {r['mdd']:>+6.1f}% {r['trades']:>4d}  {r['n_hl']:>4d} {r['n_cy']:>4d} {r['n_kc']:>4d}")

    # 汇总
    rets=[r['ret']for r in results];anns=[r['ann']for r in results]
    sharpes=[r['sharpe']for r in results];mdds=[r['mdd']for r in results]
    trades=[r['trades']for r in results]
    print(f"  {'-'*80}")
    print(f"  {'均值':<24} {np.mean(rets):>+7.1f}% {np.mean(anns):>+6.1f}% {np.mean(sharpes):>5.2f} {np.mean(mdds):>+6.1f}% {np.mean(trades):>4.0f}")
    print(f"  {'中位数':<24} {np.median(rets):>+7.1f}% {np.median(anns):>+6.1f}% {np.median(sharpes):>5.2f} {np.median(mdds):>+6.1f}% {np.median(trades):>4.0f}")
    print(f"  {'最差':<24} {np.min(rets):>+7.1f}% {np.min(anns):>+6.1f}% {np.min(sharpes):>5.2f} {np.min(mdds):>+6.1f}%")
    print(f"  {'最好':<24} {np.max(rets):>+7.1f}% {np.max(anns):>+6.1f}% {np.max(sharpes):>5.2f} {np.max(mdds):>+6.1f}%")
    print(f"  {'正收益窗口':<24} {sum(1 for r in rets if r>0)}/{len(rets)}")
    print(f"  {'回撤<-10%窗口':<24} {sum(1 for m in mdds if m<-10)}/{len(rets)}")

if __name__=='__main__':main()
