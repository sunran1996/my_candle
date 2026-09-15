# -*- coding: utf-8 -*-
"""YH05 滑动窗口回测: 4年窗口, 每2个月滚动"""
import sys,io,os,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak,pandas as pd,numpy as np
warnings.filterwarnings('ignore')

INIT=1_000_000;RESERVE=100_000;COMM=0.0003;SLIP=0.0001
BB_P=45;BB_S=2.0;RSI_P=14;RSI_L=30;RSI_H=70;ERS=65;BA=0.001
HARD_STOP=0.12;NAV_STOP=0.13;TRAIL=0.10;MOM=10;REBAL=5

MAIN_SYM='sh512890';MAIN_NAME='红利低波'
GROWTH={'创业板':'sz159915'}

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
    df['mom20']=df['close']/df['close'].shift(20)-1;df['ma200']=df['close'].rolling(200).mean()
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
    else:buy_sig=(bb_buy or rsi_buy);sell_sig=(bb_sell or rsi>=RSI_H)
    if buy_sig:return'BUY',nbw
    elif sell_sig:return'SELL',nbw
    else:return'HOLD',nbw

def add_growth(df):
    df=df.copy();df['mom']=df['close']/df['close'].shift(MOM)-1
    e10=df['close'].ewm(span=10,adjust=False).mean();e20=df['close'].ewm(span=20,adjust=False).mean()
    df['macd']=e10-e20;df['macd_s']=df['macd'].ewm(span=7,adjust=False).mean()
    df['macd_h']=df['macd']-df['macd_s'];df['ma20']=df['close'].rolling(20).mean()
    df['macd_line']=df['macd'];df['ma200']=df['close'].rolling(200).mean()
    df['ma10']=df['close'].rolling(10).mean();df['mom20']=df['close']/df['close'].shift(20)-1
    d=df['close'].diff();g=d.clip(lower=0);l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/14,adjust=False).mean()/l.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan))
    return df

def rank_growth(dfs,idx):
    s={}
    for n in GROWTH:
        pos=idx if idx>=0 else len(dfs[n])+idx
        if pos<30 or pos>=len(dfs[n]):continue
        h=dfs[n]['macd_h'].iloc[pos]
        if pd.isna(h):continue
        s[n]=h
    return sorted(s,key=s.get,reverse=True)

def run_window(raw,df_main,dfs_growth,start,end):
    dates=sorted(set.intersection(*[set(d['date'])for d in raw.values()]))
    dates=[d for d in dates if d>=start and d<=end]
    if len(dates)<60:return None

    ALL=list(GROWTH.keys())+[MAIN_NAME]
    start_px=raw[MAIN_NAME][raw[MAIN_NAME]['date']>=start]['close'].iloc[0]

    cash=0.0;rp=RESERVE;ra=False;rs=0.0
    shares={n:0.0 for n in ALL}
    shares[MAIN_NAME]=INIT/start_px*(1-COMM-SLIP)
    position=MAIN_NAME;peak=INIT;navs=[];trades=[]
    sc=2;pbw=None;ep=start_px;hse=0;last_rb=None;stopped=False;partial_done=False
    growth_ban=0

    for date in dates:
        px={}
        for n in ALL:
            r=raw[n][raw[n]['date']==date]
            if len(r):px[n]=r['close'].iloc[0]

        nav=cash+sum(shares[n]*px.get(n,0)for n in ALL)
        if nav>peak:peak=nav
        dd_nav=(nav-peak)/peak if peak>0 else 0

        if dd_nav<-NAV_STOP and any(shares[n]>0 for n in ALL):
            for n in ALL:
                if shares[n]>0 and n in px: cash+=shares[n]*px[n]*(1-COMM-SLIP);shares[n]=0.0
            trades.append({'date':date,'dir':'PANIC','name':'ALL','price':0})
            peak=nav;position=None;stopped=True;growth_ban=2
            navs.append({'date':date,'nav':nav,'pos':position});continue

        mr=df_main[df_main['date']==date]
        if len(mr)==0:navs.append({'date':date,'nav':nav,'pos':position});continue
        sig,pbw=main_signal(mr.iloc[0],pbw)
        mp=px.get(MAIN_NAME,0)

        if position==MAIN_NAME and mp>0 and ep>0 and mp<ep*(1-HARD_STOP):
            cash+=shares[MAIN_NAME]*mp*(1-COMM-SLIP);shares[MAIN_NAME]=0.0
            trades.append({'date':date,'dir':'STOP','name':MAIN_NAME,'price':mp});position=None

        # BUY → 重回红利
        if sig=='BUY' and position!=MAIN_NAME:
            if position:
                for n in GROWTH:
                    if shares[n]>0 and n in px: cash+=shares[n]*px[n]*(1-COMM-SLIP);shares[n]=0.0
                trades.append({'date':date,'dir':'SELL','name':position,'price':px.get(position,0)})
            if mp>0:
                val=min(cash,nav);bs=val/mp*(1-COMM-SLIP)
                if ra:rs+=bs*(RESERVE/(INIT+RESERVE));shares[MAIN_NAME]+=bs-bs*(RESERVE/(INIT+RESERVE))
                else:shares[MAIN_NAME]+=bs
                cash-=val
                trades.append({'date':date,'dir':'BUY','name':MAIN_NAME,'price':mp})
                position=MAIN_NAME;ep=mp;stopped=False;growth_ban=max(0,growth_ban-1)

        # SELL → 清红利
        elif sig=='SELL' and position==MAIN_NAME:
            if shares[MAIN_NAME]>0 and mp>0:
                cash+=shares[MAIN_NAME]*mp*(1-COMM-SLIP)
                if ra:rp=rs*mp*(1-COMM-SLIP);cash-=rp;rs=0;ra=False
                shares[MAIN_NAME]=0.0
                trades.append({'date':date,'dir':'SELL','name':MAIN_NAME,'price':mp});position=None

        # HOLD + 空仓 → 成长轮动
        elif sig=='HOLD' and position!=MAIN_NAME:
            if position and position in px:
                if px[position]>hse:hse=px[position]
                if px[position]<hse*(1-TRAIL):
                    sell_cash=shares[position]*px[position]*(1-COMM-SLIP)
                    cash+=sell_cash;shares[position]=0.0
                    trades.append({'date':date,'dir':'STOP','name':position,'price':px[position]})
                    if shares[MAIN_NAME]>0 and mp>0:
                        bs2=cash/mp*(1-COMM-SLIP);shares[MAIN_NAME]+=bs2;cash=0
                        trades.append({'date':date,'dir':'ADD','name':MAIN_NAME,'price':mp})
                    position=None;stopped=True;partial_done=False;growth_ban=2

            if stopped or shares[MAIN_NAME]>0:navs.append({'date':date,'nav':nav,'pos':position});continue
            if growth_ban>0:navs.append({'date':date,'nav':nav,'pos':position});continue
            bm_idx=None
            for n in GROWTH:
                idxs=dfs_growth[n][dfs_growth[n]['date']==date].index
                if len(idxs):bm_idx=idxs[0];break
            if bm_idx is None:navs.append({'date':date,'nav':nav,'pos':position});continue

            days=(date-last_rb).days if last_rb else 999
            if days>=REBAL and position is None:
                gn=list(GROWTH.keys())[0]
                gdf=dfs_growth[gn];gr=gdf[gdf['date']==date]
                if len(gr)==0:navs.append({'date':date,'nav':nav,'pos':position});continue
                g_close=gr['close'].values[0];g_ma200=gr['ma200'].values[0]
                g_rsi=gr['rsi'].values[0];g_ma10=gr['ma10'].values[0];g_mom20=gr['mom20'].values[0]
                m_mom20=df_main[df_main['date']==date]['mom20'].values[0]if len(df_main[df_main['date']==date])>0 else 0

                if not pd.isna(g_ma200)and g_close<g_ma200:navs.append({'date':date,'nav':nav,'pos':position});continue
                if not pd.isna(m_mom20)and not pd.isna(g_mom20):
                    if m_mom20<0 and g_mom20<0:navs.append({'date':date,'nav':nav,'pos':position});continue

                ranking=rank_growth(dfs_growth,bm_idx)
                bull_ok=g_close>gr['ma20'].values[0]
                macd_ok=gr['macd_h'].values[0]>0
                if ranking and macd_ok and bull_ok:
                    target=ranking[0]
                    if target in px:
                        if not pd.isna(g_rsi):
                            if g_rsi>55:navs.append({'date':date,'nav':nav,'pos':position});continue
                            if g_rsi<25:navs.append({'date':date,'nav':nav,'pos':position});continue
                        if not pd.isna(g_ma10)and g_ma10>0:
                            if g_close>g_ma10*1.05:navs.append({'date':date,'nav':nav,'pos':position});continue
                        above_zero=gr['macd_line'].values[0]>0
                        pos_pct=1.0 if above_zero else 0.3
                        val=min(cash,nav*pos_pct)
                        if val>100:
                            shares[target]=val/px[target]*(1-COMM-SLIP);cash-=val
                            trades.append({'date':date,'dir':'BUY','name':target,'price':px[target]})
                            position=target;hse=px[target]
                    last_rb=date

        nav=cash+sum(shares[n]*px.get(n,0)for n in ALL)
        navs.append({'date':date,'nav':nav,'pos':position})

    ndf=pd.DataFrame(navs)
    if len(ndf)<60:return None
    final=ndf['nav'].iloc[-1]
    ret=(final/INIT-1)*100
    ann=((1+ret/100)**(252/len(ndf))-1)*100
    dr=ndf['nav'].pct_change().dropna()
    sr=(ann/100-0.02)/(dr.std()*np.sqrt(252))if dr.std()>0 else 0
    mdd=((ndf['nav']/INIT-(ndf['nav']/INIT).cummax())/(ndf['nav']/INIT).cummax()).min()*100
    td=pd.DataFrame(trades)
    n_hl=len(td[td['name']==MAIN_NAME])if len(td)>0 else 0
    n_cy=len(td[td['name']=='创业板'])if len(td)>0 else 0

    # 年度收益
    ndf['year']=ndf['date'].dt.year
    yy={}
    for yr,grp in ndf.groupby('year'):
        if len(grp)<10:continue
        yr_ret=(grp['nav'].iloc[-1]/grp['nav'].iloc[0]-1)*100
        yy[yr]=yr_ret
    return{'start':start.strftime('%Y-%m-%d'),'end':end.strftime('%Y-%m-%d'),
           'ret':ret,'ann':ann,'sharpe':sr,'mdd':mdd,'trades':len(td),
           'n_hl':n_hl,'n_cy':n_cy,'years':yy}

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

    all_years=sorted(set(y for r in results for y in r['years']))

    print(f"\n{'='*110}")
    print(f"  YH05 4年滑动窗口回测 (每2个月滚动)  共{len(results)}个窗口")
    print(f"{'='*110}")
    header=f"  {'窗口':<24} {'收益':>8} {'年化':>7} {'夏普':>6} {'回撤':>7} {'交易':>5}"
    for y in all_years:header+=f" {y:>7}"
    print(header)
    print(f"  {'-'*108}")
    for r in results:
        line=f"  {r['start']}~{r['end']}  {r['ret']:>+7.1f}% {r['ann']:>+6.1f}% {r['sharpe']:>5.2f} {r['mdd']:>+6.1f}% {r['trades']:>4d}"
        for y in all_years:line+=f" {r['years'].get(y,float('nan')):>+7.1f}%"if y in r['years']else f" {'—':>7}"
        print(line)

    rets=[r['ret']for r in results]
    print(f"  {'-'*108}")
    print(f"  {'均值':<24} {np.mean(rets):>+7.1f}% {np.mean([r['ann']for r in results]):>+6.1f}% {np.mean([r['sharpe']for r in results]):>5.2f} {np.mean([r['mdd']for r in results]):>+6.1f}% {np.mean([r['trades']for r in results]):>4.0f}")
    print(f"  {'最差':<24} {np.min(rets):>+7.1f}% {np.min([r['ann']for r in results]):>+6.1f}% {np.min([r['sharpe']for r in results]):>5.2f} {np.min([r['mdd']for r in results]):>+6.1f}%")
    print(f"  {'最好':<24} {np.max(rets):>+7.1f}% {np.max([r['ann']for r in results]):>+6.1f}% {np.max([r['sharpe']for r in results]):>5.2f} {np.max([r['mdd']for r in results]):>+6.1f}%")
    print(f"  {'正收益窗口':<24} {sum(1 for r in rets if r>0)}/{len(rets)}")

if __name__=='__main__':main()
