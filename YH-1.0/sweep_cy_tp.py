# -*- coding: utf-8 -*-
"""扫描创业板最优止盈点 — 一次取数, 多次回测"""
import sys,io,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak,pandas as pd,numpy as np
warnings.filterwarnings('ignore')

CORE_STOCKS={'山东高速':'sh600350','渝农商行':'sh601077','皖通高速':'sh600012','江苏银行':'sh600919'}
ETF_STOCKS={'创业板':'sz159915'}
ALL_STOCKS={**CORE_STOCKS,**ETF_STOCKS}
INIT=1_000_000;COMM=0.0003;SLIP=0.0001;MAX_POS=0.25
TRAIL_STOP=0.07;HARD_STOP=0.10;COOLDOWN=20
CORE_PARAMS={
    '山东高速':{'rsi':45,'bb':0.25,'tp':0.15,'tp_hi':0.20},
    '渝农商行':{'rsi':30,'bb':0.10,'tp':0.20,'tp_hi':0.25},
    '皖通高速':{'rsi':38,'bb':0.12,'tp':0.20,'tp_hi':0.25},
    '江苏银行':{'rsi':35,'bb':0.10,'tp':0.15,'tp_hi':0.20},
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
    bp=CORE_PARAMS.get(name,{'rsi':42,'bb':0.25})
    sc=(1 if rsi<=bp['rsi'] else 0)+(1 if dist<=bp['bb'] else 0)
    if rsi<=30:sc+=1
    return sc>=1,sc

def run(raw, dfs, dates, cy_tp):
    cy_tp_hi=cy_tp+0.05
    BUY_PARAMS={**CORE_PARAMS,'创业板':{'tp':cy_tp,'tp_hi':cy_tp_hi}}
    cash=INIT;shares={n:0.0 for n in ALL_STOCKS};entry={n:0.0 for n in ALL_STOCKS};high={n:0.0 for n in ALL_STOCKS}
    accel={n:False for n in ALL_STOCKS};cooldown={n:0 for n in ALL_STOCKS}
    sold_today={n:False for n in ALL_STOCKS}
    navs=[];cy_trades=[]

    for date in dates:
        for n in ALL_STOCKS:sold_today[n]=False
        px={n:raw[n][raw[n]['date']==date]['close'].iloc[0]for n in ALL_STOCKS if len(raw[n][raw[n]['date']==date])>0}

        # sell
        for n in ALL_STOCKS:
            if shares[n]<=0:continue
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            if cp>high[n]:high[n]=cp
            pnl=cp/entry[n]-1;dd=cp/high[n]-1
            tp=BUY_PARAMS[n]['tp'];tp_hi=BUY_PARAMS[n]['tp_hi']
            do=False;sell_px=cp;why=''
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
                pnl_r=(sell_px/entry[n]-1)*100
                t={'date':date,'name':n,'dir':'SELL','price':sell_px,'pnl':pnl_r,'why':why}
                if n=='创业板':cy_trades.append(t)
                cash+=shares[n]*sell_px*(1-COMM-SLIP)
                shares[n]=0;entry[n]=0;high[n]=0;accel[n]=False;sold_today[n]=True
                if sell_px/entry[n]-1<=-HARD_STOP:cooldown[n]=COOLDOWN

        nav=cash+sum(shares[n]*px.get(n,0)for n in ALL_STOCKS)
        for n in ALL_STOCKS:
            if cooldown[n]>0:cooldown[n]-=1

        # buy core
        for n in CORE_STOCKS:
            if sold_today[n]:continue
            if shares[n]>0:continue
            if cooldown[n]>0:continue
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            ok,sc=check_buy(r.iloc[0],n)
            if not ok:continue
            target_val=nav*MAX_POS
            if cash<target_val:
                cy='创业板';cy_px=px.get(cy,0)
                if shares.get(cy,0)>0 and cy_px>0:
                    need=target_val-cash
                    if need>=5000:
                        cy_val=shares[cy]*cy_px;cy_qty_before=shares[cy]
                        sell_val=min(need,cy_val);sell_qty=sell_val/cy_px
                        sell_cost=sell_val*(COMM+SLIP)
                        shares[cy]-=sell_qty;cash+=sell_val-sell_cost
                        pnl_r=(cy_px/entry[cy]-1)*100
                        t={'date':date,'name':cy,'dir':'SELL','price':cy_px,'pnl':pnl_r,'why':f'换仓→{n}'}
                        cy_trades.append(t)
                        if shares[cy]<1e-8:shares[cy]=0;entry[cy]=0;high[cy]=0;accel[cy]=False
            val=min(cash,target_val)
            if val>5000:
                qty=val/cp*(1-COMM-SLIP);shares[n]=qty;cash-=val
                entry[n]=cp;high[n]=cp

        # buy 创业板
        n='创业板';core_held=sum(1 for nn in CORE_STOCKS if shares[nn]>0)
        if core_held<2 and shares[n]<=0 and cooldown[n]<=0:
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp>0 and len(r)>0:
                row=r.iloc[0]
                dif=row.get('macd_dif');dea=row.get('macd_dea');hist=row.get('macd_hist')
                if (not pd.isna(dif))and(not pd.isna(dea))and dif>dea:
                    hist_recent=dfs[n]['macd_hist'].iloc[-40:].dropna()
                    max_hist=hist_recent.abs().max()if len(hist_recent)>10 else 0
                    strength=abs(hist)/max_hist if max_hist>0 else 0
                    pos_frac=0.50 if strength>0.3 else 0.25
                    val=min(cash,nav*pos_frac)
                    if val>5000:
                        qty=val/cp*(1-COMM-SLIP);shares[n]=qty;cash-=val
                        entry[n]=cp;high[n]=cp
        navs.append(cash+sum(shares[n]*px.get(n,0)for n in ALL_STOCKS))

    ndf=pd.DataFrame(navs,columns=['nav']);final=ndf['nav'].iloc[-1]
    ret=(final/INIT-1)*100;ann=((1+ret/100)**(252/len(ndf))-1)*100
    dr=ndf['nav'].pct_change().dropna();vol=dr.std()*np.sqrt(252)*100
    sr=(ann-2)/vol if vol>0 else 0
    mdd=((ndf['nav']-ndf['nav'].cummax())/ndf['nav'].cummax()).min()*100

    cy_sells=[t for t in cy_trades if t['dir']=='SELL']
    pnls=[t['pnl']for t in cy_sells]if cy_sells else[]
    wr=sum(1 for p in pnls if p>0)/len(pnls)*100 if pnls else 0
    avg_pnl=np.mean(pnls)if pnls else 0
    return ann,sr,mdd,wr,len(cy_sells),avg_pnl,ret

print("获取数据...")
raw=fetch();dfs={n:add_indicators(d)for n,d in raw.items()}
dates=sorted(set.intersection(*[set(d['date'])for d in dfs.values()]))
dates=[d for d in dates if d>=pd.Timestamp('2019-01-01')]
print(f"数据: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}, {len(dates)}天")

print(f"\n{'='*70}")
print("  扫描创业板止盈点 (HI=TP+5%, 含换仓)")
print(f"{'='*70}")
print(f"{'TP':>5} {'年化':>8} {'夏普':>6} {'回撤':>7} {'创胜率':>6} {'创笔':>4} {'创均盈':>7} {'累计':>8}")
print('─'*65)

best_ann=-999;best_cfg=None
for tp in [10,12,14,15,16,18,20,22,24,25]:
    tp_val=tp/100.0
    ann,sr,mdd,cy_wr,cy_n,cy_avg,ret=run(raw,dfs,dates,tp_val)
    mark=' ←' if ann>best_ann else ''
    if ann>best_ann:best_ann=ann;best_cfg=(tp,ann,sr,mdd,cy_wr,cy_n,cy_avg,ret)
    print(f" TP={tp:>2}%  {ann:>+7.1f}% {sr:>5.2f}  {mdd:>+6.1f}% {cy_wr:>5.0f}% {cy_n:>4} {cy_avg:>+7.1f}% {ret:>+7.1f}%{mark}")

print(f"\n→ 创业板最优: TP={best_cfg[0]}% HI={best_cfg[0]+5}%  年化{best_cfg[1]:+.1f}%  夏普{best_cfg[2]:.2f}  创均盈{best_cfg[6]:+.1f}%")
