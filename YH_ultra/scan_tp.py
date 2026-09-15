# -*- coding: utf-8 -*-
"""每只股票独立扫描最优止盈点位 (取消缩放补仓)"""
import sys,io,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak,pandas as pd,numpy as np
warnings.filterwarnings('ignore')

STOCKS={'山东高速':'sh600350','渝农商行':'sh601077','皖通高速':'sh600012','江苏银行':'sh600919'}
INIT=1_000_000;COMM=0.0003;SLIP=0.0001;MAX_POS=0.25

def fetch():
    dfs={}
    for n,s in STOCKS.items():
        df=ak.stock_zh_a_daily(symbol=s,adjust='qfq');df['date']=pd.to_datetime(df['date'])
        dfs[n]=df[['date','open','high','low','close','volume']].sort_values('date').reset_index(drop=True)
    return dfs

def add_indicators(df):
    df=df.copy();c=df['close']
    df['ma20']=c.rolling(20).mean();df['ma60']=c.rolling(60).mean()
    df['bb_ma']=c.rolling(20).mean();df['bb_std']=c.rolling(20).std()
    df['bb_up']=df['bb_ma']+2*df['bb_std'];df['bb_lo']=df['bb_ma']-2*df['bb_std']
    d=c.diff();g=d.clip(lower=0);l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/14,adjust=False).mean()/l.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan))
    return df

def buy_check(row, strict=False):
    if pd.isna(row['bb_lo'])or pd.isna(row['rsi']):return False,0
    rsi=row['rsi'];c=row['close'];lo=row['bb_lo'];up=row['bb_up']
    if up<=lo:return False,0
    dist=(c-lo)/(up-lo)
    rsi_th=35 if strict else 42;bb_th=0.18 if strict else 0.25
    sc=(1 if rsi<=rsi_th else 0)+(1 if dist<=bb_th else 0)
    if rsi<=30:sc+=1
    return sc>=1,sc

def make_sell(tp):
    def fn(row,pnl,dd,pbw):
        if pnl<=-0.10:return True,''
        if dd<=-0.08:return True,''
        if pnl>=tp:return True,''
        return False,''
    return fn

# ---- 单只股票回测 ----
def run_solo(name,sym,tp):
    raw=fetch();df=add_indicators(raw[name])
    dates=[d for d in df['date'] if d>=pd.Timestamp('2019-01-01')]
    cash=INIT;shares=0;entry=0;high=0
    navs=[];trades=[];sell_fn=make_sell(tp)

    for date in dates:
        r=df[df['date']==date]
        if len(r)==0:continue
        cp=r['close'].iloc[0];row=r.iloc[0]
        # 卖出
        if shares>0:
            if cp>high:high=cp
            pnl=cp/entry-1;dd=cp/high-1
            do,_=sell_fn(row,pnl,dd,None)
            if do:
                cash+=shares*cp*(1-COMM-SLIP)
                trades.append(pnl*100)
                shares=0;entry=0;high=0
        # 买入 (无缩放)
        if shares==0:
            ok,sc=buy_check(row,False)
            if ok:
                val=min(cash,INIT*MAX_POS)
                if val>5000:
                    shares=val/cp*(1-COMM-SLIP);cash-=val
                    entry=cp;high=cp
        navs.append(cash+shares*cp)

    ndf=pd.DataFrame(navs,columns=['nav']);final=ndf['nav'].iloc[-1]
    ret=(final/INIT-1)*100;ann=((1+ret/100)**(252/len(ndf))-1)*100
    dr=ndf['nav'].pct_change().dropna();vol=dr.std()*np.sqrt(252)*100
    sr=(ann-2)/vol if vol>0 else 0
    mdd=((ndf['nav']-ndf['nav'].cummax())/ndf['nav'].cummax()).min()*100
    sells=np.array([t for t in trades if t!=0])
    wr=(sells>0).sum()/len(sells)*100 if len(sells)>0 else 0
    aw=sells[sells>0].mean() if (sells>0).sum()>0 else 0
    al=sells[sells<0].mean() if (sells<0).sum()>0 else 0
    return ann,ret,sr,mdd,wr,len(trades),aw,al

# ---- 扫描 ----
print("单只股票止盈参数扫描 (2019起, 无缩放补仓)")
print("="*80)

best_tps={}
for name,sym in STOCKS.items():
    print(f"\n{'─'*60}")
    print(f"  {name} ({sym})")
    print(f"  {'TP':>6} {'年化':>8} {'夏普':>6} {'回撤':>7} {'胜率':>6} {'交易':>5} {'均盈':>7} {'均亏':>7}")
    best_sr=0;best_tp=0
    for tp in [0.10,0.12,0.15,0.18,0.20,0.22,0.25,0.28,0.30,0.35,0.40]:
        ann,ret,sr,mdd,wr,nt,aw,al=run_solo(name,sym,tp)
        mark=' <' if sr>best_sr else ''
        print(f"  {tp*100:>5.0f}% {ann:>+7.1f}% {sr:>5.2f} {mdd:>+6.1f}% {wr:>5.0f}% {nt:>5} {aw:>+6.1f}% {al:>+6.1f}%{mark}")
        if sr>best_sr:best_sr=sr;best_tp=tp
    best_tps[name]=best_tp
    print(f"  → 最优TP={best_tp*100:.0f}% (SR={best_sr:.2f})")

# ---- 组合回测(各股用各自最优TP) ----
print(f"\n{'='*80}")
print("组合回测: 各股各自最优TP vs 统一20%TP")

def run_portfolio(tp_map):
    raw=fetch();dfs={n:add_indicators(d)for n,d in raw.items()}
    dates=sorted(set.intersection(*[set(d['date'])for d in dfs.values()]))
    dates=[d for d in dates if d>=pd.Timestamp('2019-01-01')]
    cash=INIT;shares={n:0.0 for n in STOCKS};entry={n:0.0 for n in STOCKS};high={n:0.0 for n in STOCKS}
    loss={n:False for n in STOCKS};scale_step={n:0 for n in STOCKS}
    navs=[];trades=[]

    for date in dates:
        for n in STOCKS:
            if scale_step[n]==-1:scale_step[n]=0
        px={n:raw[n][raw[n]['date']==date]['close'].iloc[0]for n in STOCKS if len(raw[n][raw[n]['date']==date])>0}

        for n in STOCKS:
            if shares[n]<=0:continue
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            if cp>high[n]:high[n]=cp
            pnl=cp/entry[n]-1;dd=cp/high[n]-1
            tp=tp_map.get(n,0.20)
            do=False
            if pnl<=-0.10:do=True
            elif dd<=-0.08:do=True
            elif pnl>=tp:do=True
            if do:
                cash+=shares[n]*cp*(1-COMM-SLIP)
                trades.append(pnl*100)
                loss[n]=pnl<0;scale_step[n]=0
                shares[n]=0;entry[n]=0;high[n]=0
                scale_step[n]=-1

        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)

        for n in STOCKS:
            if scale_step[n]==-1:continue
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            if shares[n]>0:continue
            ok,sc=buy_check(r.iloc[0],False)
            if not ok:continue
            val=min(cash,nav*MAX_POS)
            if val>5000:
                qty=val/cp*(1-COMM-SLIP)
                shares[n]=qty;cash-=val
                entry[n]=cp;high[n]=cp

        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)
        navs.append(nav)

    ndf=pd.DataFrame(navs,columns=['nav']);final=ndf['nav'].iloc[-1]
    ret=(final/INIT-1)*100;ann=((1+ret/100)**(252/len(ndf))-1)*100
    dr=ndf['nav'].pct_change().dropna();vol=dr.std()*np.sqrt(252)*100
    sr=(ann-2)/vol if vol>0 else 0
    mdd=((ndf['nav']-ndf['nav'].cummax())/ndf['nav'].cummax()).min()*100
    sells=np.array([t for t in trades if t!=0])
    wr=(sells>0).sum()/len(sells)*100 if len(sells)>0 else 0
    aw=sells[sells>0].mean() if (sells>0).sum()>0 else 0
    al=sells[sells<0].mean() if (sells<0).sum()>0 else 0
    return ann,ret,sr,mdd,wr,len(trades),aw,al

uniform={n:0.20 for n in STOCKS}
ann_u,ret_u,sr_u,mdd_u,wr_u,nt_u,aw_u,al_u=run_portfolio(uniform)
print(f"\n  统一20%TP: 年化{ann_u:+.1f}% 夏普{sr_u:.2f} 回撤{mdd_u:+.1f}% 胜率{wr_u:.0f}% 交易{nt_u}")

ann_o,ret_o,sr_o,mdd_o,wr_o,nt_o,aw_o,al_o=run_portfolio(best_tps)
print(f"  各自最优TP: 年化{ann_o:+.1f}% 夏普{sr_o:.2f} 回撤{mdd_o:+.1f}% 胜率{wr_o:.0f}% 交易{nt_o}")
print(f"  TP配置: { {k:f'{v*100:.0f}%' for k,v in best_tps.items()} }")
