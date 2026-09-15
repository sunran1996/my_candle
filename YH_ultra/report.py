# -*- coding: utf-8 -*-
"""分股票最优参数详细回测报告"""
import sys,io,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak,pandas as pd,numpy as np
warnings.filterwarnings('ignore')

STOCKS={'山东高速':'sh600350','渝农商行':'sh601077','皖通高速':'sh600012','江苏银行':'sh600919'}
INIT=1_000_000;COMM=0.0003;SLIP=0.0001;MAX_POS=0.25

# 各自最优买点
OPT={
    '山东高速': (45, 0.25),
    '渝农商行': (30, 0.10),
    '皖通高速': (38, 0.12),
    '江苏银行': (35, 0.10),
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
    d=c.diff();g=d.clip(lower=0);l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/14,adjust=False).mean()/l.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan))
    return df

# ============ 单只股票详细回测 ============
def run_solo_detailed(name, rsi_th, bb_th):
    raw=fetch();df=add_indicators(raw[name])
    dates=[d for d in df['date'] if d>=pd.Timestamp('2019-01-01')]
    cash=INIT;shares=0;entry=0;high=0
    navs=[];trades=[]

    for date in dates:
        r=df[df['date']==date]
        if len(r)==0:continue
        cp=r['close'].iloc[0];row=r.iloc[0]
        if shares>0:
            if cp>high:high=cp
            pnl=cp/entry-1;dd=cp/high-1
            do=False
            if pnl<=-0.10:do=True
            elif dd<=-0.08:do=True
            elif pnl>=0.20:do=True
            if do:
                cash+=shares*cp*(1-COMM-SLIP)
                trades.append({'date':date,'pnl':pnl*100})
                shares=0;entry=0;high=0
        if shares==0:
            if pd.isna(row['bb_lo'])or pd.isna(row['rsi']):continue
            rsi=row['rsi'];c=row['close'];lo=row['bb_lo'];up=row['bb_up']
            if up<=lo:continue
            dist=(c-lo)/(up-lo)
            sc=(1 if rsi<=rsi_th else 0)+(1 if dist<=bb_th else 0)
            if rsi<=30:sc+=1
            if sc>=1:
                val=min(cash,INIT*MAX_POS)
                if val>5000:
                    shares=val/cp*(1-COMM-SLIP);cash-=val
                    entry=cp;high=cp
        navs.append({'date':date,'nav':cash+shares*cp})

    ndf=pd.DataFrame(navs)
    # 统计
    final=ndf['nav'].iloc[-1]
    ret=(final/INIT-1)*100;ann=((1+ret/100)**(252/len(ndf))-1)*100
    dr=ndf['nav'].pct_change().dropna();vol=dr.std()*np.sqrt(252)*100
    sr=(ann-2)/vol if vol>0 else 0
    mdd=((ndf['nav']-ndf['nav'].cummax())/ndf['nav'].cummax()).min()*100

    sells=[t for t in trades if t['pnl']!=0]
    pnls=np.array([s['pnl'] for s in sells])
    wr=(pnls>0).sum()/len(pnls)*100 if len(pnls)>0 else 0
    aw=pnls[pnls>0].mean()if(pnls>0).sum()>0 else 0
    al=pnls[pnls<0].mean()if(pnls<0).sum()>0 else 0

    # 逐年
    ndf['year']=ndf['date'].dt.year
    years={}
    for yr,grp in ndf.groupby('year'):
        if len(grp)<10:continue
        yr_ret=(grp['nav'].iloc[-1]/grp['nav'].iloc[0]-1)*100
        yr_mdd=((grp['nav']-grp['nav'].cummax())/grp['nav'].cummax()).min()*100
        years[yr]=(yr_ret,yr_mdd)

    return ann,ret,sr,mdd,wr,len(sells),aw,al,years,ndf

# ============ 组合回测(详细) ============
def run_portfolio_detailed(buy_params):
    raw=fetch();dfs={n:add_indicators(d)for n,d in raw.items()}
    dates=sorted(set.intersection(*[set(d['date'])for d in dfs.values()]))
    dates=[d for d in dates if d>=pd.Timestamp('2019-01-01')]
    cash=INIT;shares={n:0.0 for n in STOCKS};entry={n:0.0 for n in STOCKS};high={n:0.0 for n in STOCKS}
    navs=[];trades=[]
    stock_trades={n:[] for n in STOCKS}

    for date in dates:
        px={n:raw[n][raw[n]['date']==date]['close'].iloc[0]for n in STOCKS if len(raw[n][raw[n]['date']==date])>0}

        for n in STOCKS:
            if shares[n]<=0:continue
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            if cp>high[n]:high[n]=cp
            pnl=cp/entry[n]-1;dd=cp/high[n]-1
            do=False
            if pnl<=-0.10:do=True
            elif dd<=-0.08:do=True
            elif pnl>=0.20:do=True
            if do:
                cash+=shares[n]*cp*(1-COMM-SLIP)
                trades.append({'date':date,'name':n,'pnl':pnl*100})
                stock_trades[n].append({'date':date,'pnl':pnl*100})
                shares[n]=0;entry[n]=0;high[n]=0

        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)

        for n in STOCKS:
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            if shares[n]>0:continue
            row=r.iloc[0]
            if pd.isna(row['bb_lo'])or pd.isna(row['rsi']):continue
            rsi=row['rsi'];c=row['close'];lo=row['bb_lo'];up=row['bb_up']
            if up<=lo:continue
            dist=(c-lo)/(up-lo)
            rsi_th,bb_th=buy_params[n]
            sc=(1 if rsi<=rsi_th else 0)+(1 if dist<=bb_th else 0)
            if rsi<=30:sc+=1
            if sc<1:continue
            val=min(cash,nav*MAX_POS)
            if val>5000:
                qty=val/cp*(1-COMM-SLIP)
                shares[n]=qty;cash-=val
                entry[n]=cp;high[n]=cp

        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)
        navs.append({'date':date,'nav':nav})

    ndf=pd.DataFrame(navs)
    final=ndf['nav'].iloc[-1]
    ret=(final/INIT-1)*100;ann=((1+ret/100)**(252/len(ndf))-1)*100
    dr=ndf['nav'].pct_change().dropna();vol=dr.std()*np.sqrt(252)*100
    sr=(ann-2)/vol if vol>0 else 0
    mdd=((ndf['nav']-ndf['nav'].cummax())/ndf['nav'].cummax()).min()*100

    sells_all=[t for t in trades if t['pnl']!=0]
    pnls=np.array([s['pnl'] for s in sells_all])
    wr=(pnls>0).sum()/len(pnls)*100 if len(pnls)>0 else 0
    aw=pnls[pnls>0].mean()if(pnls>0).sum()>0 else 0
    al=pnls[pnls<0].mean()if(pnls<0).sum()>0 else 0

    ndf['year']=ndf['date'].dt.year
    years={}
    for yr,grp in ndf.groupby('year'):
        if len(grp)<10:continue
        yr_ret=(grp['nav'].iloc[-1]/grp['nav'].iloc[0]-1)*100
        yr_mdd=((grp['nav']-grp['nav'].cummax())/grp['nav'].cummax()).min()*100
        years[yr]=(yr_ret,yr_mdd)

    # 每只股票的贡献
    stock_stats={}
    for n in STOCKS:
        st=[t for t in stock_trades[n] if t['pnl']!=0]
        if st:
            sp=np.array([t['pnl'] for t in st])
            swr=(sp>0).sum()/len(sp)*100
            saw=sp[sp>0].mean()if(sp>0).sum()>0 else 0
            sal=sp[sp<0].mean()if(sp<0).sum()>0 else 0
            stock_stats[n]={'trades':len(sp),'wr':swr,'aw':saw,'al':sal}
        else:
            stock_stats[n]={'trades':0,'wr':0,'aw':0,'al':0}

    return ann,ret,sr,mdd,wr,len(sells_all),aw,al,years,stock_stats

# ============ 运行 ============
print("="*80)
print("  分股票最优买点 — 详细回测报告 (2019至今, 无缩放)")
print("="*80)

# ---- 单只股票 ----
print("\n  ┌─ 单只股票独立回测 (各自最优参数, TP=20%)")
print("  │")
uni_params={n:(42,0.25) for n in STOCKS}

for name in STOCKS:
    rsi_th,bb_th=OPT[name]
    ann,ret,sr,mdd,wr,nt,aw,al,years,_=run_solo_detailed(name,rsi_th,bb_th)
    print(f"  ├─ {name} (RSI≤{rsi_th} BB≤{bb_th:.2f})")
    print(f"  │  年化{ann:+.1f}%  累计{ret:+.0f}%  夏普{sr:.2f}  回撤{mdd:+.1f}%  胜率{wr:.0f}%  交易{nt}  均盈{aw:+.1f}%  均亏{al:+.1f}%")
    for yr in sorted(years.keys()):
        yr_ret,yr_mdd=years[yr]
        print(f"  │  {yr}  {yr_ret:+.1f}%  MaxDD{yr_mdd:+.1f}%")
    print("  │")

# ---- 组合 ----
print("  ├─────────────────────────────────────────────")
ann,ret,sr,mdd,wr,nt,aw,al,years,ss=run_portfolio_detailed(OPT)
print(f"  ├─ 组合回测 (各自最优参数)")
print(f"  │  年化{ann:+.1f}%  累计{ret:+.0f}%  夏普{sr:.2f}  回撤{mdd:+.1f}%  胜率{wr:.0f}%  交易{nt}  均盈{aw:+.1f}%  均亏{al:+.1f}%")
print(f"  │")
for yr in sorted(years.keys()):
    yr_ret,yr_mdd=years[yr]
    print(f"  │  {yr}  {yr_ret:+.1f}%  MaxDD{yr_mdd:+.1f}%")
print(f"  │")
print(f"  ├─ 分股票贡献:")
for n,st in ss.items():
    rsi_th,bb_th=OPT[n]
    print(f"  │  {n} (RSI≤{rsi_th} BB≤{bb_th:.2f}): {st['trades']}笔 胜率{st['wr']:.0f}%  均盈{st['aw']:+.1f}%  均亏{st['al']:+.1f}%")
print(f"  └─────────────────────────────────────────────")

# ---- 对比统一参数 ----
ann_u,ret_u,sr_u,mdd_u,wr_u,nt_u,aw_u,al_u,years_u,_=run_portfolio_detailed(uni_params)
print(f"\n  ┌─ 组合对比")
print(f"  ├─ 统一参数(RSI≤42 BB≤0.25): 年化{ann_u:+.1f}%  夏普{sr_u:.2f}  回撤{mdd_u:+.1f}%  交易{nt_u}")
print(f"  ├─ 各自最优:                年化{ann:+.1f}%  夏普{sr:.2f}  回撤{mdd:+.1f}%  交易{nt}")
print(f"  └─ 提升:                   Δ{ann-ann_u:+.1f}%  Δ{sr-sr_u:+.2f}  Δ{mdd-mdd_u:+.1f}%")
