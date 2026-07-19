"""YH04 简化Walk-Forward: 仅创业板+红利低波(数据充足)"""
import sys, io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak, pandas as pd, numpy as np, matplotlib, os, warnings
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.ticker as mticker
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')
_fonts=[f.name for f in fm.fontManager.ttflist]
CN='WenQuanYi Zen Hei' if 'WenQuanYi Zen Hei' in _fonts else ('SimHei'if'SimHei'in _fonts else'DejaVu Sans')
plt.rcParams['font.sans-serif']=[CN]; plt.rcParams['axes.unicode_minus']=False

INIT=1_000_000; RESERVE=100_000; COMM=0.0003; SLIP=0.0001; DCA=20000
MOM=10; REBAL=5; TRAIL=0.10; NAV_STOP=0.13; HARD_STOP=0.12
BB_P=45; BB_S=2.0; RSI_P=14; RSI_L=30; RSI_H=70; ERS=65; BA=0.001
MAIN_SYM='sh512890'; MAIN_NAME='红利低波'
GROWTH={'创业板':'sz159915'}
TRAIN=2; TEST=1

def fetch():
    dfs={}
    for n,s in {**GROWTH,MAIN_NAME:MAIN_SYM}.items():
        df=ak.fund_etf_hist_sina(symbol=s); df['date']=pd.to_datetime(df['date'])
        dfs[n]=df[['date','close']].sort_values('date').reset_index(drop=True)
    return dfs

def add_main(df):
    df=df.copy(); r=df['close'].pct_change().fillna(0); r[abs(r)>0.1]=0
    df['adj']=(1+r).cumprod()
    df['ma']=df['adj'].rolling(BB_P).mean(); df['std']=df['adj'].rolling(BB_P).std()
    df['up']=df['ma']+BB_S*df['std']; df['lo']=df['ma']-BB_S*df['std']
    df['ua']=df['up'].diff().diff().rolling(3,min_periods=1).mean()
    df['pa']=df['adj'].diff().diff().rolling(3,min_periods=1).mean()
    d=df['adj'].diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/RSI_P,adjust=False).mean()/l.ewm(alpha=1/RSI_P,adjust=False).mean().replace(0,np.nan))
    return df

def main_signal(row,pbw):
    adj,rsi,up,lo=row['adj'],row['rsi'],row['up'],row['lo']
    if pd.isna(lo)or pd.isna(rsi): return'HOLD',pbw
    bw=(up-lo)/row['ma']if row['ma']>0 else 0.1
    exp=(pbw is not None and bw>pbw); nbw=bw
    bb_buy=(adj<=lo); bb_sell=(adj>=up); rsi_buy=(rsi<=RSI_L)
    if exp:
        raw_sell=(bb_sell and rsi>=ERS)
        ua=row['ua']if not pd.isna(row['ua'])else 0; pa=row['pa']if not pd.isna(row['pa'])else 0
        sell_sig=raw_sell and not((ua>BA)and(pa>0)); buy_sig=(bb_buy or rsi_buy)
    else: buy_sig=(bb_buy or rsi_buy); sell_sig=(bb_sell or rsi>=RSI_H)
    if buy_sig: return'BUY',nbw
    elif sell_sig: return'SELL',nbw
    else: return'HOLD',nbw

def add_growth(df):
    df=df.copy(); df['mom']=df['close']/df['close'].shift(MOM)-1
    e10=df['close'].ewm(span=10,adjust=False).mean(); e20=df['close'].ewm(span=20,adjust=False).mean()
    df['macd']=e10-e20; df['macd_s']=df['macd'].ewm(span=7,adjust=False).mean()
    df['macd_h']=df['macd']-df['macd_s']
    return df

def run_one(raw,df_main,dfs_growth,dates):
    ALL=list(GROWTH.keys())+[MAIN_NAME]
    sp=raw[MAIN_NAME]['close'].iloc[0]
    cash=0.0; rp=RESERVE; ra=False; rs=0.0; total_inv=INIT
    shares={n:0.0 for n in ALL}; shares[MAIN_NAME]=INIT/sp*(1-COMM-SLIP)
    pos=MAIN_NAME; peak=INIT; navs=[]
    sc=2; pbw=None; ep=sp; hse=0; last_rb=None; stopped=False; last_month=None
    for date in dates:
        ym=(date.year,date.month)
        if DCA>0 and last_month and ym!=last_month: cash+=DCA; total_inv+=DCA
        last_month=ym
        px={}
        for n in ALL:
            r=raw[n][raw[n]['date']==date]
            if len(r): px[n]=r['close'].iloc[0]
        nav=cash+sum(shares[n]*px.get(n,0)for n in ALL)
        if nav>peak: peak=nav
        dd_n=(nav-peak)/peak if peak>0 else 0
        if dd_n<-NAV_STOP and any(shares[n]>0 for n in ALL):
            for n in ALL:
                if shares[n]>0 and n in px: cash+=shares[n]*px[n]*(1-COMM-SLIP); shares[n]=0.0
            peak=nav; pos=None; stopped=True; navs.append(nav); continue
        mr=df_main[df_main['date']==date]
        if len(mr)==0: navs.append(nav); continue
        sig,pbw=main_signal(mr.iloc[0],pbw); mp=px.get(MAIN_NAME,0)
        if pos==MAIN_NAME and mp>0 and ep>0 and mp<ep*(1-HARD_STOP):
            cash+=shares[MAIN_NAME]*mp*(1-COMM-SLIP); shares[MAIN_NAME]=0.0; pos=None
        if sig=='BUY'and pos!=MAIN_NAME:
            if pos:
                for n in GROWTH:
                    if shares[n]>0 and n in px: cash+=shares[n]*px[n]*(1-COMM-SLIP); shares[n]=0.0
            if mp>0: val=min(cash,nav); shares[MAIN_NAME]+=val/mp*(1-COMM-SLIP); cash-=val; pos=MAIN_NAME; ep=mp; stopped=False
        elif sig=='SELL'and pos==MAIN_NAME:
            if shares[MAIN_NAME]>0 and mp>0: cash+=shares[MAIN_NAME]*mp*(1-COMM-SLIP); shares[MAIN_NAME]=0.0; pos=None
        elif DCA>0 and sig=='HOLD' and pos==MAIN_NAME and cash>100 and mp>0:
            shares[MAIN_NAME]+=cash/mp*(1-COMM-SLIP); cash=0
        elif sig=='HOLD'and pos!=MAIN_NAME:
            if pos and pos in px:
                if px[pos]>hse: hse=px[pos]
                if px[pos]<hse*(1-TRAIL): cash+=shares[pos]*px[pos]*(1-COMM-SLIP); shares[pos]=0.0; pos=None; stopped=True
            if stopped: navs.append(nav); continue
            bm_idx=None
            for n in GROWTH:
                idxs=dfs_growth[n][dfs_growth[n]['date']==date].index
                if len(idxs): bm_idx=idxs[0]; break
            if bm_idx is None: navs.append(nav); continue
            days=(date-last_rb).days if last_rb else 999
            if days>=REBAL and pos is None:
                # Only one growth ETF, just check MACD > 0
                if bm_idx<len(dfs_growth['创业板']) and dfs_growth['创业板']['macd_h'].iloc[bm_idx]>0:
                    tgt='创业板'
                    if tgt in px: val=min(cash,nav)
                    if val>100: shares[tgt]=val/px[tgt]*(1-COMM-SLIP); cash-=val; pos=tgt; hse=px[tgt]
                last_rb=date
        nav=cash+sum(shares[n]*px.get(n,0)for n in ALL); navs.append(nav)
    return pd.Series(navs), total_inv

print("="*60)
print(f"  YH04 Walk-Forward (创业+红利)  固定参数")
print("="*60)
raw=fetch()
df_main_full=add_main(raw[MAIN_NAME])
dfs_g=add_growth(raw['创业板'])
all_dates=sorted(set.intersection(*[set(d['date'])for d in raw.values()]))
min_d=all_dates[0]; max_d=all_dates[-1]

ws=[]; w_start=pd.Timestamp(year=min_d.year,month=min_d.month,day=min_d.day)
while True:
    te=w_start+pd.DateOffset(years=TRAIN)-pd.DateOffset(days=1)
    ts=te+pd.DateOffset(days=1); tse=ts+pd.DateOffset(years=TEST)-pd.DateOffset(days=1)
    if tse>max_d:
        if ts<max_d: ws.append((w_start,te,ts,max_d))
        break
    ws.append((w_start,te,ts,tse)); w_start=ts

print(f"  窗口: {len(ws)}个\n")
is_ret=[]; oos_ret=[]; is_sr=[]; oos_sr=[]
for i,(tr_s,tr_e,te_s,te_e)in enumerate(ws):
    # IS
    is_d=[d for d in all_dates if tr_s<=d<=tr_e]
    dm=df_main_full[(df_main_full['date']>=tr_s)&(df_main_full['date']<=tr_e)].reset_index(drop=True)
    dg2=dfs_g[(dfs_g['date']>=tr_s)&(dfs_g['date']<=tr_e)].reset_index(drop=True)
    ri={n:raw[n][(raw[n]['date']>=tr_s)&(raw[n]['date']<=tr_e)].reset_index(drop=True)for n in['创业板',MAIN_NAME]}
    nv_is,inv_is=run_one(ri,dm,{'创业板':dg2},is_d)
    r_is=(nv_is.iloc[-1]/inv_is-1)*100 if inv_is>0 else 0; dr_is=nv_is.pct_change().dropna()
    sr_is=(r_is/100/len(nv_is)*252-0.02)/(dr_is.std()*np.sqrt(252))if dr_is.std()>0 else 0
    # OOS
    oos_d=[d for d in all_dates if te_s<=d<=te_e]
    dm=df_main_full[(df_main_full['date']>=te_s)&(df_main_full['date']<=te_e)].reset_index(drop=True)
    dg2=dfs_g[(dfs_g['date']>=te_s)&(dfs_g['date']<=te_e)].reset_index(drop=True)
    ri={n:raw[n][(raw[n]['date']>=te_s)&(raw[n]['date']<=te_e)].reset_index(drop=True)for n in['创业板',MAIN_NAME]}
    nv_oos,inv_oos=run_one(ri,dm,{'创业板':dg2},oos_d)
    r_oos=(nv_oos.iloc[-1]/inv_oos-1)*100 if inv_oos>0 else 0; dr_oos=nv_oos.pct_change().dropna()
    sr_oos=(r_oos/100/len(nv_oos)*252-0.02)/(dr_oos.std()*np.sqrt(252))if dr_oos.std()>0 else 0
    is_ret.append(r_is); oos_ret.append(r_oos); is_sr.append(sr_is); oos_sr.append(sr_oos)
    qual='OK'if sr_oos>1.0 else('--'if sr_oos>0.5 else'XX')
    print(f"  W{i+1} IS {tr_s.strftime('%Y')}-{tr_e.strftime('%Y')}: {r_is:+.1f}% SR{sr_is:.2f}  OOS {te_s.strftime('%Y')}: {r_oos:+.1f}% SR{sr_oos:.2f} [{qual}]")

print(f"\n  OOS平均Sharpe: {np.mean(oos_sr):.2f}  最差: {np.min(oos_sr):.2f}  正收益: {sum(1 for r in oos_ret if r>0)}/{len(oos_ret)}")
decay=(np.mean(is_sr)-np.mean(oos_sr))/np.mean(is_sr)*100 if np.mean(is_sr)!=0 else 0
print(f"  Sharpe衰减: {decay:.1f}%  {'OK'if decay<20 else 'WARN'}")
