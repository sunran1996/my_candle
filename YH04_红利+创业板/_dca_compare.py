"""YH04 DCA vs 无DCA 净值对比 + 回撤区间标注"""
import sys, io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak, pandas as pd, numpy as np, matplotlib, os, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')
_fonts=[f.name for f in fm.fontManager.ttflist]
CN='WenQuanYi Zen Hei' if 'WenQuanYi Zen Hei' in _fonts else ('SimHei'if'SimHei'in _fonts else'DejaVu Sans')
plt.rcParams['font.sans-serif']=[CN]; plt.rcParams['axes.unicode_minus']=False

INIT=1_000_000; RESERVE=100_000; COMM=0.0003; SLIP=0.0001
MOM=10; REBAL=5; TRAIL=0.10; NAV_STOP=0.13; HARD_STOP=0.12
BB_P=45; BB_S=2.0; RSI_P=14; RSI_L=30; RSI_H=70; ERS=65; BA=0.001
MAIN_SYM='sh512890'; MAIN_NAME='红利低波'
GROWTH={'创业板':'sz159915','科创50':'sh588000','人工智能':'sh515070','半导体':'sh512480'}

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

def run_one(raw,df_main,dfs_growth,dates,dca_val):
    ALL=list(GROWTH.keys())+[MAIN_NAME]
    sp=raw[MAIN_NAME]['close'].iloc[0]
    cash=0.0; rp=RESERVE; ra=False; rs=0.0; total_inv=INIT
    shares={n:0.0 for n in ALL}; shares[MAIN_NAME]=INIT/sp*(1-COMM-SLIP)
    pos=MAIN_NAME; peak=INIT; navs=[]
    sc=2; pbw=None; ep=sp; hse=0; last_rb=None; stopped=False; last_month=None
    for date in dates:
        ym=(date.year,date.month)
        if dca_val>0 and last_month and ym!=last_month: cash+=dca_val; total_inv+=dca_val
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
            peak=nav; pos=None; stopped=True
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
        elif dca_val>0 and sig=='HOLD'and pos==MAIN_NAME and cash>100 and mp>0:
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
                scores={}
                for n in GROWTH:
                    p2=bm_idx
                    if p2<len(dfs_growth[n]): v=dfs_growth[n]['macd_h'].iloc[p2]
                    else: v=np.nan
                    if not pd.isna(v): scores[n]=v
                ranking=sorted(scores,key=scores.get,reverse=True)
                if ranking and scores[ranking[0]]>0:
                    tgt=ranking[0]
                    if tgt in px: val=min(cash,nav)
                    if val>100: shares[tgt]=val/px[tgt]*(1-COMM-SLIP); cash-=val; pos=tgt; hse=px[tgt]
                last_rb=date
        nav=cash+sum(shares[n]*px.get(n,0)for n in ALL); navs.append(nav)
    return pd.DataFrame({'date':dates,'nav':navs}), total_inv

print("计算中...")
raw=fetch(); df_main=add_main(raw[MAIN_NAME])
dfs_g={n:add_growth(d) for n,d in raw.items() if n!=MAIN_NAME}
dates=sorted(set.intersection(*[set(d['date'])for d in raw.values()]))
dates=[d for d in dates if d>=pd.Timestamp('2020-01-01')]

ndf0,inv0=run_one(raw,df_main,dfs_g,dates,0)
ndf1,inv1=run_one(raw,df_main,dfs_g,dates,20000)

ret0=(ndf0['nav'].iloc[-1]/INIT-1)*100
ret1=(ndf1['nav'].iloc[-1]/inv1-1)*100

print(f"无DCA: 100万→{ndf0['nav'].iloc[-1]/1e4:.0f}万  +{ret0:.1f}%")
print(f"DCA:   {inv1/1e4:.0f}万→{ndf1['nav'].iloc[-1]/1e4:.0f}万  +{ret1:.1f}%  净赚{ndf1['nav'].iloc[-1]-inv1:,.0f}")

# ===== 图表 =====
fig,ax=plt.subplots(figsize=(16,8),facecolor='white')
ax.plot(ndf0['date'],ndf0['nav']/INIT,color='#CC2222',lw=2.0,label=f'无DCA (100万→{ndf0["nav"].iloc[-1]/1e4:.0f}万 +{ret0:.1f}%)')
ax.plot(ndf1['date'],ndf1['nav']/INIT,color='#2563EB',lw=2.0,label=f'DCA月投2万 ({inv1/1e4:.0f}万→{ndf1["nav"].iloc[-1]/1e4:.0f}万 +{ret1:.1f}%)')

# 回撤区间(绿色)
cum=ndf0['nav']/INIT; dd_series=(cum-cum.cummax())/cum.cummax()
in_dd=False; dd_start=None
for i,(d2,dd_val)in enumerate(zip(ndf0['date'],dd_series)):
    if dd_val<-0.1 and not in_dd: dd_start=d2; in_dd=True
    elif dd_val>-0.03 and in_dd and dd_start:
        ax.axvspan(dd_start,d2,alpha=0.18,color='green')
        in_dd=False; dd_start=None
if in_dd and dd_start: ax.axvspan(dd_start,ndf0['date'].iloc[-1],alpha=0.18,color='green')

ax.axhline(y=1.0,color='#888',lw=0.8,ls='--')
ax.legend(fontsize=13,loc='upper left')
ax.set_ylabel('净值倍数',fontsize=12); ax.grid(True,alpha=0.12)
ax.set_title(f'YH04 DCA vs 无DCA 净值对比 | 绿色=回撤>10%区间',fontsize=15,fontweight='bold')

# 信息卡
info=(f'无DCA: +{ret0:.1f}%  Sharpe {0:.2f}\n'
      f'含DCA: +{ret1:.1f}%  净赚{ndf1["nav"].iloc[-1]-inv1:,.0f}\n'
      f'投入{inv1/1e4:.0f}万  终值{ndf1["nav"].iloc[-1]/1e4:.0f}万')
ax.text(0.02,0.95,info,transform=ax.transAxes,fontsize=11,va='top',family='monospace',
        bbox=dict(boxstyle='round',facecolor='#FFFACD',edgecolor='#DDD',alpha=0.9))

out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'dca_compare.png')
plt.savefig(out,dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
print(f'图表: {out}')

# DCA明细
print(f'\nDCA版本逐年:')
ndf1['year']=ndf1['date'].dt.year
for yr,grp in ndf1.groupby('year'):
    if len(grp)<10: continue
    yr_ret=(grp['nav'].iloc[-1]/grp['nav'].iloc[0]-1)*100
    print(f'  {yr}: {yr_ret:+.1f}%')
