"""YH04 Walk-Forward 滚动窗口验证"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import akshare as ak, pandas as pd, numpy as np, matplotlib, os, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

_fonts = [f.name for f in fm.fontManager.ttflist]
CN = 'WenQuanYi Zen Hei' if 'WenQuanYi Zen Hei' in _fonts else ('SimHei' if 'SimHei' in _fonts else 'DejaVu Sans')
plt.rcParams['font.sans-serif'] = [CN]; plt.rcParams['axes.unicode_minus'] = False

# ======================== 参数(固定) ========================
INIT=1_000_000; RESERVE=100_000; COMM=0.0003; SLIP=0.0001; DCA=0
MOM=10; REBAL=5; TRAIL=0.10; HARD_STOP=0.12
BB_P=45; BB_S=2.0; RSI_P=14; RSI_L=30; RSI_H=70; ERS=65; BA=0.001
MAIN_SYM='sh512890'; MAIN_NAME='红利低波'
GROWTH={'创业板':'sz159915','科创50':'sh588000','人工智能':'sh515070','半导体':'sh512480'}
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
    if pd.isna(lo) or pd.isna(rsi): return 'HOLD',pbw
    bw=(up-lo)/row['ma'] if row['ma']>0 else 0.1
    exp=(pbw is not None and bw>pbw); nbw=bw
    bb_buy=(adj<=lo); bb_sell=(adj>=up); rsi_buy=(rsi<=RSI_L)
    if exp:
        raw_sell=(bb_sell and rsi>=ERS)
        ua=row['ua'] if not pd.isna(row['ua']) else 0; pa=row['pa'] if not pd.isna(row['pa']) else 0
        sell_sig=raw_sell and not((ua>BA)and(pa>0))
        buy_sig=(bb_buy or rsi_buy)
    else: buy_sig=(bb_buy or rsi_buy); sell_sig=(bb_sell or rsi>=RSI_H)
    if buy_sig: return 'BUY',nbw
    elif sell_sig: return 'SELL',nbw
    else: return 'HOLD',nbw

def add_growth(df):
    df=df.copy(); df['mom']=df['close']/df['close'].shift(MOM)-1
    e12=df['close'].ewm(span=12,adjust=False).mean(); e26=df['close'].ewm(span=26,adjust=False).mean()
    df['macd']=e12-e26; df['macd_s']=df['macd'].ewm(span=9,adjust=False).mean()
    df['macd_h']=df['macd']-df['macd_s']
    return df

def rank_growth(dfs,idx):
    s={}
    for n in GROWTH:
        if idx>=len(dfs[n]): continue
        h=dfs[n]['macd_h'].iloc[idx]
        if pd.isna(h): continue; s[n]=h
    return sorted(s,key=s.get,reverse=True)

def run_one(raw,df_main,dfs_growth,dates):
    ALL=list(GROWTH.keys())+[MAIN_NAME]
    sp=raw[MAIN_NAME]['close'].iloc[0]
    cash=0.0; rp=RESERVE; ra=False; rs=0.0
    shares={n:0.0 for n in ALL}; shares[MAIN_NAME]=INIT/sp*(1-COMM-SLIP)
    pos=MAIN_NAME; peak=INIT; navs=[]
    sc=2; pbw=None; ep=sp; hse=0; last_rb=None; stopped=False
    for date in dates:
        px={}
        for n in ALL:
            r=raw[n][raw[n]['date']==date]
            if len(r): px[n]=r['close'].iloc[0]
        nav=cash+sum(shares[n]*px.get(n,0) for n in ALL)
        if nav>peak: peak=nav
        mr=df_main[df_main['date']==date]
        if len(mr)==0: navs.append(nav); continue
        sig,pbw=main_signal(mr.iloc[0],pbw); mp=px.get(MAIN_NAME,0)
        if pos==MAIN_NAME and mp>0 and ep>0 and mp<ep*(1-HARD_STOP):
            cash+=shares[MAIN_NAME]*mp*(1-COMM-SLIP); shares[MAIN_NAME]=0.0; pos=None
        if sig=='BUY' and pos!=MAIN_NAME:
            if pos:
                for n in GROWTH:
                    if shares[n]>0 and n in px: cash+=shares[n]*px[n]*(1-COMM-SLIP); shares[n]=0.0
            if mp>0: val=min(cash,nav); shares[MAIN_NAME]+=val/mp*(1-COMM-SLIP); cash-=val; pos=MAIN_NAME; ep=mp; stopped=False
        elif sig=='SELL' and pos==MAIN_NAME:
            if shares[MAIN_NAME]>0 and mp>0: cash+=shares[MAIN_NAME]*mp*(1-COMM-SLIP); shares[MAIN_NAME]=0.0; pos=None
        elif sig=='HOLD' and pos!=MAIN_NAME:
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
                ranking=rank_growth(dfs_growth,bm_idx)
                if ranking and dfs_growth[ranking[0]]['macd_h'].iloc[bm_idx]>0:
                    tgt=ranking[0]
                    if tgt in px: val=min(cash,nav)
                    if val>100: shares[tgt]=val/px[tgt]*(1-COMM-SLIP); cash-=val; pos=tgt; hse=px[tgt]
                last_rb=date
        nav=cash+sum(shares[n]*px.get(n,0) for n in ALL); navs.append(nav)
    return pd.Series(navs)

print("="*60)
print(f"  YH04 Walk-Forward  (固定参数, 不优化)")
print(f"  动量{MOM}日  调仓{REBAL}日  止损{TRAIL*100:.0f}%  冷却:ON")
print("="*60)

print("获取数据...")
raw=fetch()
df_main_full=add_main(raw[MAIN_NAME])
dfs_growth_full={n:add_growth(d) for n,d in raw.items() if n!=MAIN_NAME}
all_dates=sorted(set.intersection(*[set(d['date']) for d in raw.values()]))
min_d=all_dates[0]; max_d=all_dates[-1]

# 生成窗口
ws=[]; w_start=pd.Timestamp(year=min_d.year,month=min_d.month,day=min_d.day)
while True:
    te=w_start+pd.DateOffset(years=TRAIN)-pd.DateOffset(days=1)
    ts=te+pd.DateOffset(days=1); tse=ts+pd.DateOffset(years=TEST)-pd.DateOffset(days=1)
    if tse>max_d:
        if ts<max_d: ws.append((w_start,te,ts,max_d))
        break
    ws.append((w_start,te,ts,tse)); w_start=ts

print(f"  窗口: {len(ws)}个 ({TRAIN}年训练 → {TEST}年测试)\n")

is_ret=[]; oos_ret=[]; is_sr=[]; oos_sr=[]
for i,(tr_s,tr_e,te_s,te_e) in enumerate(ws):
    # IS
    is_dates=[d for d in all_dates if tr_s<=d<=tr_e]
    df_main=df_main_full[(df_main_full['date']>=tr_s)&(df_main_full['date']<=tr_e)].reset_index(drop=True)
    df_g={n: dfs_growth_full[n][(dfs_growth_full[n]['date']>=tr_s)&(dfs_growth_full[n]['date']<=tr_e)].reset_index(drop=True) for n in GROWTH}
    raw_is={n: raw[n][(raw[n]['date']>=tr_s)&(raw[n]['date']<=tr_e)].reset_index(drop=True) for n in list(GROWTH.keys())+[MAIN_NAME]}
    nv_is=run_one(raw_is,df_main,df_g,is_dates)
    r_is=(nv_is.iloc[-1]/INIT-1)*100; dr_is=nv_is.pct_change().dropna()
    sr_is=(r_is/100/len(nv_is)*252-0.02)/(dr_is.std()*np.sqrt(252)) if dr_is.std()>0 else 0
    # OOS
    oos_dates=[d for d in all_dates if te_s<=d<=te_e]
    df_main=df_main_full[(df_main_full['date']>=te_s)&(df_main_full['date']<=te_e)].reset_index(drop=True)
    df_g={n: dfs_growth_full[n][(dfs_growth_full[n]['date']>=te_s)&(dfs_growth_full[n]['date']<=te_e)].reset_index(drop=True) for n in GROWTH}
    raw_oos={n: raw[n][(raw[n]['date']>=te_s)&(raw[n]['date']<=te_e)].reset_index(drop=True) for n in list(GROWTH.keys())+[MAIN_NAME]}
    nv_oos=run_one(raw_oos,df_main,df_g,oos_dates)
    r_oos=(nv_oos.iloc[-1]/INIT-1)*100; dr_oos=nv_oos.pct_change().dropna()
    sr_oos=(r_oos/100/len(nv_oos)*252-0.02)/(dr_oos.std()*np.sqrt(252)) if dr_oos.std()>0 else 0

    is_ret.append(r_is); oos_ret.append(r_oos); is_sr.append(sr_is); oos_sr.append(sr_oos)
    qual='✓' if sr_oos>1.0 else ('△' if sr_oos>0.5 else '⚠')
    print(f"  W{i+1} IS {tr_s.strftime('%Y')}-{tr_e.strftime('%Y')}: {r_is:+.1f}% SR{sr_is:.2f}  →  OOS {te_s.strftime('%Y')}: {r_oos:+.1f}% SR{sr_oos:.2f} {qual}")

print(f"\n  ┌─ 汇总 ─────────────────────────────────────")
print(f"  │ {'':<16} {'训练集(IS)':>12} {'测试集(OOS)':>12}")
print(f"  │ {'平均收益':<16} {np.mean(is_ret):>+11.1f}% {np.mean(oos_ret):>+11.1f}%")
print(f"  │ {'平均Sharpe':<16} {np.mean(is_sr):>12.2f} {np.mean(oos_sr):>12.2f}")
print(f"  │ {'最差Sharpe':<16} {np.min(is_sr):>12.2f} {np.min(oos_sr):>12.2f}")
decay=(np.mean(is_sr)-np.mean(oos_sr))/np.mean(is_sr)*100 if np.mean(is_sr)!=0 else 0
pos_wins=sum(1 for r in oos_ret if r>0)
print(f"  │ OOS正收益: {pos_wins}/{len(oos_ret)}  Sharpe衰减: {decay:.1f}%")
score=max(0,100-decay*1.5-np.std(oos_sr)/abs(np.mean(oos_sr))*50 if abs(np.mean(oos_sr))>0 else 100)
g='A' if score>=80 else ('B' if score>=60 else ('C' if score>=40 else 'D'))
print(f"  │ 综合: {score:.0f}/100 ({g})")

# 柱状图
fig,axes=plt.subplots(1,2,figsize=(14,5),facecolor='white')
x=np.arange(len(ws)); w=0.35
axes[0].bar(x-w/2,is_ret,w,color='#e74c3c',alpha=0.8,label='IS'); axes[0].bar(x+w/2,oos_ret,w,color='#27ae60',alpha=0.8,label='OOS')
for i,(a,b) in enumerate(zip(is_ret,oos_ret)):
    axes[0].text(i-w/2,a+1,f'{a:+.0f}%',ha='center',fontsize=9,fontweight='bold',color='#c0392b')
    axes[0].text(i+w/2,b+1,f'{b:+.0f}%',ha='center',fontsize=9,fontweight='bold',color='#1e8449')
axes[0].axhline(y=0,color='black',lw=1); axes[0].legend(); axes[0].set_xticks(x)
axes[0].set_xticklabels([f'W{i+1}' for i in range(len(ws))]); axes[0].set_title('收益对比')
axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%+.0f%%')); axes[0].grid(True,alpha=0.15,axis='y')
axes[1].plot(range(1,len(ws)+1),is_sr,'o-',color='#e74c3c',lw=1.5,label='IS Sharpe')
axes[1].plot(range(1,len(ws)+1),oos_sr,'s-',color='#27ae60',lw=1.5,label='OOS Sharpe')
axes[1].axhline(y=1.0,color='black',lw=0.8,ls='--',alpha=0.3); axes[1].legend()
axes[1].set_xticks(range(1,len(ws)+1)); axes[1].set_title('Sharpe对比'); axes[1].grid(True,alpha=0.15)
fig.suptitle(f'YH04 Walk-Forward | {TRAIN}Y→{TEST}Y | OOS均Sharpe{np.mean(oos_sr):.2f} | 评级{g}',fontsize=13,fontweight='bold')
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'walkforward_chart.png')
plt.savefig(out,dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
print(f'\n  图表: {out}')
