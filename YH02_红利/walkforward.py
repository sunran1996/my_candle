import sys,io,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
from data_feed import get_data
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False

INIT=1_000_000;COMM=0.0001;SLIP=0.0001
BB_P=45;BB_S=2.0;RSI_P=14;RSI_L=30;RSI_H=70;ERS=65;HSTOP=0.12;ACC=0.001

def ci(df):
    ret=df['close'].pct_change().fillna(0);ret[abs(ret)>0.1]=0
    df['adj_close']=(1+ret).cumprod()
    df['ma']=df['adj_close'].rolling(BB_P).mean();df['std']=df['adj_close'].rolling(BB_P).std()
    df['upper']=df['ma']+BB_S*df['std'];df['lower']=df['ma']-BB_S*df['std']
    df['uv']=df['upper'].diff();df['ua']=df['uv'].diff().rolling(3,min_periods=1).mean()
    df['pv']=df['adj_close'].diff();df['pa']=df['pv'].diff().rolling(3,min_periods=1).mean()
    d=df['adj_close'].diff();g=d.clip(lower=0);l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/RSI_P,adjust=False).mean()/l.ewm(alpha=1/RSI_P,adjust=False).mean().replace(0,np.nan))
    return df

def bt(test_dates_set,df):
    sp=df['close'].iloc[0];cash=0.0;shares=INIT/sp*(1-COMM-SLIP)
    peak=INIT;navs=[];pbw=None;sc=0;ep=sp
    for _,row in df.iterrows():
        d=row['date'];px=row['close'];adj=row['adj_close'];lo=row['lower'];up=row['upper'];rsi=row['rsi']
        if np.isnan(lo)or np.isnan(rsi):continue
        nav=cash+shares*px
        if nav>peak:peak=nav
        if shares>0 and px<ep*(1-HSTOP):cash+=shares*px*(1-COMM-SLIP);shares=0
        bb_b=adj<=lo;bb_s=adj>=up;rsi_b=rsi<=RSI_L;rsi_s=rsi>=RSI_H
        bw=(up-lo)/row['ma']if row['ma']>0 else 0.1
        exp=(pbw is not None and bw>pbw)
        if exp:
            buy=(bb_b or rsi_b);raw_s=bb_s and rsi>=ERS
            ua=row['ua']if not np.isnan(row['ua'])else 0;pa=row['pa']if not np.isnan(row['pa'])else 0
            sell=raw_s and not(ua>ACC and pa>0)
        else:buy=(bb_b or rsi_b);sell=(bb_s or rsi_s)
        pbw=bw
        if buy or sell:sc+=1
        if sc<2:
            if d in test_dates_set:navs.append(nav)
            continue
        if buy and cash>0:shares+=cash/px*(1-COMM-SLIP);cash=0;ep=px
        elif sell and shares>0:cash+=shares*px*(1-COMM-SLIP);shares=0
        if d in test_dates_set:navs.append(cash+shares*px)
    return navs

print('Fetching data...')
df=get_data('sh512890');df['date']=pd.to_datetime(df['date']);df=df.sort_values('date')
ci(df)
all_dates=sorted(df['date'].tolist())
all_dates=[d for d in all_dates if d>=pd.Timestamp('2018-01-01')]

TRAIN=3;TEST=1
windows=[]
yr=2018
while yr+TRAIN+TEST<=2027:
    ts=pd.Timestamp(str(yr)+'-01-01')
    tes=pd.Timestamp(str(yr+TRAIN)+'-01-01')
    tee=pd.Timestamp(str(yr+TRAIN+TEST)+'-01-01')
    trd=[d for d in all_dates if d>=ts and d<tes]
    ted=[d for d in all_dates if d>=tes and d<tee]
    if len(trd)>400 and len(ted)>150:
        windows.append((str(yr)+'-'+str(yr+TRAIN),ts,tes,tee,trd,ted))
    yr+=1

print('')
print('  YH02 Walk-Forward (train'+str(TRAIN)+'yr -> test'+str(TEST)+'yr)')
print('  '+'-'*70)
print('  '+'Window'.ljust(12)+'Train'.ljust(14)+'Test'.ljust(14)+'OOS'.rjust(10)+'Ann'.rjust(8)+'Sharpe'.rjust(7)+'MaxDD'.rjust(8))
print('  '+'-'*70)

an=[];results=[]
for label,ts,tes,tee,trd,ted in windows:
    dfs=df[(df['date']>=ts)&(df['date']<tee)].reset_index(drop=True)
    ci(dfs)
    teset=set(ted)
    oos=bt(teset,dfs)
    if len(oos)<10:continue
    nav_arr=np.array(oos)
    ret=(nav_arr[-1]/INIT-1)*100
    ann=((1+ret/100)**(252/len(nav_arr))-1)*100
    dr=pd.Series(nav_arr).pct_change().dropna()
    sr=(ann/100-0.02)/(dr.std()*np.sqrt(252))if dr.std()>0 else 0
    cum=pd.Series(nav_arr/INIT);mdd=((cum-cum.cummax())/cum.cummax()).min()*100
    tr=trd[0].strftime('%Y-%m');te=str(tes.year)+'-'+str(tee.year)
    print('  '+label.ljust(12)+tr.ljust(14)+te.ljust(14)+('{:>+9.1f}%'.format(ret)).rjust(10)+('{:>+7.1f}%'.format(ann)).rjust(8)+('{:>+6.2f}'.format(sr)).rjust(7)+('{:>+7.1f}%'.format(mdd)).rjust(8))
    results.append({'w':label,'ret':ret,'ann':ann,'sr':sr,'mdd':mdd});an.append(nav_arr)

rets=[r['ret']for r in results];anns=[r['ann']for r in results]
srs=[r['sr']for r in results];mdds=[r['mdd']for r in results]
print('  '+'-'*70)
print('  '+'Avg'.ljust(12)+''.ljust(14)+''.ljust(14)+('{:>+9.1f}%'.format(np.mean(rets))).rjust(10)+('{:>+7.1f}%'.format(np.mean(anns))).rjust(8)+('{:>+6.2f}'.format(np.mean(srs))).rjust(7)+('{:>+7.1f}%'.format(np.mean(mdds))).rjust(8))
print('  '+'Worst'.ljust(12)+''.ljust(14)+''.ljust(14)+('{:>+9.1f}%'.format(np.min(rets))).rjust(10)+('{:>+7.1f}%'.format(np.min(anns))).rjust(8)+('{:>+6.2f}'.format(np.min(srs))).rjust(7)+('{:>+7.1f}%'.format(np.max(mdds))).rjust(8))
wr=sum(1 for r in rets if r>0)/len(rets)*100
print('  WinRate: '+str(int(wr))+'% ('+str(sum(1 for r in rets if r>0))+'/'+str(len(rets))+')')

if an:
    fig,axes=plt.subplots(1,2,figsize=(18,6),facecolor='white')
    ax=axes[0];cols=plt.cm.tab10(np.linspace(0,1,len(an)))
    for i,nav in enumerate(an):
        n2=nav/nav[0];w=results[i]['w'];rr=results[i]['ret']
        ax.plot(range(len(n2)),n2,color=cols[i],lw=1.8,label=w+' ('+('{:+.0f}'.format(rr))+'%)')
    ax.axhline(y=1,color='#888',lw=0.8,ls='--');ax.legend(fontsize=8,loc='upper left');ax.grid(True,alpha=0.12)
    ax.set_title('YH02 Walk-Forward OOS (avg '+('{:+.1f}'.format(np.mean(rets)))+'%)',fontsize=13,fontweight='bold')
    ax=axes[1]
    c2=['#CC2222'if r>=0 else'#228B22'for r in rets]
    bars=ax.bar(range(len(rets)),rets,color=c2,alpha=0.85,edgecolor='white',lw=1)
    for bar,val in zip(bars,rets):
        off=1.5 if val>=0 else-3.5
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+off,'{:+.1f}%'.format(val),ha='center',fontsize=12,fontweight='bold',color='#CC2222'if val>=0 else'#228B22')
    ax.axhline(y=0,color='black',lw=1)
    ax.set_xticks(range(len(rets)));ax.set_xticklabels([r['w']for r in results],fontsize=10)
    ax.set_title('OOS Returns by Window',fontsize=13,fontweight='bold');ax.grid(True,alpha=0.12,axis='y')
    plt.savefig('walkforward_chart.png',dpi=150,bbox_inches='tight',facecolor='white');plt.close()
    print('  Chart: walkforward_chart.png')
