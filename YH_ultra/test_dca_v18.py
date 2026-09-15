# -*- coding: utf-8 -*-
"""v18 (TS=7%) + DCA每月2万注入"""
import sys,io,os,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak,pandas as pd,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

_fonts=[f.name for f in fm.fontManager.ttflist]
CN='WenQuanYi Zen Hei' if 'WenQuanYi Zen Hei' in _fonts else ('SimHei' if 'SimHei' in _fonts else 'DejaVu Sans')
plt.rcParams['font.sans-serif']=[CN];plt.rcParams['axes.unicode_minus']=False

STOCKS={'山东高速':'sh600350','渝农商行':'sh601077','皖通高速':'sh600012','江苏银行':'sh600919'}
INIT=1_000_000;COMM=0.0003;SLIP=0.0001;MAX_POS=0.25;DCA=20000
# v18参数: TS=7%
TAKE_PROFIT=0.20;TAKE_PROFIT_HI=0.25;TRAIL_STOP=0.07;HARD_STOP=0.10;COOLDOWN=20
BUY_PARAMS={
    '山东高速':{'rsi':45,'bb':0.25},'渝农商行':{'rsi':30,'bb':0.10},
    '皖通高速':{'rsi':38,'bb':0.12},'江苏银行':{'rsi':35,'bb':0.10},
}
SCRIPT=os.path.dirname(os.path.abspath(__file__))

def fetch():
    dfs={}
    for n,s in STOCKS.items():
        df=ak.stock_zh_a_daily(symbol=s,adjust='qfq');df['date']=pd.to_datetime(df['date'])
        dfs[n]=df[['date','open','high','low','close','volume']].sort_values('date').reset_index(drop=True)
    return dfs

def add_indicators(df):
    df=df.copy();c=df['close']
    df['bb_ma']=c.rolling(20).mean();df['bb_std']=c.rolling(20).std()
    df['bb_up']=df['bb_ma']+2*df['bb_std'];df['bb_lo']=df['bb_ma']-2*df['bb_std']
    df['bb_up_d2']=df['bb_up'].diff().diff()
    d=c.diff();g=d.clip(lower=0);l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/14,adjust=False).mean()/l.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan))
    return df

def check_buy(row,name):
    if pd.isna(row['bb_lo'])or pd.isna(row['rsi']):return False,0
    rsi=row['rsi'];c=row['close'];lo=row['bb_lo'];up=row['bb_up']
    if up<=lo:return False,0
    dist=(c-lo)/(up-lo)
    bp=BUY_PARAMS.get(name,{'rsi':42,'bb':0.25})
    sc=(1 if rsi<=bp['rsi'] else 0)+(1 if dist<=bp['bb'] else 0)
    if rsi<=30:sc+=1
    return sc>=1,sc

print("获取数据...")
raw=fetch();dfs={n:add_indicators(d)for n,d in raw.items()}
dates=sorted(set.intersection(*[set(d['date'])for d in dfs.values()]))
dates=[d for d in dates if d>=pd.Timestamp('2019-01-01')]

def run(tag, dca_mode):
    cash=INIT;shares={n:0.0 for n in STOCKS};entry={n:0.0 for n in STOCKS};high={n:0.0 for n in STOCKS}
    accel={n:False for n in STOCKS};cooldown={n:0 for n in STOCKS}
    navs=[];cost_basis=[];trades=[]
    total_invested=INIT;last_month=None

    for date in dates:
        if dca_mode:
            this_month=date.month
            if last_month is not None and this_month!=last_month:
                cash+=DCA;total_invested+=DCA
            last_month=this_month

        sold_today={n:False for n in STOCKS}
        px={n:raw[n][raw[n]['date']==date]['close'].iloc[0]for n in STOCKS if len(raw[n][raw[n]['date']==date])>0}

        for n in STOCKS:
            if shares[n]<=0:continue
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            if cp>high[n]:high[n]=cp
            pnl=cp/entry[n]-1;dd=cp/high[n]-1
            do=False;sell_px=cp;why=''

            if pnl<=-HARD_STOP:do=True;why='hard'
            elif accel[n]:
                if pnl>=TAKE_PROFIT_HI:do=True;why='accel_tp25'
                elif dd<=-TRAIL_STOP:
                    floor=entry[n]*1.20;stop_px=max(high[n]*(1-TRAIL_STOP),floor)
                    if cp<=stop_px:do=True;sell_px=max(cp,floor);why='accel_floor'
            elif dd<=-TRAIL_STOP:do=True;why='trail'
            elif pnl>=TAKE_PROFIT:
                d2=r.iloc[0].get('bb_up_d2')
                if not pd.isna(d2) and d2>0:accel[n]=True
                else:do=True;why='tp20'

            if do:
                cash+=shares[n]*sell_px*(1-COMM-SLIP)
                trades.append((sell_px/entry[n]-1)*100)
                shares[n]=0;entry[n]=0;high[n]=0;accel[n]=False;sold_today[n]=True
                if sell_px/entry[n]-1<=-HARD_STOP:cooldown[n]=COOLDOWN

        nav_val=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)
        for n in STOCKS:
            if cooldown[n]>0:cooldown[n]-=1

        for n in STOCKS:
            if sold_today[n]:continue
            if shares[n]>0:continue
            if cooldown[n]>0:continue
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            ok,sc=check_buy(r.iloc[0],n)
            if not ok:continue
            val=min(cash,nav_val*MAX_POS)
            if val>5000:
                qty=val/cp*(1-COMM-SLIP);shares[n]=qty;cash-=val
                entry[n]=cp;high[n]=cp

        nav_val=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)
        navs.append(nav_val)
        cost_basis.append(total_invested)

    ndf=pd.DataFrame({'date':dates,'nav':navs,'cost':cost_basis})
    return ndf,total_invested,trades

print("回测 v18 纯...")
ndf18,total18,trades18=run('v18纯',dca_mode=False)
fin18=ndf18['nav'].iloc[-1];ret18=(fin18/INIT-1)*100
ann18=((1+ret18/100)**(252/len(ndf18))-1)*100
dr18=ndf18['nav'].pct_change().dropna();vol18=dr18.std()*np.sqrt(252)*100
sr18=(ann18-2)/vol18 if vol18>0 else 0
mdd18=((ndf18['nav']-ndf18['nav'].cummax())/ndf18['nav'].cummax()).min()*100

print("回测 v18+DCA注入...")
ndf_dca,total_dca,trades_dca=run('v18+DCA',dca_mode=True)
fin_dca=ndf_dca['nav'].iloc[-1]
ret_dca=(fin_dca/total_dca-1)*100
ann_dca_apparent=((fin_dca/INIT)**(252/len(ndf_dca))-1)*100
dr_dca=ndf_dca['nav'].pct_change().dropna();vol_dca=dr_dca.std()*np.sqrt(252)*100
sr_dca=(ann_dca_apparent-2)/vol_dca if vol_dca>0 else 0
mdd_dca=((ndf_dca['nav']-ndf_dca['nav'].cummax())/ndf_dca['nav'].cummax()).min()*100

pnls18=np.array(trades18);pnls_dca=np.array(trades_dca)
wr18=(pnls18>0).sum()/len(pnls18)*100 if len(pnls18)>0 else 0
wr_dca=(pnls_dca>0).sum()/len(pnls_dca)*100 if len(pnls_dca)>0 else 0

# ============ 画图 ============
print("画图...")
fig,axes=plt.subplots(2,1,figsize=(20,12),facecolor='white',
    gridspec_kw={'height_ratios':[3,1],'hspace':0.08})

ax=axes[0]
ax.plot(ndf18['date'],ndf18['nav']/10000,color='#2ECC71',lw=2.0,
        label=f'v18纯 TS=7% (终值{fin18/10000:.0f}万)',zorder=3)
ax.plot(ndf_dca['date'],ndf_dca['nav']/10000,color='#E74C3C',lw=2.0,
        label=f'v18+DCA注入 (终值{fin_dca/10000:.0f}万)',zorder=3)
ax.plot(ndf_dca['date'],ndf_dca['cost']/10000,color='#888888',lw=1.0,ls='--',alpha=0.7,
        label=f'DCA成本线 (总投入{total_dca/10000:.0f}万)')
ax.axhline(y=1,color='#888888',lw=0.8,ls=':',alpha=0.4)

ax.fill_between(ndf_dca['date'],ndf_dca['cost']/10000,ndf_dca['nav']/10000,
                where=ndf_dca['nav']>=ndf_dca['cost'],color='#E74C3C',alpha=0.08)
ax.fill_between(ndf_dca['date'],ndf_dca['cost']/10000,ndf_dca['nav']/10000,
                where=ndf_dca['nav']<ndf_dca['cost'],color='#888888',alpha=0.08)

ax.set_ylabel('净值 (万元)',fontsize=13,fontweight='bold')
ax.legend(loc='upper left',fontsize=11,framealpha=0.9)
ax.grid(True,alpha=0.15)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{x:.0f}万'))

ax.annotate(f'v18: {fin18/10000:.0f}万\n年化{ann18:+.1f}% 夏普{sr18:.2f} 回撤{mdd18:+.1f}%',
    xy=(ndf18['date'].iloc[-1],ndf18['nav'].iloc[-1]/10000),
    xytext=(-160,-30),textcoords='offset points',fontsize=10,color='#2ECC71',fontweight='bold')
ax.annotate(f'v18+DCA: {fin_dca/10000:.0f}万\n总投入{total_dca/10000:.0f}万 净利{(fin_dca-total_dca)/10000:.0f}万',
    xy=(ndf_dca['date'].iloc[-1],ndf_dca['nav'].iloc[-1]/10000),
    xytext=(-160,-50),textcoords='offset points',fontsize=10,color='#E74C3C',fontweight='bold')

# 下半部分: 净值差
ax2=axes[1]
diff=(ndf_dca['nav']-ndf18['nav'])/10000
pos=diff>=0;neg=diff<0
ax2.fill_between(ndf18['date'],0,diff,where=pos,color='#E74C3C',alpha=0.4,label='DCA领先')
ax2.fill_between(ndf18['date'],0,diff,where=neg,color='#2ECC71',alpha=0.4,label='v18领先')
ax2.plot(ndf18['date'],diff,color='#333333',lw=0.8)
ax2.axhline(y=0,color='#888888',lw=0.5,ls='-')
ax2.set_ylabel('净值差\n(万元)',fontsize=11,fontweight='bold')
ax2.set_xlabel('日期',fontsize=12,fontweight='bold')
ax2.grid(True,alpha=0.1)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{x:+.0f}万'))
ax2.legend(loc='upper left',fontsize=10)

final_diff=diff.iloc[-1]
ax2.annotate(f'Δ {final_diff:+.0f}万',
    xy=(ndf18['date'].iloc[-1],final_diff),
    xytext=(-100,15),textcoords='offset points',fontsize=11,fontweight='bold',
    color='#E74C3C' if final_diff>0 else '#2ECC71')

fig.suptitle('YH_ultra v18 (TS=7%): 净值曲线对比 — 加不加DCA每月2万注入',fontsize=16,fontweight='bold',y=0.98)
fig.autofmt_xdate()

out=os.path.join(SCRIPT,'nav_dca_v18.png')
fig.savefig(out,dpi=150,bbox_inches='tight',facecolor='white')
plt.close(fig)

print(f"\n{'='*70}")
print(f"  v18纯 (TS=7%):")
print(f"    终值{fin18/10000:.0f}万  年化{ann18:+.1f}%  夏普{sr18:.2f}  回撤{mdd18:+.1f}%")
print(f"    交易{len(trades18)}笔  胜率{wr18:.0f}%")
print(f"  v18+DCA注入:")
print(f"    终值{fin_dca/10000:.0f}万  总投入{total_dca/10000:.0f}万  净利{(fin_dca-total_dca)/10000:.0f}万")
print(f"    交易{len(trades_dca)}笔  胜率{wr_dca:.0f}%")
print(f"    表观年化{ann_dca_apparent:+.1f}%  夏普{sr_dca:.2f}  回撤{mdd_dca:+.1f}%")
print(f"  DCA增量ROI: {(fin_dca-total_dca)/(total_dca-INIT)*100:+.0f}%")
print(f"\n  图表: {out}")
