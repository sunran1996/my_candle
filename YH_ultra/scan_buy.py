# -*- coding: utf-8 -*-
"""每只股票扫描最优买点(RSI+BB), 画K线买卖点"""
import sys,io,os,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak,pandas as pd,numpy as np
import matplotlib; matplotlib.use('Agg')
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

_fonts=[f.name for f in fm.fontManager.ttflist]
CN='WenQuanYi Zen Hei' if 'WenQuanYi Zen Hei' in _fonts else ('SimHei' if 'SimHei' in _fonts else 'DejaVu Sans')
plt.rcParams['font.sans-serif']=[CN];plt.rcParams['axes.unicode_minus']=False

STOCKS={'山东高速':'sh600350','渝农商行':'sh601077','皖通高速':'sh600012','江苏银行':'sh600919'}
INIT=1_000_000;COMM=0.0003;SLIP=0.0001;MAX_POS=0.25

SCRIPT=os.path.dirname(os.path.abspath(__file__))

def fetch():
    dfs={}
    for n,s in STOCKS.items():
        df=ak.stock_zh_a_daily(symbol=s,adjust='qfq');df['date']=pd.to_datetime(df['date'])
        dfs[n]=df[['date','open','high','low','close','volume']].sort_values('date').reset_index(drop=True)
    return dfs

RAW_CACHE=None
def get_data():
    global RAW_CACHE
    if RAW_CACHE is None:
        RAW_CACHE=fetch()
    return RAW_CACHE

def add_indicators(df):
    df=df.copy();c=df['close']
    df['ma20']=c.rolling(20).mean();df['ma60']=c.rolling(60).mean()
    df['bb_ma']=c.rolling(20).mean();df['bb_std']=c.rolling(20).std()
    df['bb_up']=df['bb_ma']+2*df['bb_std'];df['bb_lo']=df['bb_ma']-2*df['bb_std']
    d=c.diff();g=d.clip(lower=0);l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/14,adjust=False).mean()/l.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan))
    return df

def make_buy(rsi_th, bb_th):
    def fn(row,strict):
        if pd.isna(row['bb_lo'])or pd.isna(row['rsi']):return False,0
        rsi=row['rsi'];c=row['close'];lo=row['bb_lo'];up=row['bb_up']
        if up<=lo:return False,0
        dist=(c-lo)/(up-lo)
        sc=(1 if rsi<=rsi_th else 0)+(1 if dist<=bb_th else 0)
        if rsi<=30:sc+=1
        return sc>=1,sc
    return fn

def sell_fn(row,pnl,dd,pbw,tp=0.20):
    if pnl<=-0.10:return True,''
    if dd<=-0.08:return True,''
    if pnl>=tp:return True,''
    return False,''

# ---- 单只股票回测 ----
def run_solo(name, rsi_th, bb_th, tp=0.20):
    raw=get_data();df=add_indicators(raw[name])
    dates=[d for d in df['date'] if d>=pd.Timestamp('2019-01-01')]
    cash=INIT;shares=0;entry=0;high=0;navs=[]
    buy_fn=make_buy(rsi_th,bb_th)
    trades=[]

    for date in dates:
        r=df[df['date']==date]
        if len(r)==0:continue
        cp=r['close'].iloc[0];row=r.iloc[0]
        if shares>0:
            if cp>high:high=cp
            pnl=cp/entry-1;dd=cp/high-1
            do,_=sell_fn(row,pnl,dd,None,tp)
            if do:
                cash+=shares*cp*(1-COMM-SLIP)
                trades.append({'date':date,'dir':'SELL','price':cp,'pnl':pnl*100})
                shares=0;entry=0;high=0
        if shares==0:
            ok,sc=buy_fn(row,False)
            if ok:
                val=min(cash,INIT*MAX_POS)
                if val>5000:
                    shares=val/cp*(1-COMM-SLIP);cash-=val
                    entry=cp;high=cp
                    trades.append({'date':date,'dir':'BUY','price':cp,'pnl':0})
        navs.append(cash+shares*cp)

    ndf=pd.DataFrame(navs,columns=['nav']);final=ndf['nav'].iloc[-1]
    ret=(final/INIT-1)*100;ann=((1+ret/100)**(252/len(ndf))-1)*100
    dr=ndf['nav'].pct_change().dropna();vol=dr.std()*np.sqrt(252)*100
    sr=(ann-2)/vol if vol>0 else 0
    mdd=((ndf['nav']-ndf['nav'].cummax())/ndf['nav'].cummax()).min()*100
    sells=[t for t in trades if t['dir']=='SELL' and t['pnl']!=0]
    pnls=np.array([s['pnl'] for s in sells])
    wr=(pnls>0).sum()/len(pnls)*100 if len(pnls)>0 else 0
    aw=pnls[pnls>0].mean()if(pnls>0).sum()>0 else 0
    al=pnls[pnls<0].mean()if(pnls<0).sum()>0 else 0
    return ann,ret,sr,mdd,wr,len(sells),aw,al

# ---- 组合回测(含交易记录) ----
def run_portfolio(buy_params, tp_params):
    """buy_params: {name: (rsi_th, bb_th)}; tp_params: {name: tp}"""
    raw=get_data();dfs={n:add_indicators(d)for n,d in raw.items()}
    dates=sorted(set.intersection(*[set(d['date'])for d in dfs.values()]))
    dates=[d for d in dates if d>=pd.Timestamp('2019-01-01')]
    cash=INIT;shares={n:0.0 for n in STOCKS};entry={n:0.0 for n in STOCKS};high={n:0.0 for n in STOCKS}
    navs=[];buy_marks={n:[]for n in STOCKS};sell_marks={n:[]for n in STOCKS}

    for date in dates:
        px={n:raw[n][raw[n]['date']==date]['close'].iloc[0]for n in STOCKS if len(raw[n][raw[n]['date']==date])>0}

        # 卖出
        for n in STOCKS:
            if shares[n]<=0:continue
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            if cp>high[n]:high[n]=cp
            pnl=cp/entry[n]-1;dd=cp/high[n]-1
            tp=tp_params.get(n,0.20)
            do=False
            if pnl<=-0.10:do=True
            elif dd<=-0.08:do=True
            elif pnl>=tp:do=True
            if do:
                cash+=shares[n]*cp*(1-COMM-SLIP)
                sell_marks[n].append((date,cp,pnl*100))
                shares[n]=0;entry[n]=0;high[n]=0

        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)

        # 买入(无缩放)
        for n in STOCKS:
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            if shares[n]>0:continue
            rsi_th,bb_th=buy_params.get(n,(42,0.25))
            buy_fn=make_buy(rsi_th,bb_th)
            ok,sc=buy_fn(r.iloc[0],False)
            if not ok:continue
            val=min(cash,nav*MAX_POS)
            if val>5000:
                qty=val/cp*(1-COMM-SLIP)
                shares[n]=qty;cash-=val
                entry[n]=cp;high[n]=cp
                buy_marks[n].append((date,cp))

        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)
        navs.append({'date':date,'nav':nav})

    ndf=pd.DataFrame(navs);final=ndf['nav'].iloc[-1]
    ret=(final/INIT-1)*100;ann=((1+ret/100)**(252/len(ndf))-1)*100
    dr=ndf['nav'].pct_change().dropna();vol=dr.std()*np.sqrt(252)*100
    sr=(ann-2)/vol if vol>0 else 0
    mdd=((ndf['nav']-ndf['nav'].cummax())/ndf['nav'].cummax()).min()*100
    total_sells=sum(len(v) for v in sell_marks.values())
    total_buys=sum(len(v) for v in buy_marks.values())
    return ann,ret,sr,mdd,total_buys,total_sells,buy_marks,sell_marks,ndf

# ============ 扫描 ============
print("="*80)
print("每只股票RSI×BB买点扫描 (2019起, 无缩放, TP=20%)")
print("="*80)

RSI_RANGE=[30,35,38,40,42,45,48,50]
BB_RANGE=[0.10,0.12,0.15,0.18,0.20,0.22,0.25,0.30]

best_params={}
for name,sym in STOCKS.items():
    print(f"\n{'─'*70}")
    print(f"  {name} ({sym})")
    # 打印表头: 行为BB 列为RSI
    header=f"  {'BB\\RSI':<8}"
    for rsi in RSI_RANGE:
        header+=f"{rsi:>6}"
    print(header)
    results={}
    best_sr=-99;best_combo=(0,0)
    for bb in BB_RANGE:
        line=f"  {bb:<8.2f}"
        for rsi in RSI_RANGE:
            ann,ret,sr,mdd,wr,nt,aw,al=run_solo(name,rsi,bb)
            results[(rsi,bb)]=sr
            line+=f"{sr:>+5.2f}"
            if sr>best_sr:best_sr=sr;best_combo=(rsi,bb)
        print(line)
    best_params[name]=best_combo
    print(f"  → 最优: RSI<={best_combo[0]}, BB<={best_combo[1]:.2f} (SR={best_sr:.2f})")

# ---- 组合回测 ----
print(f"\n{'='*80}")
print("组合回测对比")
print(f"{'='*80}")

# 统一参数
uni_params={n:(42,0.25) for n in STOCKS}
uni_tp={n:0.20 for n in STOCKS}
ann_u,ret_u,sr_u,mdd_u,bu_u,se_u,_,_,_=run_portfolio(uni_params,uni_tp)
print(f"\n统一参数(RSI42/BB0.25):  年化{ann_u:+.1f}% 夏普{sr_u:.2f} 回撤{mdd_u:+.1f}% 交易{se_u}")

# 各自最优买点
opt_params=best_params
opt_tp={n:0.20 for n in STOCKS}
ann_o,ret_o,sr_o,mdd_o,bu_o,se_o,buys,sells,nav_df=run_portfolio(opt_params,opt_tp)
print(f"各自最优买点:             年化{ann_o:+.1f}% 夏普{sr_o:.2f} 回撤{mdd_o:+.1f}% 交易{se_o}")
print(f"配置: ",{n:f'RSI<={v[0]} BB<={v[1]:.2f}' for n,v in best_params.items()})

# ============ 画K线图 ============
print(f"\n{'='*80}")
print("生成K线买卖点图表...")

raw=get_data()
# 统一基准的交易点
_,_,_,_,_,_,buys_u,sells_u,_=run_portfolio(uni_params,uni_tp)
# 各自最优的交易点
_,_,_,_,_,_,buys_o,sells_o,_=run_portfolio(opt_params,opt_tp)

RED='#CC0000';GREEN='#008800';PURPLE='#9B59B6';ORANGE='#E67E22';BLUE='#3498DB'

fig=plt.figure(figsize=(24,18),facecolor='white')
gs=fig.add_gridspec(4,1,height_ratios=[1,1,1,1],hspace=0.3,top=0.96,bottom=0.04,left=0.04,right=0.97)

cn_c=mpf.make_marketcolors(up=RED,down=GREEN,edge='inherit',wick='inherit',volume='inherit')
cn_s=mpf.make_mpf_style(marketcolors=cn_c,gridstyle='',rc={'font.sans-serif':[CN],'axes.unicode_minus':False})

for idx,name in enumerate(STOCKS):
    ax=fig.add_subplot(gs[idx])
    df_raw=raw[name].copy()
    df_raw=df_raw.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
    df_raw=df_raw.set_index('date')

    # K线 (显示近3年)
    start=pd.Timestamp.now()-pd.DateOffset(years=3)
    ohlc=df_raw[df_raw.index>=start][['Open','High','Low','Close','Volume']]
    mpf.plot(ohlc,type='candle',ax=ax,volume=False,style=cn_s)

    # BB线
    df_ind=add_indicators(raw[name].set_index('date'))
    x=list(range(len(ohlc)))
    bb_up=df_ind['bb_up'].values[-len(ohlc):]
    bb_lo=df_ind['bb_lo'].values[-len(ohlc):]
    ax.plot(x,bb_up,color=PURPLE,lw=0.6,ls='--',alpha=0.4)
    ax.plot(x,bb_lo,color=PURPLE,lw=0.6,ls='--',alpha=0.4)

    # 统一参数的买卖点(三角)
    for d,px in buys_u.get(name,[]):
        if d>=start:
            xi=list(ohlc.index).index(d) if d in ohlc.index else None
            if xi is not None:
                ax.scatter(xi,ohlc['Low'].iloc[xi]*0.98,marker='^',c=RED,s=40,zorder=5)
    for d,px,pnl in sells_u.get(name,[]):
        if d>=start:
            xi=list(ohlc.index).index(d) if d in ohlc.index else None
            if xi is not None:
                ax.scatter(xi,ohlc['High'].iloc[xi]*1.02,marker='v',c=GREEN,s=40,zorder=5)

    # 标注最优参数
    rsi_th,bb_th=best_params[name]
    rsi_val=df_ind['rsi'].iloc[-1]
    c=df_ind['close'].iloc[-1]
    bb_up_v=df_ind['bb_up'].iloc[-1];bb_lo_v=df_ind['bb_lo'].iloc[-1]
    bb_pos=(c-bb_lo_v)/(bb_up_v-bb_lo_v)*100 if bb_up_v>bb_lo_v else 50
    ax.set_title(f'{name} {c:.2f} | 最优: RSI≤{rsi_th} BB≤{bb_th:.2f} | 当前: RSI{rsi_val:.0f} BB{bb_pos:.0f}%',
                 fontsize=11,fontweight='bold')
    ax.tick_params(labelsize=8);ax.grid(True,alpha=0.1)
    ax.legend(['BB上下轨'],loc='upper left',fontsize=7)

fig.suptitle('YH_ultra 个股买卖点 (▲买 ▼卖)',fontsize=14,fontweight='bold',y=0.99)
out_path=os.path.join(SCRIPT,'buy_sell_chart.png')
fig.savefig(out_path,dpi=150,bbox_inches='tight',facecolor='white')
plt.close(fig)
print(f"图表: {out_path}")
print("完成!")
