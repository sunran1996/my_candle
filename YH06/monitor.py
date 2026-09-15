# -*- coding: utf-8 -*-
"""
YH04  YH02红利低波主线 + MACD成长指数轮动副线
用法: python monitor.py                     → 实时信号
      python monitor.py --from 2020-01-01   → 回测
      python monitor.py --from 2020-01-01 --dca 2  → 回测+DCA
"""
import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import akshare as ak, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

_fonts=[f.name for f in fm.fontManager.ttflist]
CN='WenQuanYi Zen Hei' if 'WenQuanYi Zen Hei' in _fonts else ('SimHei'if'SimHei'in _fonts else'DejaVu Sans')
plt.rcParams['font.sans-serif']=[CN]; plt.rcParams['axes.unicode_minus']=False

# ======================== 参数 ========================
INIT=1_000_000; COMM=0.0003; SLIP=0.0001; DCA=0
MOM=10; REBAL=5; TRAIL=0.10
BB_P=45; BB_S=2.0; RSI_P=14; RSI_L=30; RSI_H=70; ERS=65
SCRIPT=os.path.dirname(os.path.abspath(__file__))

MAIN_SYM='sh512890'; MAIN_NAME='红利低波'
GROWTH={'创业板':'sz159915','科创50':'sh588000','人工智能':'sh515070','半导体':'sh512480'}

# ================================================================
def fetch():
    dfs={}
    for n,s in {**GROWTH,MAIN_NAME:MAIN_SYM}.items():
        df=ak.fund_etf_hist_sina(symbol=s); df['date']=pd.to_datetime(df['date'])
        dfs[n]=df[['date','open','high','low','close','volume']].sort_values('date').reset_index(drop=True)
    return dfs

def add_main(df):
    df=df.copy(); r=df['close'].pct_change().fillna(0); r[abs(r)>0.1]=0
    df['adj']=(1+r).cumprod()
    # BB用close(与daily_signal一致,信号更灵敏)
    df['bb_ma']=df['close'].rolling(BB_P).mean(); df['bb_std']=df['close'].rolling(BB_P).std()
    df['up']=df['bb_ma']+BB_S*df['bb_std']; df['lo']=df['bb_ma']-BB_S*df['bb_std']
    d=df['adj'].diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/RSI_P,adjust=False).mean()/l.ewm(alpha=1/RSI_P,adjust=False).mean().replace(0,np.nan))
    return df

def main_signal(row,pbw):
    price,rsi,up,lo=row['close'],row['rsi'],row['up'],row['lo']
    if pd.isna(lo)or pd.isna(rsi): return'HOLD',pbw
    bw=(up-lo)/row['bb_ma']if row['bb_ma']>0 else 0.1
    exp=(pbw is not None and bw>pbw); nbw=bw
    bb_buy=(price<=lo); bb_sell=(price>=up); rsi_buy=(rsi<=RSI_L)
    if exp:
        sell_ok=(bb_sell and rsi>=ERS); buy_ok=(bb_buy or rsi_buy)
    else: buy_ok=(bb_buy or rsi_buy); sell_ok=(bb_sell or rsi>=RSI_H)
    if buy_ok: return'BUY',nbw
    elif sell_ok: return'SELL',nbw
    else: return'HOLD',nbw

def add_growth(df):
    df=df.copy(); df['mom']=df['close']/df['close'].shift(10)-1
    e10=df['close'].ewm(span=10,adjust=False).mean(); e20=df['close'].ewm(span=20,adjust=False).mean()
    df['macd_h']=e10-e20-(e10-e20).ewm(span=7,adjust=False).mean()
    df['macd_line']=e10-e20; df['ma20']=df['close'].rolling(20).mean()
    return df

def rank_growth(dfs,date):
    """按date在各成长ETF中查找对应行,按macd_h排名"""
    s={}
    for n in GROWTH:
        idxs=dfs[n][dfs[n]['date']==date].index
        if len(idxs)==0: continue
        pos=idxs[0]
        if pos<30: continue
        m=dfs[n]['macd_h'].iloc[pos]
        if pd.isna(m): continue
        s[n]=m
    return sorted(s,key=s.get,reverse=True)

# ================================================================
def run_backtest(start_str):
    start=pd.Timestamp(start_str)
    print("获取数据...")
    raw=fetch()
    df_main=add_main(raw[MAIN_NAME])
    dfs_g={n:add_growth(d) for n,d in raw.items() if n!=MAIN_NAME}
    dates=sorted(set.intersection(*[set(d['date'])for d in raw.values()]))
    dates=[d for d in dates if d>=start]
    if len(dates)<60: return

    ALL=list(GROWTH.keys())+[MAIN_NAME]; sp=raw[MAIN_NAME][raw[MAIN_NAME]['date']>=start]['close'].iloc[0]
    cash=0.0; total_inv=INIT
    shares={n:0.0 for n in ALL}; shares[MAIN_NAME]=INIT/sp*(1-COMM-SLIP)
    pos=MAIN_NAME; peak=INIT; navs=[]; trades=[]
    pbw=None; ep=sp; hse=0; last_rb=None; last_month=None; stopped=False

    for date in dates:
        ym=(date.year,date.month)
        if DCA>0 and last_month and ym!=last_month: cash+=DCA; total_inv+=DCA
        last_month=ym
        px={n:raw[n][raw[n]['date']==date]['close'].values[0] for n in ALL if len(raw[n][raw[n]['date']==date])>0}
        nav=cash+sum(shares[n]*px.get(n,0)for n in ALL)
        if nav>peak: peak=nav
        mr=df_main[df_main['date']==date]
        if len(mr)==0: navs.append({'date':date,'nav':nav,'pos':pos}); continue
        sig,pbw=main_signal(mr.iloc[0],pbw); mp=px.get(MAIN_NAME,0)

        if sig=='BUY'and pos!=MAIN_NAME:
            if pos:
                for n in GROWTH:
                    if shares[n]>0 and n in px: cash+=shares[n]*px[n]*(1-COMM-SLIP); shares[n]=0.0
                trades.append({'date':date,'dir':'SELL','name':pos,'price':px.get(pos,0),'nav':nav})
            if mp>0: val=min(cash,nav); shares[MAIN_NAME]+=val/mp*(1-COMM-SLIP); cash-=val
            trades.append({'date':date,'dir':'BUY','name':MAIN_NAME,'price':mp,'nav':nav}); pos=MAIN_NAME; ep=mp; stopped=False

        elif sig=='SELL'and pos==MAIN_NAME:
            if shares[MAIN_NAME]>0 and mp>0: cash+=shares[MAIN_NAME]*mp*(1-COMM-SLIP); shares[MAIN_NAME]=0.0
            trades.append({'date':date,'dir':'SELL','name':MAIN_NAME,'price':mp,'nav':nav}); pos=None

        elif DCA>0 and sig=='HOLD'and pos==MAIN_NAME and cash>100 and mp>0:
            shares[MAIN_NAME]+=cash/mp*(1-COMM-SLIP); cash=0

        elif sig=='HOLD'and pos!=MAIN_NAME:
            if pos and pos in px:
                if px[pos]>hse: hse=px[pos]
                if px[pos]<hse*(1-TRAIL):
                    cash+=shares[pos]*px[pos]*(1-COMM-SLIP); shares[pos]=0.0
                    trades.append({'date':date,'dir':'STOP','name':pos,'price':px[pos],'nav':nav}); pos=None; stopped=True
            if stopped: navs.append({'date':date,'nav':nav,'pos':pos}); continue
            days=(date-last_rb).days if last_rb else 999
            if days>=REBAL and pos is None:
                ranking=rank_growth(dfs_g,date)
                if ranking:
                    tgt=ranking[0]
                    tgt_idxs=dfs_g[tgt][dfs_g[tgt]['date']==date].index
                    if len(tgt_idxs)>0:
                        tgt_row=dfs_g[tgt].iloc[tgt_idxs[0]]
                        macd_ok=tgt_row['macd_h']>0
                        bull_ok=not pd.isna(tgt_row['ma20']) and px.get(tgt,0)>tgt_row['ma20']
                        if macd_ok and bull_ok:
                            if tgt in px:
                                val=min(cash,nav)
                                if val>100: shares[tgt]=val/px[tgt]*(1-COMM-SLIP); cash-=val
                                trades.append({'date':date,'dir':'BUY','name':tgt,'price':px[tgt],'nav':nav}); pos=tgt; hse=px[tgt]
                            last_rb=date
        navs.append({'date':date,'nav':nav,'pos':pos})

    # 统计
    ndf=pd.DataFrame(navs); final=ndf['nav'].iloc[-1]
    ret=(final/(total_inv if DCA>0 else INIT)-1)*100
    ann=((1+ret/100)**(252/len(ndf))-1)*100
    dr=ndf['nav'].pct_change().dropna(); sr=(ann/100-0.02)/(dr.std()*np.sqrt(252))if dr.std()>0 else 0
    mdd=((ndf['nav']/INIT-(ndf['nav']/INIT).cummax())/(ndf['nav']/INIT).cummax()).min()*100
    td=pd.DataFrame(trades); n_trades=len(td)

    dca_label=f' DCA月投{DCA/1e4:.0f}万' if DCA>0 else''
    print(f"\n  YH04  YH02主线+成长副线{dca_label}  |  动量{MOM}日  止损{TRAIL*100:.0f}%")
    print(f"  Return: {ret:+.2f}%  Annual: {ann:+.2f}%  Sharpe: {sr:.3f}  MaxDD: {mdd:+.2f}%")
    if DCA>0: print(f"  交易: {n_trades}笔  投入{total_inv/1e4:.0f}万  终值{final/1e4:.1f}万  净赚{final-total_inv:,.0f}")
    else: print(f"  交易: {n_trades}笔  终值{final:,.0f}")

    # 交易明细
    if n_trades>0:
        print(f"  标的分布: {td['name'].value_counts().to_dict()}")
        print(f"\n  {'日期':<12} {'标的':<8} {'方向':<6} {'价格':>8} {'盈亏':>8} {'总市值':>12}")
        epx={}
        for _,t in td.iterrows():
            nm=t['name']; d=t['date'].strftime('%Y-%m-%d')[:10]; pv=t.get('price',0); nv=t.get('nav',0)
            if t['dir']=='BUY': epx[nm]=pv; print(f'  {d:<12} {nm:<8} 买入   {pv:>8.3f}  {"—":>8}  {nv:>12,.0f}')
            else:
                ep=epx.pop(nm,pv); pnl=(pv/ep-1)*100 if ep>0 else 0
                print(f'  {d:<12} {nm:<8} {t["dir"]:<6} {pv:>8.3f}  {pnl:>+7.1f}%  {nv:>12,.0f}')
        # 保存交易明细CSV
        td_save=td.copy(); td_save['date']=td_save['date'].dt.strftime('%Y-%m-%d')
        csv_path=os.path.join(SCRIPT,'trades.csv')
        td_save.to_csv(csv_path,index=False,encoding='utf-8-sig')
        print(f"\n  交易明细已保存: {csv_path}")

    # 逐年
    ndf['year']=ndf['date'].dt.year
    ys=[]; yrets=[]
    print(f"\n  {'年份':<6} {'收益':>8} {'MaxDD':>8}")
    for yr,grp in ndf.groupby('year'):
        if len(grp)<10: continue
        yr_ret=(grp['nav'].iloc[-1]/grp['nav'].iloc[0]-1)*100
        yr_mdd=((grp['nav']/grp['nav'].iloc[0]-(grp['nav']/grp['nav'].iloc[0]).cummax())/(grp['nav']/grp['nav'].iloc[0]).cummax()).min()*100
        print(f"  {yr:<6} {yr_ret:>+7.1f}% {yr_mdd:>+7.1f}%")
        ys.append(yr); yrets.append(yr_ret)

    # ===== mplfinance图表 =====
    if len(ndf)>1:
        import mplfinance as mpf
        cn_c=mpf.make_marketcolors(up='#CC0000',down='#008800',edge='inherit',wick='inherit',volume='inherit')
        cn_s=mpf.make_mpf_style(marketcolors=cn_c,gridstyle='',rc={'font.sans-serif':[CN,'DejaVu Sans'],'axes.unicode_minus':False})
        # 只画已交易的: 红利低波+创业板
        fig=plt.figure(figsize=(8,16),facecolor='white')
        gs=fig.add_gridspec(4,1,height_ratios=[2,2,2,1.2],hspace=0.15,left=0.08,right=0.92,top=0.97,bottom=0.03)
        for pi,(name,color) in enumerate([('红利低波','#9B59B6'),('创业板','#E74C3C')]):
            df_k=raw[name]; df_k=df_k[df_k['date']>=start]
            df_k=df_k.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            df_k=df_k.set_index('date')[['Open','High','Low','Close','Volume']]
            ax=fig.add_subplot(gs[pi]); mpf.plot(df_k.tail(500),type='candle',ax=ax,volume=False,style=cn_s)
            ax.set_title(name,fontsize=13,loc='left',color=color,fontweight='bold'); ax.tick_params(labelsize=9)
        # 净值
        ax2=fig.add_subplot(gs[2])
        ax2.plot(ndf['date'],ndf['nav']/INIT,color='#CC2222',lw=2.0); ax2.axhline(y=1.0,color='#888',lw=0.8,ls='--')
        ax2.set_title(f'策略净值 {ret:+.1f}%  夏普{sr:.3f}  回撤{mdd:.1f}%',fontsize=12,loc='left',fontweight='bold')
        ax2.tick_params(labelsize=9); ax2.grid(True,alpha=0.12)
        # 年度收益
        ax3=fig.add_subplot(gs[3])
        colors2=['#CC2222'if r>=0 else'#228B22'for r in yrets]
        bars=ax3.bar(range(len(ys)),yrets,color=colors2,alpha=0.85)
        for bar,val in zip(bars,yrets):
            ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+(1 if val>=0 else-2),
                    f'{val:+.1f}%',ha='center',fontsize=12,fontweight='bold')
        ax3.axhline(y=0,color='black',lw=1); ax3.set_xticks(range(len(ys)))
        ax3.set_xticklabels([str(y)for y in ys],fontsize=12); ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%+.0f%%'))
        ax3.set_title('年度收益',fontsize=12,loc='left',fontweight='bold'); ax3.grid(True,alpha=0.12,axis='y')
        fig.suptitle(f'YH04  全区间回测{dca_label}\n{ret:+.1f}% | {n_trades}笔',fontsize=15,fontweight='bold',y=0.99)
        plt.savefig(os.path.join(SCRIPT,'backtest_chart.png'),dpi=120,bbox_inches='tight',facecolor='white'); plt.close()
        print(f'  图表: {SCRIPT}/backtest_chart.png')
        ndf[['date','nav']].to_csv(os.path.join(SCRIPT,'_nav_yh06.csv'), index=False)
        print(f'  净值导出: {SCRIPT}/_nav_yh06.csv')

def live_signal():
    print("获取数据...")
    raw=fetch(); df_main=add_main(raw[MAIN_NAME])
    dfs_g={n:add_growth(d) for n,d in raw.items() if n!=MAIN_NAME}
    idx=-1; row=df_main.iloc[idx]; date=row['date']
    price=row['close']; rsi=row['rsi']; lo=row['lo']; up=row['up']
    bb_pos=(price-lo)/(up-lo)*100 if up>lo else 50
    bb_buy=price<=lo; bb_sell=price>=up; rsi_buy=rsi<=RSI_L
    buy_ok=bb_buy or rsi_buy; sell_ok=(bb_sell and rsi>=ERS)

    g_idx=max(len(d)for d in dfs_g.values())-1
    scores={}
    for n in GROWTH:
        p2=min(g_idx,len(dfs_g[n])-1); v=dfs_g[n]['macd_h'].iloc[p2]
        if not pd.isna(v): scores[n]=v
    ranking=sorted(scores,key=scores.get,reverse=True); leader=ranking[0]if ranking else'—'

    if buy_ok: sig='买入'; advice=f'全仓{MAIN_NAME}'
    elif sell_ok and scores.get(leader,0)>0: sig='换仓'; advice=f'{leader}'
    elif sell_ok: sig='持币'; advice='MACD全负,等翻红'
    else: sig='持有'; advice=f'{MAIN_NAME}@{price:.3f}'

    print(f"\n{'='*50}")
    print(f"  YH04  {date.strftime('%Y-%m-%d')}  红利低波: {sig}")
    print(f"  价格{price:.3f} RSI{rsi:.0f} BB{bb_pos:.0f}%")
    print(f"  建议: {advice}")
    print(f"  副线 MACD: {' > '.join(f'{n}({scores[n]:+.3f})'for n in ranking[:4])}")
    print(f"{'='*50}")

def main():
    import argparse; p=argparse.ArgumentParser()
    p.add_argument('--from',dest='fr',type=str,default=None)
    p.add_argument('--dca',dest='dca',type=float,default=0)
    a=p.parse_args()
    global DCA; DCA=a.dca*10000
    if a.fr: run_backtest(a.fr)
    else: live_signal()

if __name__=='__main__': main()
