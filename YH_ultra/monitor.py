# -*- coding: utf-8 -*-
"""
YH_ultra v7  YH02红利主线 + 创业板/科创50双副线(动量排名选最强)
等信号入场, 主线BUY无视冷却, 五重过滤
"""
import sys,io,os,warnings;sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak,pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt,matplotlib.font_manager as fm
warnings.filterwarnings('ignore')
_fonts=[f.name for f in fm.fontManager.ttflist]
CN='WenQuanYi Zen Hei'if'WenQuanYi Zen Hei'in _fonts else('SimHei'if'SimHei'in _fonts else'DejaVu Sans')
plt.rcParams['font.sans-serif']=[CN];plt.rcParams['axes.unicode_minus']=False

INIT=1_000_000;COMM=0.0003;SLIP=0.0001;DCA=0
BB_P=45;BB_S=2.0;RSI_P=14;RSI_L=30;RSI_H=70;ERS=65;BA=0.001
HARD_STOP=0.12;NAV_STOP=0.13;TRAIL=0.10;MOM=10
SCRIPT=os.path.dirname(os.path.abspath(__file__))

MAIN_SYM='sh512890';MAIN_NAME='红利低波'
GROWTH={'创业板':'sz159915','科创50':'sh588000'}
ALL_NAMES=[MAIN_NAME]+list(GROWTH.keys());ALL_SYMS={MAIN_NAME:MAIN_SYM,**GROWTH}

def fetch():
    dfs={}
    for n,s in ALL_SYMS.items():
        df=ak.fund_etf_hist_sina(symbol=s);df['date']=pd.to_datetime(df['date'])
        dfs[n]=df[['date','close']].sort_values('date').reset_index(drop=True)
    return dfs

def add_main(df):
    df=df.copy();r=df['close'].pct_change().fillna(0);r[abs(r)>0.1]=0;df['adj']=(1+r).cumprod()
    df['ma']=df['adj'].rolling(BB_P).mean();df['std']=df['adj'].rolling(BB_P).std()
    df['up']=df['ma']+BB_S*df['std'];df['lo']=df['ma']-BB_S*df['std']
    df['ua']=df['up'].diff().diff().rolling(3,min_periods=1).mean()
    df['pa']=df['adj'].diff().diff().rolling(3,min_periods=1).mean()
    d=df['adj'].diff();g=d.clip(lower=0);l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/RSI_P,adjust=False).mean()/l.ewm(alpha=1/RSI_P,adjust=False).mean().replace(0,np.nan))
    df['mom20']=df['close']/df['close'].shift(20)-1;return df

def add_growth(df):
    df=df.copy();df['mom']=df['close']/df['close'].shift(MOM)-1
    e10=df['close'].ewm(span=10,adjust=False).mean();e20=df['close'].ewm(span=20,adjust=False).mean()
    df['macd']=e10-e20;df['macd_s']=df['macd'].ewm(span=7,adjust=False).mean()
    df['macd_h']=df['macd']-df['macd_s'];df['ma20']=df['close'].rolling(20).mean();df['macd_line']=df['macd']
    df['ma120']=df['close'].rolling(120).mean();df['ma10']=df['close'].rolling(10).mean();df['mom20']=df['close']/df['close'].shift(20)-1
    d=df['close'].diff();g=d.clip(lower=0);l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/14,adjust=False).mean()/l.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan))
    return df

def rank_momentum(dfs,date):
    scores=[]
    for n in GROWTH:
        idxs=dfs[n][dfs[n]['date']==date].index
        if len(idxs)==0:continue
        v=dfs[n]['mom'].iloc[idxs[0]]
        if not pd.isna(v):scores.append((n,v))
    scores.sort(key=lambda x:x[1],reverse=True)
    return scores

# ======================== 回测 ========================
def run_backtest(start_str):
    start=pd.Timestamp(start_str)
    print("获取数据...")
    raw=fetch();df_main=add_main(raw[MAIN_NAME]);dfs_g={n:add_growth(d)for n,d in raw.items()if n!=MAIN_NAME}
    dates=sorted(set.intersection(*[set(d['date'])for d in raw.values()]))
    dates=[d for d in dates if d>=start]
    if len(dates)<60:return

    cash=INIT;total_inv=INIT;shares={n:0.0 for n in ALL_NAMES}
    pos=None;peak=INIT;navs=[];trades=[]
    sc=2;pbw=None;ep=0;hse={};lrb=None;last_month=None;stp=False;gb=0;started=False

    for date in dates:
        ym=(date.year,date.month)
        if DCA>0 and last_month and ym!=last_month:cash+=DCA;total_inv+=DCA
        last_month=ym

        px={}
        for n in ALL_NAMES:r=raw[n][raw[n]['date']==date];px[n]=r['close'].iloc[0]if len(r)>0 else 0

        nav=cash+sum(shares[n]*px.get(n,0)for n in ALL_NAMES)
        if nav>peak:peak=nav
        dd_nav=(nav-peak)/peak if peak>0 else 0

        mr=df_main[df_main['date']==date]
        if len(mr)==0:navs.append({'date':date,'nav':nav,'pos':pos});continue
        row=mr.iloc[0]
        adj,rsi,up,lo=row['adj'],row['rsi'],row['up'],row['lo']
        if pd.isna(lo)or pd.isna(rsi):navs.append({'date':date,'nav':nav,'pos':pos});continue
        bw=(up-lo)/row['ma']if row['ma']>0 else 0.1
        ex=(pbw is not None and bw>pbw);nbw=bw
        bb_buy=(adj<=lo);bb_sell=(adj>=up);rsi_buy=(rsi<=RSI_L)
        if ex:raw_sell=(bb_sell and rsi>=ERS);ua=row['ua']if not pd.isna(row['ua'])else 0;pa=row['pa']if not pd.isna(row['pa'])else 0;sell_sig=raw_sell and not((ua>BA)and(pa>0));buy_sig=(bb_buy or rsi_buy)
        else:buy_sig=(bb_buy or rsi_buy);sell_sig=(bb_sell or rsi>=RSI_H)
        sig='BUY'if buy_sig else('SELL'if sell_sig else'HOLD');pbw=nbw;mp=px[MAIN_NAME]

        if not started:
            if buy_sig:shares[MAIN_NAME]=cash/mp*(1-COMM-SLIP);cash=0;pos=MAIN_NAME;ep=mp;started=True;stp=False;gb=max(0,gb-1);trades.append({'date':date,'dir':'BUY','name':MAIN_NAME,'price':mp})
            elif sig=='HOLD':
                mmm=df_main[df_main['date']==date]['mom20'].values[0]if len(df_main[df_main['date']==date])>0 else 0
                ranking=rank_momentum(dfs_g,date)
                for tg,_ in ranking:
                    if tg not in px or not px[tg]:continue
                    gdf=dfs_g[tg];gr=gdf[gdf['date']==date]
                    if len(gr)==0:continue
                    gc=gr['close'].values[0];gma=gr['ma120'].values[0];grs=gr['rsi'].values[0];gm10=gr['ma10'].values[0];gmm=gr['mom20'].values[0]
                    mh=gr['macd_h'].values[0];ml=gr['macd_line'].values[0];m20=gr['ma20'].values[0]
                    ok=mh>0
                    if ok and not pd.isna(gma)and gc<gma:ok=False
                    if ok and not pd.isna(grs):
                        if grs>55:ok=False
                        if grs<25:ok=False
                    if ok and not pd.isna(gm10)and gm10>0:
                        if gc>gm10*1.05:ok=False
                    if ok:
                        az=ml>0;pp=1.0 if az else 0.3;val=min(cash,nav*pp)
                        if val>100:shares[tg]=val/px[tg]*(1-COMM-SLIP);cash-=val;pos=tg;hse[tg]=px[tg];started=True;lrb=date;trades.append({'date':date,'dir':'BUY','name':tg,'price':px[tg]})
                    break
            navs.append({'date':date,'nav':nav,'pos':pos});continue

        # === Normal ===
        if dd_nav<-NAV_STOP and any(shares[n]>0 for n in ALL_NAMES):
            for n in ALL_NAMES:
                if shares[n]>0 and px.get(n,0)>0:cash+=shares[n]*px[n]*(1-COMM-SLIP);shares[n]=0.0
            peak=nav;pos=None;stp=True;gb=2;hse.clear()
            trades.append({'date':date,'dir':'PANIC','name':'ALL','price':0})
            for gn in GROWTH:
                if px.get(gn,0)>0:trades.append({'date':date,'dir':'PANIC','name':gn,'price':px[gn]})
            navs.append({'date':date,'nav':nav,'pos':pos});continue
        if pos==MAIN_NAME and mp>0 and ep>0 and mp<ep*(1-HARD_STOP):cash+=shares[MAIN_NAME]*mp*(1-COMM-SLIP);shares[MAIN_NAME]=0.0;pos=None;trades.append({'date':date,'dir':'STOP','name':MAIN_NAME,'price':mp})
        if buy_sig and pos!=MAIN_NAME:
            for gn in GROWTH:
                if shares.get(gn,0)>0 and px.get(gn,0)>0:cash+=shares[gn]*px[gn]*(1-COMM-SLIP);shares[gn]=0.0;trades.append({'date':date,'dir':'SELL','name':gn,'price':px[gn]})
            hse.clear()
            if mp>0:val=min(cash,nav);shares[MAIN_NAME]+=val/mp*(1-COMM-SLIP);cash-=val;pos=MAIN_NAME;ep=mp;stp=False;gb=max(0,gb-1);trades.append({'date':date,'dir':'BUY','name':MAIN_NAME,'price':mp})
            navs.append({'date':date,'nav':cash+sum(shares[n]*px.get(n,0)for n in ALL_NAMES),'pos':pos});continue
        if buy_sig or sell_sig:sc+=1
        if sc<2:navs.append({'date':date,'nav':nav,'pos':pos});continue
        if sell_sig and pos==MAIN_NAME:
            if shares[MAIN_NAME]>0 and mp>0:cash+=shares[MAIN_NAME]*mp*(1-COMM-SLIP);shares[MAIN_NAME]=0.0;pos=None;trades.append({'date':date,'dir':'SELL','name':MAIN_NAME,'price':mp})
        elif sig=='HOLD'and pos!=MAIN_NAME:
            # trailing stop — 每个副线独立检查
            for gn in GROWTH:
                if shares.get(gn,0)<=0:continue
                p=px.get(gn,0)
                if p<=0:continue
                if p>hse.get(gn,0):hse[gn]=p
                if p<hse.get(gn,0)*(1-TRAIL):
                    cash+=shares[gn]*p*(1-COMM-SLIP);shares[gn]=0.0
                    trades.append({'date':date,'dir':'STOP','name':gn,'price':p})
                    if gn in hse:del hse[gn];stp=True;gb=2
            # 如果所有副线都清掉了, 回到空仓状态
            if not any(shares.get(gn,0)>0 for gn in GROWTH):pos=None
            if shares.get(MAIN_NAME,0)>0:navs.append({'date':date,'nav':nav,'pos':pos});continue
            if gb>0:navs.append({'date':date,'nav':nav,'pos':pos});continue
            if pos is None or pos in GROWTH:
                mmm=df_main[df_main['date']==date]['mom20'].values[0]if len(df_main[df_main['date']==date])>0 else 0
                ranking=rank_momentum(dfs_g,date)
                for tg,_ in ranking:
                    if tg not in px or not px[tg]:continue
                    gdf=dfs_g[tg];gr=gdf[gdf['date']==date]
                    if len(gr)==0:continue
                    gc=gr['close'].values[0];gma=gr['ma120'].values[0];grs=gr['rsi'].values[0];gm10=gr['ma10'].values[0];gmm=gr['mom20'].values[0]
                    mh=gr['macd_h'].values[0];ml=gr['macd_line'].values[0];m20=gr['ma20'].values[0]
                    ok=mh>0  # MACD柱>0
                    if ok and not pd.isna(gma)and gc<gma:ok=False
                    if ok and not pd.isna(grs):
                        if grs>55:ok=False
                        if grs<25:ok=False
                    if ok and not pd.isna(gm10)and gm10>0:
                        if gc>gm10*1.05:ok=False
                    if ok:
                        if pos is None:
                            # 空仓入场
                            az=ml>0;pp=1.0 if az else 0.3;val=min(cash,nav*pp)
                            if val>100:shares[tg]=val/px[tg]*(1-COMM-SLIP);cash-=val;pos=tg;hse[tg]=px[tg];lrb=date;trades.append({'date':date,'dir':'BUY','name':tg,'price':px[tg]})
                        elif shares.get(tg,0)>0:
                            # 已持有动量第一的副线, 不动
                            pass
                        else:
                            # 副线轮换: 从现有副线切一半到动量更强的
                            held=[gn for gn in GROWTH if shares.get(gn,0)>0]
                            if held:
                                # 从持有中价值最大的副线切一半
                                src=max(held,key=lambda gn:shares[gn]*px.get(gn,0))
                                sv=shares[src]*px[src]*0.5
                                cash+=sv*(1-COMM-SLIP);shares[src]*=0.5
                                trades.append({'date':date,'dir':'SELL','name':src,'price':px[src],'note':'半仓轮换'})
                                az=ml>0;pp=1.0 if az else 0.3;val=min(cash,nav*pp)
                                if val>100:
                                    shares[tg]=shares.get(tg,0)+val/px[tg]*(1-COMM-SLIP)
                                    cash-=val;hse[tg]=px[tg]
                                    trades.append({'date':date,'dir':'BUY','name':tg,'price':px[tg]})
                                if shares.get(pos,0)<=0:
                                    alive=[gn for gn in GROWTH if shares.get(gn,0)>0]
                                    pos=alive[0]if alive else None
                        break
        nav=cash+sum(shares[n]*px.get(n,0)for n in ALL_NAMES);navs.append({'date':date,'nav':nav,'pos':pos})

    ndf=pd.DataFrame(navs);final=ndf['nav'].iloc[-1]
    ret=(final/(total_inv if DCA>0 else INIT)-1)*100
    ann=((1+ret/100)**(252/len(ndf))-1)*100
    dr=ndf['nav'].pct_change().dropna();sr=(ann/100-0.02)/(dr.std()*np.sqrt(252))if dr.std()>0 else 0
    mdd=((ndf['nav']/INIT-(ndf['nav']/INIT).cummax())/(ndf['nav']/INIT).cummax()).min()*100
    td=pd.DataFrame(trades)

    dca_label=f' DCA月投{DCA/1e4:.0f}万'if DCA>0 else''
    print(f"\n  YH_ultra v7  红利主线+创业板/科创50双副线(动量排名){dca_label}")
    print(f"  Return: {ret:+.2f}%  Annual: {ann:+.2f}%  Sharpe: {sr:.3f}  MaxDD: {mdd:+.2f}%")
    if DCA>0:print(f"  交易: {len(td)}笔  投入{total_inv/1e4:.0f}万  终值{final/1e4:.1f}万")
    else:print(f"  交易: {len(td)}笔  终值{final:,.0f}")
    if len(td)>0:print(f"  标的分布: {td['name'].value_counts().to_dict()}")

    ndf['year']=ndf['date'].dt.year
    print(f"\n  {'年份':<6} {'收益':>8} {'MaxDD':>8}")
    for yr,grp in ndf.groupby('year'):
        if len(grp)<10:continue
        yr_ret=(grp['nav'].iloc[-1]/grp['nav'].iloc[0]-1)*100
        yr_mdd=((grp['nav']/grp['nav'].iloc[0]-(grp['nav']/grp['nav'].iloc[0]).cummax())/(grp['nav']/grp['nav'].iloc[0]).cummax()).min()*100
        print(f"  {yr:<6} {yr_ret:>+7.1f}% {yr_mdd:>+7.1f}%")

    if len(td)>0:
        print(f"\n  {'日期':<12} {'方向':<6} {'标的':<8} {'价格':>8}")
        for _,t in td.iterrows():
            nm=str(t.get('name','')or'')
            px_v=float(t.get('price',0)or 0)
            print(f"  {t['date'].strftime('%Y-%m-%d'):<12} {str(t['dir']):<6} {nm:<8} {px_v:>8.3f}")

    if len(ndf)>1:
        # 五看板: 统计+红利K线+净值+创业板K线+科创50K线  手机比例
        ohlc_main=raw[MAIN_NAME][raw[MAIN_NAME]['date']>=start][['date','close']].copy()
        ohlc_main['Open']=ohlc_main['close'];ohlc_main['High']=ohlc_main['close'];ohlc_main['Low']=ohlc_main['close']
        ohlc_main=ohlc_main.rename(columns={'close':'Close'}).set_index('date')
        bb_df=df_main[df_main['date']>=start][['date','adj']].set_index('date')

        fig=plt.figure(figsize=(6,12.5),facecolor='#FAFAFA')
        gs=fig.add_gridspec(5,1,height_ratios=[0.5,2.2,1.2,1.3,1.3],hspace=0.35,left=0.06,right=0.94,top=0.98,bottom=0.03)

        # P0: 统计
        ax0=fig.add_subplot(gs[0]);ax0.axis('off');ax0.set_ylim(0,4)
        ax0.text(0,3.5,f'YH_ultra v7  红利+创业板/科创50双副线',fontsize=13,fontweight='bold',color='#1A1A1A')
        ax0.text(0,2.2,f'累计{ret:+.1f}%  年化{ann:+.1f}%  夏普{sr:.2f}  回撤{mdd:+.1f}%  {len(td)}笔',fontsize=9,color='#555')
        n_hl=td[td['name']==MAIN_NAME].shape[0]if len(td)>0 else 0
        n_cy=td[td['name']=='创业板'].shape[0]if len(td)>0 else 0
        n_kc=td[td['name']=='科创50'].shape[0]if len(td)>0 else 0
        ax0.text(0,1.0,f'红利{n_hl}笔  创业板{n_cy}笔  科创50{n_kc}笔',fontsize=9,color='#888')

        # P1: 红利低波 K线+BB
        ax1=fig.add_subplot(gs[1])
        od=ohlc_main.index;av=bb_df['adj'].reindex(od).values
        bm=bb_df['adj'].rolling(45).mean().reindex(od).values
        bu=bm+2*bb_df['adj'].rolling(45).std().reindex(od).values
        bl=bm-2*bb_df['adj'].rolling(45).std().reindex(od).values
        ax1.fill_between(range(len(od)),bu,bl,alpha=0.05,color='#3498db')
        ax1.plot(range(len(od)),av,color='#333',lw=0.8)
        ax1.plot(range(len(od)),bu,color='#e74c3c',lw=0.5,ls='--',alpha=0.6)
        ax1.plot(range(len(od)),bl,color='#27ae60',lw=0.5,ls='--',alpha=0.6)
        ax1.plot(range(len(od)),bm,color='#bdc3c7',lw=0.5,alpha=0.4)
        # 红利交易点
        for _,t in td.iterrows():
            nm=str(t.get('name','')or'')
            if nm!=MAIN_NAME:continue
            td2=pd.Timestamp(t['date']).date()
            for j,dv in enumerate(od):
                if pd.Timestamp(dv).date()==td2:
                    yv=av[j]
                    if t['dir']=='BUY':ax1.scatter(j,yv,color='#c0392b',s=50,marker='^',zorder=5,edgecolors='white',lw=1)
                    elif t['dir']in('SELL','STOP'):ax1.scatter(j,yv,color='#27ae60',s=50,marker='v',zorder=5,edgecolors='white',lw=1)
                    elif t['dir']=='PANIC':ax1.scatter(j,yv,color='#000',s=60,marker='X',zorder=5,edgecolors='white',lw=1.5);break
        ax1.set_title('红利低波 + BB(45,2)',fontsize=11,fontweight='bold');ax1.tick_params(labelsize=7);ax1.grid(True,alpha=0.12)

        # P2: 净值曲线
        ax2=fig.add_subplot(gs[2]);ax2.set_facecolor('#FFFFFF')
        nc='#CC2222'if ret>=0 else'#228B22'
        ax2.fill_between(ndf['date'],1,ndf['nav']/INIT,alpha=0.08,color=nc)
        ax2.plot(ndf['date'],ndf['nav']/INIT,color=nc,lw=1.8)
        ax2.axhline(y=1,color='#AAA',lw=0.8,ls='--')
        nv_map=dict(zip(ndf['date'].dt.strftime('%Y-%m-%d'),ndf['nav']/INIT))
        for _,t in td.iterrows():
            ds=t['date'].strftime('%Y-%m-%d')if hasattr(t['date'],'strftime')else str(t['date'])[:10]
            if ds not in nv_map:continue
            yv=nv_map[ds];nm=str(t.get('name','')or'');ig=nm in('创业板','科创50')
            if t['dir']=='BUY':
                c='#3498DB'if ig else'#CC0000';mk='D'if ig else'^'
                ax2.scatter(t['date'],yv,color=c,s=80 if ig else 35,marker=mk,zorder=6,edgecolors='white',lw=1.2)
                if ig:ax2.annotate(nm[:2],(t['date'],yv),textcoords='offset points',xytext=(4,12),fontsize=8,color=c,fontweight='bold')
            elif t['dir']in('SELL','STOP','PANIC'):
                c='#E67E22'if ig else'#008800';mk='d'if ig else'v'
                ax2.scatter(t['date'],yv,color=c,s=80 if ig else 35,marker=mk,zorder=6,edgecolors='white',lw=1.2)
                if ig:ax2.annotate(nm[:2],(t['date'],yv),textcoords='offset points',xytext=(4,-12),fontsize=8,color=c,fontweight='bold')
        nav_ret=(ndf['nav'].iloc[-1]/INIT-1)*100
        ax2.set_title(f'净值  {nav_ret:+.1f}%',fontsize=11,fontweight='bold',color=nc);ax2.tick_params(labelsize=7);ax2.grid(True,alpha=0.12)

        # P3+P4: 副线K线
        for gname,gcolor in[('创业板','#3498DB'),('科创50','#E67E22')]:
            idx=3 if gname=='创业板'else 4
            ax=fig.add_subplot(gs[idx])
            gdf=raw[gname][raw[gname]['date']>=start]
            gdates=gdf['date'].values;gclose=gdf['close'].values
            ax.plot(gdates,gclose,color=gcolor,lw=1.2)
            # MA20
            gma20=pd.Series(gclose).rolling(20).mean().values
            ax.plot(gdates,gma20,color=gcolor,lw=0.6,ls='--',alpha=0.4)
            for _,t in td.iterrows():
                nm=str(t.get('name','')or'')
                if nm!=gname:continue
                ds=t['date'].strftime('%Y-%m-%d')if hasattr(t['date'],'strftime')else str(t['date'])[:10]
                for j,dv in enumerate(gdates):
                    d1=str(pd.Timestamp(dv).date());d2=str(pd.Timestamp(t['date']).date())
                    if d1==d2:
                        yv=gclose[j]
                        if t['dir']=='BUY':ax.scatter(pd.Timestamp(dv),yv,color='#CC0000',s=80,marker='^',zorder=6,edgecolors='white',lw=1.5)
                        elif t['dir']in('SELL','STOP','PANIC'):ax.scatter(pd.Timestamp(dv),yv,color='#008800',s=80,marker='v',zorder=6,edgecolors='white',lw=1.5);break
            ax.set_title(f'{gname} + MA20',fontsize=11,fontweight='bold',color=gcolor);ax.tick_params(labelsize=7);ax.grid(True,alpha=0.12)

        plt.savefig(os.path.join(SCRIPT,'backtest_chart.png'),dpi=150,bbox_inches='tight',facecolor='#FAFAFA');plt.close()
        print(f'  图表: {SCRIPT}\\backtest_chart.png')


def live_signal():
    print("获取数据...")
    raw=fetch();df_main=add_main(raw[MAIN_NAME]);dfs_g={n:add_growth(d)for n,d in raw.items()if n!=MAIN_NAME}
    idx=-1;row=df_main.iloc[idx];date=row['date'];price=row['close'];rsi=row['rsi'];lo=row['lo'];up=row['up']
    bb_pos=(price-lo)/(up-lo)*100 if up>lo else 50
    bb_buy=price<=lo;bb_sell=price>=up;rsi_buy=rsi<=RSI_L
    buy_ok=bb_buy or rsi_buy;sell_ok=(bb_sell and rsi>=ERS)
    sig='买入'if buy_ok else('卖出'if sell_ok else'持有')
    ranking=rank_momentum(dfs_g,date)
    rank_str=' > '.join(f'{n}({v:+.1%})'for n,v in ranking)
    print(f"\n{'='*60}")
    print(f"  YH_ultra v7  {date.strftime('%Y-%m-%d')}  红利: {sig}")
    print(f"  价格{price:.3f}  RSI{rsi:.1f}  BB{bb_pos:.0f}%")
    print(f"  副线动量排名: {rank_str}")
    print(f"{'='*60}")


def main():
    import argparse;p=argparse.ArgumentParser()
    p.add_argument('--from',dest='fr',type=str,default=None);p.add_argument('--dca',dest='dca',type=float,default=0)
    a=p.parse_args();global DCA;DCA=a.dca*10000
    if a.fr:run_backtest(a.fr)
    else:live_signal()

if __name__=='__main__':main()
