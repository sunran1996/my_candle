# -*- coding: utf-8 -*-
"""
YH04  YH02红利低波主线 + 成长指数趋势副线
无DCA: 一次性满仓红利低波
含DCA: 月投2万, 主线加仓, 副线存钱不分仓
"""
import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import akshare as ak, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

_fonts = [f.name for f in fm.fontManager.ttflist]
CN = 'WenQuanYi Zen Hei' if 'WenQuanYi Zen Hei' in _fonts else ('SimHei' if 'SimHei' in _fonts else 'DejaVu Sans')
plt.rcParams['font.sans-serif'] = [CN]; plt.rcParams['axes.unicode_minus'] = False

# ======================== 参数 ========================
INIT   = 1_000_000; RESERVE = 100_000; COMM = 0.0003; SLIP = 0.0001
DCA    = 0         # 月定投额, 0=关闭
MOM    = 10        # 动量周期
REBAL  = 5         # 调仓周期
SWITCH = 0.03      # 换仓阈值
TRAIL  = 0.10      # 副线移动止损
NAV_STOP = 0.13    # 总市值回撤13%→全清仓
SCRIPT = os.path.dirname(os.path.abspath(__file__))

# YH02参数
BB_P=45; BB_S=2.0; RSI_P=14; RSI_L=30; RSI_H=70; ERS=65; BA=0.001
HARD_STOP = 0.12

# 标的
MAIN_SYM  = 'sh512890'; MAIN_NAME = '红利低波'
GROWTH = {'创业板':'sz159915','科创50':'sh588000','人工智能':'sh515070','半导体':'sh512480'}

# ================================================================
def fetch():
    dfs = {}
    for n, s in {**GROWTH, MAIN_NAME: MAIN_SYM}.items():
        df = ak.fund_etf_hist_sina(symbol=s); df['date'] = pd.to_datetime(df['date'])
        dfs[n] = df[['date','close']].sort_values('date').reset_index(drop=True)
    return dfs

def add_main(df):
    """YH02指标"""
    df = df.copy()
    r = df['close'].pct_change().fillna(0); r[abs(r)>0.1]=0
    df['adj'] = (1+r).cumprod()
    df['ma']=df['adj'].rolling(BB_P).mean(); df['std']=df['adj'].rolling(BB_P).std()
    df['up']=df['ma']+BB_S*df['std']; df['lo']=df['ma']-BB_S*df['std']
    df['ua']=df['up'].diff().diff().rolling(3,min_periods=1).mean()
    df['pa']=df['adj'].diff().diff().rolling(3,min_periods=1).mean()
    d=df['adj'].diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/RSI_P,adjust=False).mean()/l.ewm(alpha=1/RSI_P,adjust=False).mean().replace(0,np.nan))
    return df

def main_signal(row, pbw):
    """YH02信号"""
    adj,rsi,up,lo=row['adj'],row['rsi'],row['up'],row['lo']
    if pd.isna(lo) or pd.isna(rsi): return 'HOLD', pbw
    bw=(up-lo)/row['ma'] if row['ma']>0 else 0.1
    exp=(pbw is not None and bw>pbw); nbw=bw
    bb_buy=(adj<=lo); bb_sell=(adj>=up); rsi_buy=(rsi<=RSI_L)
    if exp:
        raw_sell=(bb_sell and rsi>=ERS)
        ua=row['ua'] if not pd.isna(row['ua']) else 0
        pa=row['pa'] if not pd.isna(row['pa']) else 0
        sell_sig=raw_sell and not((ua>BA)and(pa>0))
        buy_sig=(bb_buy or rsi_buy)
    else:
        buy_sig=(bb_buy or rsi_buy); sell_sig=(bb_sell or rsi>=RSI_H)
    if buy_sig: return 'BUY', nbw
    elif sell_sig: return 'SELL', nbw
    else: return 'HOLD', nbw

def add_growth(df):
    df=df.copy(); df['mom']=df['close']/df['close'].shift(MOM)-1
    e10=df['close'].ewm(span=10,adjust=False).mean()
    e20=df['close'].ewm(span=20,adjust=False).mean()
    df['macd']=e10-e20; df['macd_s']=df['macd'].ewm(span=7,adjust=False).mean()
    df['macd_h']=df['macd']-df['macd_s']
    return df

def rank_growth(dfs, idx):
    s={}
    for n in GROWTH:
        pos=idx if idx>=0 else len(dfs[n])+idx
        if pos<30 or pos>=len(dfs[n]): continue
        h=dfs[n]['macd_h'].iloc[pos]
        if pd.isna(h): continue
        s[n]=h
    return sorted(s,key=s.get,reverse=True)

# ================================================================
def run_backtest(start_str):
    start=pd.Timestamp(start_str)
    print("获取数据...")
    raw=fetch()
    df_main=add_main(raw[MAIN_NAME])
    dfs_growth={n:add_growth(d) for n,d in raw.items() if n!=MAIN_NAME}
    dates=sorted(set.intersection(*[set(d['date']) for d in raw.values()]))
    dates=[d for d in dates if d>=start]
    if len(dates)<60: return

    ALL=list(GROWTH.keys())+[MAIN_NAME]
    start_px=raw[MAIN_NAME][raw[MAIN_NAME]['date']>=start]['close'].iloc[0]

    # 状态变量
    cash=0.0; rp=RESERVE; ra=False; rs=0.0; total_inv=INIT
    shares={n:0.0 for n in ALL}
    shares[MAIN_NAME]=INIT/start_px*(1-COMM-SLIP)
    position=MAIN_NAME; peak=INIT; navs=[]; trades=[]
    sc=2; pbw=None; ep=start_px; hse=0; last_rb=None; last_month=None; stopped=False; partial_done=False

    for date in dates:
        # DCA: 现金直接到账
        ym=(date.year,date.month)
        if DCA>0 and last_month and ym!=last_month:
            cash+=DCA; total_inv+=DCA
        last_month=ym

        # 价格
        px={}
        for n in ALL:
            r=raw[n][raw[n]['date']==date]
            if len(r): px[n]=r['close'].iloc[0]

        nav=cash+sum(shares[n]*px.get(n,0) for n in ALL)
        if nav>peak: peak=nav
        dd_nav=(nav-peak)/peak if peak>0 else 0

        # 总市值回撤13% → 全清仓
        if dd_nav<-NAV_STOP and any(shares[n]>0 for n in ALL):
            for n in ALL:
                if shares[n]>0 and n in px:
                    cash+=shares[n]*px[n]*(1-COMM-SLIP); shares[n]=0.0
            trades.append({'date':date,'dir':'PANIC','name':'ALL','price':0,'nav':nav})
            peak=nav; position=None; stopped=True
            navs.append({'date':date,'nav':nav,'pos':position}); continue

        # YH02信号
        mr=df_main[df_main['date']==date]
        if len(mr)==0: navs.append({'date':date,'nav':nav,'pos':position}); continue
        sig,pbw=main_signal(mr.iloc[0],pbw)
        mp=px.get(MAIN_NAME,0)

        # 硬止损(红利)
        if position==MAIN_NAME and mp>0 and ep>0 and mp<ep*(1-HARD_STOP):
            cash+=shares[MAIN_NAME]*mp*(1-COMM-SLIP); shares[MAIN_NAME]=0.0
            trades.append({'date':date,'dir':'STOP','name':MAIN_NAME,'price':mp,'nav':nav})
            position=None

        # ==== YH02 BUY → 全仓红利低波 ====
        if sig=='BUY' and position!=MAIN_NAME:
            # 清成长
            if position:
                for n in GROWTH:
                    if shares[n]>0 and n in px: cash+=shares[n]*px[n]*(1-COMM-SLIP); shares[n]=0.0
                trades.append({'date':date,'dir':'SELL','name':position,'price':px.get(position,0),'nav':nav})
            # 全仓红利
            if mp>0:
                val=min(cash,nav); bs=val/mp*(1-COMM-SLIP)
                if ra: rs+=bs*(RESERVE/(INIT+RESERVE)); shares[MAIN_NAME]+=bs-bs*(RESERVE/(INIT+RESERVE))
                else: shares[MAIN_NAME]+=bs
                cash-=val
                trades.append({'date':date,'dir':'BUY','name':MAIN_NAME,'price':mp,'nav':nav})
                position=MAIN_NAME; ep=mp; stopped=False

        # ==== YH02 SELL → 清红利 ====
        elif sig=='SELL' and position==MAIN_NAME:
            if shares[MAIN_NAME]>0 and mp>0:
                cash+=shares[MAIN_NAME]*mp*(1-COMM-SLIP)
                if ra: rp=rs*mp*(1-COMM-SLIP); cash-=rp; rs=0; ra=False
                shares[MAIN_NAME]=0.0
                trades.append({'date':date,'dir':'SELL','name':MAIN_NAME,'price':mp,'nav':nav})
                position=None

        # ==== HOLD + 红利 → DCA加仓 ====
        elif sig=='HOLD' and position==MAIN_NAME and DCA>0 and cash>100 and mp>0:
            bs2=cash/mp*(1-COMM-SLIP); shares[MAIN_NAME]+=bs2; cash=0

        # ==== 成长STOP后, 如果有红利持仓 → 挪到红利 ====
        elif sig=='HOLD' and position!=MAIN_NAME and shares[MAIN_NAME]>0 and mp>0:
            # 有红利底仓, 不追成长; 红利信号仍是HOLD, 不动
            pass

        # ==== HOLD + 空仓 → 成长轮动 ====
        elif sig=='HOLD' and position!=MAIN_NAME:
            # 移动止损
            if position and position in px:
                if px[position]>hse: hse=px[position]
                if px[position]<hse*(1-TRAIL):
                    # 卖出成长, 如果有红利底仓就加仓红利
                    sell_cash=shares[position]*px[position]*(1-COMM-SLIP)
                    cash+=sell_cash; shares[position]=0.0
                    trades.append({'date':date,'dir':'STOP','name':position,'price':px[position],'nav':nav})
                    if shares[MAIN_NAME]>0 and mp>0:
                        bs2=cash/mp*(1-COMM-SLIP); shares[MAIN_NAME]+=bs2; cash=0
                        trades.append({'date':date,'dir':'ADD','name':MAIN_NAME,'price':mp,'nav':nav,'mode':'副线结束归队'})
                    position=None; stopped=True; partial_done=False

            # 进场 (止损后空仓等主线BUY, 或有红利底仓不再追成长)
            if stopped or shares[MAIN_NAME]>0: navs.append({'date':date,'nav':nav,'pos':position}); continue
            bm_idx=None
            for n in GROWTH:
                idxs=dfs_growth[n][dfs_growth[n]['date']==date].index
                if len(idxs): bm_idx=idxs[0]; break
            if bm_idx is None: navs.append({'date':date,'nav':nav,'pos':position}); continue

            days=(date-last_rb).days if last_rb else 999
            if days>=REBAL and position is None:
                ranking=rank_growth(dfs_growth,bm_idx)
                if ranking and dfs_growth[ranking[0]]['macd_h'].iloc[bm_idx]>0:
                    target=ranking[0]
                    if target in px:
                        # DCA现金不留入成长(仅在主线红利时加仓)
                        invest_cash=cash
                        val=min(invest_cash,nav)
                        if val>100:
                            shares[target]=val/px[target]*(1-COMM-SLIP); cash-=val
                            trades.append({'date':date,'dir':'BUY','name':target,'price':px[target],'nav':nav})
                            position=target; hse=px[target]
                    last_rb=date

        nav=cash+sum(shares[n]*px.get(n,0) for n in ALL)
        navs.append({'date':date,'nav':nav,'pos':position})

    # ========== 统计 ==========
    ndf=pd.DataFrame(navs); final=ndf['nav'].iloc[-1]
    ret=(final/total_inv-1)*100 if total_inv>0 else 0
    ann=((1+ret/100)**(252/len(ndf))-1)*100
    dr=ndf['nav'].pct_change().dropna()
    sr=(ann/100-0.02)/(dr.std()*np.sqrt(252)) if dr.std()>0 else 0
    mdd=((ndf['nav']/INIT-(ndf['nav']/INIT).cummax())/(ndf['nav']/INIT).cummax()).min()*100
    td=pd.DataFrame(trades)

    dca_label=f' DCA月投{DCA/1e4:.0f}万' if DCA>0 else ''
    print(f"\n  YH04  YH02主线+成长副线{dca_label}  |  动量{MOM}日  止损{TRAIL*100:.0f}%")
    print(f"  Return: {ret:+.2f}%  Annual: {ann:+.2f}%  Sharpe: {sr:.3f}  MaxDD: {mdd:+.2f}%")
    if DCA>0:
        print(f"  交易: {len(td)}笔  投入{total_inv/1e4:.0f}万  终值{final/1e4:.1f}万  净赚{final-total_inv:,.0f}")
    else:
        print(f"  交易: {len(td)}笔  终值{final:,.0f}")

    if len(td)>0:
        print(f"  标的分布: {td['name'].value_counts().to_dict()}")
        print(f"\n  {'日期':<12} {'标的':<8} {'方向':<6} {'价格':>8} {'盈亏':>8} {'总市值':>12}")
        print(f"  {'─'*55}")
        epx={}
        for _,t in td.iterrows():
            nm=t['name']; d=t['date'].strftime('%Y-%m-%d')[:10]
            pv=t.get('price',0) if pd.notna(t.get('price',0)) else 0
            nv=t.get('nav',0) if pd.notna(t.get('nav',0)) else 0
            if t['dir']=='BUY':
                epx[nm]=pv
                print(f'  {d:<12} {nm:<8} 买入   {pv:>8.3f}  {"—":>8}  {nv:>12,.0f}')
            else:
                ep=epx.pop(nm,pv); pnl=(pv/ep-1)*100 if ep>0 else 0
                print(f'  {d:<12} {nm:<8} {t["dir"]:<6} {pv:>8.3f}  {pnl:>+7.1f}%  {nv:>12,.0f}')
        print(f'  最终市值: {final:,.0f}')

    ndf['year']=ndf['date'].dt.year
    print(f"\n  {'年份':<6} {'收益':>8} {'MaxDD':>8}")
    for yr,grp in ndf.groupby('year'):
        if len(grp)<10: continue
        yr_ret=(grp['nav'].iloc[-1]/grp['nav'].iloc[0]-1)*100
        yr_mdd=((grp['nav']/grp['nav'].iloc[0]-(grp['nav']/grp['nav'].iloc[0]).cummax())/(grp['nav']/grp['nav'].iloc[0]).cummax()).min()*100
        print(f"  {yr:<6} {yr_ret:>+7.1f}% {yr_mdd:>+7.1f}%")

    # ── 收益曲线 + 年度收益 ──
    if len(ndf)>1:
        fig,axes=plt.subplots(2,1,figsize=(16,10),facecolor='white',gridspec_kw={'height_ratios':[2.5,1],'hspace':0.3,'top':0.94})
        ax=axes[0]
        ax.plot(ndf['date'],ndf['nav']/INIT,color='#CC2222',lw=2.0,label=f'YH04 ({ret:+.1f}%)')
        # 回撤区间标注
        cum=ndf['nav']/INIT; dd_series=(cum-cum.cummax())/cum.cummax()
        in_dd=False; dd_start=None
        for i,(date2,dd_val) in enumerate(zip(ndf['date'],dd_series)):
            if dd_val<-0.1 and not in_dd: dd_start=date2; in_dd=True
            elif dd_val>-0.02 and in_dd and dd_start:
                ax.axvspan(dd_start,date2,alpha=0.08,color='red')
                in_dd=False; dd_start=None
        if in_dd and dd_start: ax.axvspan(dd_start,ndf['date'].iloc[-1],alpha=0.08,color='red')
        ax.axhline(y=1.0,color='#888',lw=0.8,ls='--')
        ax.legend(fontsize=12,loc='upper left'); ax.set_ylabel('净值倍数',fontsize=12)
        ax.grid(True,alpha=0.12)
        ax.set_title(f'YH04 净值曲线 | {ret:+.1f}% | 夏普{sr:.3f} | 回撤{mdd:+.1f}%',fontsize=15,fontweight='bold')
        # 年度收益柱状图
        ax=axes[1]
        years=[]; rets=[]
        for yr,grp in ndf.groupby('year'):
            if len(grp)<10: continue
            years.append(yr); rets.append((grp['nav'].iloc[-1]/grp['nav'].iloc[0]-1)*100)
        colors2=['#CC2222' if r>=0 else '#228B22' for r in rets]
        bars=ax.bar(range(len(years)),rets,color=colors2,alpha=0.85,edgecolor='white',lw=1)
        for bar,val in zip(bars,rets):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+(2 if val>=0 else -3),
                    f'{val:+.1f}%',ha='center',fontsize=12,fontweight='bold',color='#CC2222' if val>=0 else '#228B22')
        ax.axhline(y=0,color='black',lw=1)
        ax.set_xticks(range(len(years))); ax.set_xticklabels([str(y) for y in years],fontsize=13,fontweight='bold')
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%+.0f%%'))
        ax.set_ylabel('年度收益',fontsize=11); ax.grid(True,alpha=0.12,axis='y')
        ax.set_title('年度收益',fontsize=13,fontweight='bold')
        plt.savefig(os.path.join(SCRIPT,'nav_chart.png'),dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
        print(f'  净值图: {SCRIPT}/nav_chart.png')

    # ── 交易全景 ──
    if len(ndf)>1 and len(td)>0:
        names=[MAIN_NAME]+list(GROWTH.keys())
        colors=['#9B59B6','#E74C3C','#3498DB','#F39C12','#2ECC71']
        fig,axes=plt.subplots(5,1,figsize=(16,18),facecolor='white',gridspec_kw={'hspace':0.1,'top':0.96,'bottom':0.02})
        trade_dates=sorted(set(td['date'].dropna()))
        for i,(name,ax) in enumerate(zip(names,axes)):
            df=raw[name]; df=df[df['date']>=start]
            if name==MAIN_NAME:
                adj_series=df_main[df_main['date']>=start]
                ax.plot(adj_series['date'],adj_series['adj'],color=colors[i],lw=0.8,alpha=0.9)
            else:
                ax.plot(df['date'],df['close'],color=colors[i],lw=0.8,alpha=0.9)
            for _,t in td.iterrows():
                if pd.isna(t.get('name')) or t['name']!=name: continue
                d2=t['date']
                if name==MAIN_NAME:
                    ar=df_main[df_main['date']==d2]
                    if len(ar): pv2=ar['adj'].values[0]
                    else: continue
                else:
                    if d2 not in df['date'].values: continue
                    pv2=df[df['date']==d2]['close'].values[0]
                if t['dir']=='BUY': ax.scatter(d2,pv2,color='#CC2222',s=70,marker='^',zorder=5,edgecolors='white',lw=1.2)
                else: ax.scatter(d2,pv2,color='#228B22',s=70,marker='v',zorder=5,edgecolors='white',lw=1.2)
            for td2 in trade_dates: ax.axvline(x=td2,color='#999',lw=0.5,ls='--',alpha=0.6)
            ax.set_ylabel(name,fontsize=9,color=colors[i],fontweight='bold')
            ax.tick_params(labelsize=7); ax.grid(True,alpha=0.08)
            if i<4: ax.set_xticklabels([])
        axes[0].set_title(f'YH04 交易全景{dca_label} | {ret:+.1f}% | 夏普{sr:.3f} | {len(td)}笔',fontsize=14,fontweight='bold')
        plt.savefig(os.path.join(SCRIPT,'backtest_chart.png'),dpi=120,bbox_inches='tight',facecolor='white'); plt.close()


def live_signal():
    print("获取数据...")
    raw=fetch(); df_main=add_main(raw[MAIN_NAME])
    dfs_growth={n:add_growth(d) for n,d in raw.items() if n!=MAIN_NAME}
    idx=len(df_main)-1; row=df_main.iloc[idx]; date=row['date']
    adj,rsi=row['adj'],row['rsi']; lo,up=row['lo'],row['up']
    bb_pos=(adj-lo)/(up-lo)*100 if up>lo else 50
    bb_buy=adj<=lo; bb_sell=adj>=up; rsi_buy=rsi<=RSI_L
    exp=True
    if exp: buy_ok=bb_buy or rsi_buy; sell_ok=bb_sell and rsi>=ERS
    else: buy_ok=bb_buy or rsi_buy; sell_ok=bb_sell or rsi>=RSI_H
    sig='买入' if buy_ok else ('卖出' if sell_ok else '持有')
    print(f"\n{'='*60}")
    print(f"  YH04  {date.strftime('%Y-%m-%d')}  红利低波: {sig}")
    print(f"  价格{row['close']:.3f}  RSI{rsi:.1f}  BB{bb_pos:.0f}%  {'扩张' if exp else '收缩'}")
    print(f"{'─'*60}")
    print(f"  成长指数 MACD排名 (EMA10/20/7):")
    rank=rank_growth(dfs_growth,-1)
    for i,n in enumerate(rank[:4]):
        rp=len(dfs_growth[n])-1
        p=dfs_growth[n]['close'].iloc[rp]; m=dfs_growth[n]['mom'].iloc[rp]
        macd=dfs_growth[n]['macd_h'].iloc[rp]
        bar='█'*(5-i)
        print(f"  #{i+1} {n:<6} {p:.3f}  动量{m:+.1%}  MACD{macd:+.3f}  {bar}")
    print(f"{'='*60}")

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--from',dest='fr',type=str,default=None)
    p.add_argument('--dca',dest='dca',type=float,default=0,help='月定投额(万)')
    a=p.parse_args()
    global DCA; DCA=a.dca*10000
    if a.fr: run_backtest(a.fr)
    else: live_signal()

if __name__=='__main__':
    main()
