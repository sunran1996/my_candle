# -*- coding: utf-8 -*-
"""
YH05  YH02红利低波主线 + 成长指数趋势副线 (从YH05继承,继续打磨)
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
GROWTH = {'创业板':'sz159915'}

# ================================================================
def fetch():
    dfs = {}
    for n, s in {**GROWTH, MAIN_NAME: MAIN_SYM}.items():
        df = ak.fund_etf_hist_sina(symbol=s); df['date'] = pd.to_datetime(df['date'])
        dfs[n] = df[['date','close']].sort_values('date').reset_index(drop=True)
    return dfs

def add_main(df):
    """YH02指标 + MA200趋势"""
    df = df.copy()
    r = df['close'].pct_change().fillna(0); r[abs(r)>0.1]=0
    df['adj'] = (1+r).cumprod()
    df['ma']=df['adj'].rolling(BB_P).mean(); df['std']=df['adj'].rolling(BB_P).std()
    df['up']=df['ma']+BB_S*df['std']; df['lo']=df['ma']-BB_S*df['std']
    df['ua']=df['up'].diff().diff().rolling(3,min_periods=1).mean()
    df['pa']=df['adj'].diff().diff().rolling(3,min_periods=1).mean()
    d=df['adj'].diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/RSI_P,adjust=False).mean()/l.ewm(alpha=1/RSI_P,adjust=False).mean().replace(0,np.nan))
    df['mom20']=df['close']/df['close'].shift(20)-1  # 用于D1相关性过滤
    df['ma200']=df['close'].rolling(200).mean()      # 红利趋势过滤
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
    df['macd_h']=df['macd']-df['macd_s']; df['ma20']=df['close'].rolling(20).mean()
    df['macd_line']=df['macd']; df['ma200']=df['close'].rolling(200).mean()
    # 入场过滤指标
    df['ma10']=df['close'].rolling(10).mean()
    df['mom20']=df['close']/df['close'].shift(20)-1  # 20日动量
    d=df['close'].diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/14,adjust=False).mean()/l.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan))
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
    # 日期对齐: 用全部ETF的交集(含对齐用的下划线ETF)
    dates=sorted(set.intersection(*[set(d['date']) for d in raw.values()]))
    dates=[d for d in dates if d>=start]
    dates=[d for d in dates if d>=start]
    if len(dates)<60: return

    ALL=list(GROWTH.keys())+[MAIN_NAME]
    start_px=raw[MAIN_NAME][raw[MAIN_NAME]['date']>=start]['close'].iloc[0]

    # 状态变量
    cash=0.0; rp=RESERVE; ra=False; rs=0.0; total_inv=INIT
    shares={n:0.0 for n in ALL}
    shares[MAIN_NAME]=INIT/start_px*(1-COMM-SLIP)
    position=MAIN_NAME; peak=INIT; navs=[]; trades=[]; invs=[]
    sc=2; pbw=None; ep=start_px; hse=0; last_rb=None; last_month=None; stopped=False; partial_done=False
    growth_ban=0  # 创业板连败保护: >0禁止入场, BUY时-1, STOP时=2

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
            peak=nav; position=None; stopped=True; growth_ban=2
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

        # ==== YH02 BUY → 红利低波 ====
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
                growth_ban=max(0,growth_ban-1)

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
                    growth_ban=2  # 止损后需2次BUY才解禁(跳过一轮)

            # 进场 (止损后空仓等主线BUY, 或有红利底仓不再追成长)
            if stopped or shares[MAIN_NAME]>0: navs.append({'date':date,'nav':nav,'pos':position}); continue
            if growth_ban>0: navs.append({'date':date,'nav':nav,'pos':position}); continue  # 连败保护
            bm_idx=None
            for n in GROWTH:
                idxs=dfs_growth[n][dfs_growth[n]['date']==date].index
                if len(idxs): bm_idx=idxs[0]; break
            if bm_idx is None: navs.append({'date':date,'nav':nav,'pos':position}); continue

            days=(date-last_rb).days if last_rb else 999
            if days>=REBAL and position is None:
                gn=list(GROWTH.keys())[0]
                gdf=dfs_growth[gn];gr=gdf[gdf['date']==date]
                if len(gr)==0:navs.append({'date':date,'nav':nav,'pos':position});continue
                g_close=gr['close'].values[0];g_ma200=gr['ma200'].values[0]
                g_rsi=gr['rsi'].values[0];g_ma10=gr['ma10'].values[0]
                g_mom20=gr['mom20'].values[0]
                # 红利20日动量
                m_mom20=df_main[df_main['date']==date]['mom20'].values[0]if len(df_main[df_main['date']==date])>0 else 0

                # 熊市过滤: 创业板<MA200 → 不操作
                if not pd.isna(g_ma200) and g_close<g_ma200:
                    navs.append({'date':date,'nav':nav,'pos':position});continue

                # D1: 相关性过滤 → 红利+创业板双杀时不入场
                if not pd.isna(m_mom20)and not pd.isna(g_mom20):
                    if m_mom20<0 and g_mom20<0:
                        navs.append({'date':date,'nav':nav,'pos':position});continue

                ranking=rank_growth(dfs_growth,bm_idx)
                bull_ok = g_close > gr['ma20'].values[0]
                macd_ok = gr['macd_h'].values[0]>0
                if ranking and macd_ok and bull_ok:
                    target=ranking[0]
                    if target in px:
                        # A1: RSI过滤 → RSI 25-55之间(回调到位,不追高不接飞刀)
                        if not pd.isna(g_rsi):
                            if g_rsi>55:navs.append({'date':date,'nav':nav,'pos':position});continue
                            if g_rsi<25:navs.append({'date':date,'nav':nav,'pos':position});continue
                        # A2: 回踩MA10 → 价格距MA10不超过5%
                        if not pd.isna(g_ma10)and g_ma10>0:
                            if g_close>g_ma10*1.05:navs.append({'date':date,'nav':nav,'pos':position});continue

                        above_zero = gr['macd_line'].values[0] > 0
                        pos_pct = 1.0 if above_zero else 0.3
                        val=min(cash, nav*pos_pct)
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
    print(f"\n  YH05  YH02主线+成长副线{dca_label}  |  动量{MOM}日  止损{TRAIL*100:.0f}%")
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
        # DCA修正: 扣除当年新增投入
        yr_dca = DCA * 12 if DCA>0 else 0
        yr_ret = ((grp['nav'].iloc[-1] - yr_dca) / grp['nav'].iloc[0] - 1)*100 if DCA>0 else (grp['nav'].iloc[-1]/grp['nav'].iloc[0]-1)*100
        yr_mdd=((grp['nav']/grp['nav'].iloc[0]-(grp['nav']/grp['nav'].iloc[0]).cummax())/(grp['nav']/grp['nav'].iloc[0]).cummax()).min()*100
        print(f"  {yr:<6} {yr_ret:>+7.1f}% {yr_mdd:>+7.1f}%")

    # ── 净值曲线 + 年度收益 ──
    if len(ndf)>1 and len(td)>0:
        # 合并成一个nav_chart: 上=净值+交易标记, 下=年度收益
        fig,axes=plt.subplots(2,1,figsize=(18,10),facecolor='white',
                              gridspec_kw={'height_ratios':[2.5,1],'hspace':0.25,'top':0.95})
        ax=axes[0]
        ax.plot(ndf['date'],ndf['nav']/INIT,color='#CC2222',lw=2.0,label='YH05 净值')
        ax.axhline(y=1.0,color='#888',lw=0.8,ls='--')

        # 交易标记 (用实际净值点位)
        nav_map=dict(zip(ndf['date'].dt.strftime('%Y-%m-%d'),ndf['nav']/INIT))
        for _,t in td.iterrows():
            d_str=t['date'].strftime('%Y-%m-%d') if hasattr(t['date'],'strftime') else str(t['date'])[:10]
            if d_str not in nav_map: continue
            yp=nav_map[d_str]
            if t['dir'] in ('BUY','ADD'): ax.scatter(t['date'],yp,color='#CC0000',s=70,marker='^',zorder=5,edgecolors='white',lw=1.2)
            elif t['dir'] in ('SELL',): ax.scatter(t['date'],yp,color='#008800',s=70,marker='v',zorder=5,edgecolors='white',lw=1.2)
            elif t['dir']=='STOP': ax.scatter(t['date'],yp,color='#E67E22',s=80,marker='v',zorder=5,edgecolors='white',lw=1.5)
            elif t['dir']=='PANIC': ax.scatter(t['date'],yp,color='#000000',s=90,marker='X',zorder=5,edgecolors='white',lw=1.5)

        # 回撤区
        cum=ndf['nav']/INIT; dd_series=(cum-cum.cummax())/cum.cummax()
        in_dd=False; dd_start=None
        for i,(date2,dd_val) in enumerate(zip(ndf['date'],dd_series)):
            if dd_val<-0.1 and not in_dd: dd_start=date2; in_dd=True
            elif dd_val>-0.02 and in_dd and dd_start:
                ax.axvspan(dd_start,date2,alpha=0.06,color='red')
                in_dd=False; dd_start=None
        if in_dd and dd_start: ax.axvspan(dd_start,ndf['date'].iloc[-1],alpha=0.06,color='red')

        ax.legend(fontsize=10,loc='upper left'); ax.set_ylabel('净值倍数',fontsize=11)
        ax.grid(True,alpha=0.12)
        ax.set_title(f'YH05 净值曲线 + 交易点位 | {ret:+.1f}% | 年化{ann:+.1f}% | 夏普{sr:.3f} | 回撤{mdd:.1f}% | {len(td)}笔',
                     fontsize=14,fontweight='bold')

        # P2: 年度收益
        ax=axes[1]
        years=[]; rets=[]
        for yr,grp in ndf.groupby('year'):
            if len(grp)<10: continue
            years.append(yr); rets.append((grp['nav'].iloc[-1]/grp['nav'].iloc[0]-1)*100)
        colors2=['#CC2222' if r>=0 else '#228B22' for r in rets]
        bars=ax.bar(range(len(years)),rets,color=colors2,alpha=0.85,edgecolor='white',lw=1)
        for bar,val in zip(bars,rets):
            off=1.5 if val>=0 else-3.5
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+off,
                    f'{val:+.1f}%',ha='center',fontsize=13,fontweight='bold',
                    color='#CC2222' if val>=0 else '#228B22')
        ax.axhline(y=0,color='black',lw=1)
        ax.set_xticks(range(len(years))); ax.set_xticklabels([str(y) for y in years],fontsize=13,fontweight='bold')
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%+.0f%%'))
        ax.set_ylabel('年度收益',fontsize=11); ax.grid(True,alpha=0.12,axis='y')
        ax.set_title('年度收益',fontsize=13,fontweight='bold')

        plt.savefig(os.path.join(SCRIPT,'nav_chart.png'),dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
        print(f'  图表: {SCRIPT}/nav_chart.png')


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
    print(f"  YH05  {date.strftime('%Y-%m-%d')}  红利低波: {sig}")
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
