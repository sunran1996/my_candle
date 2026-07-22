# -*- coding: utf-8 -*-
"""YH04 每日信号 + K线收益图"""
import sys, io, os, json, ssl, time, base64, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import akshare as ak, pandas as pd, numpy as np
import urllib.request as ur
import matplotlib; matplotlib.use('Agg')
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

_fonts=[f.name for f in fm.fontManager.ttflist]
CN='WenQuanYi Zen Hei' if 'WenQuanYi Zen Hei' in _fonts else ('SimHei'if'SimHei'in _fonts else'DejaVu Sans')
plt.rcParams['font.sans-serif']=[CN]; plt.rcParams['axes.unicode_minus']=False
# mplfinance字体通过make_mpf_style的rc参数配置

MAIN_SYM='sh512890'; MAIN_NAME='红利低波'
GROWTH={'创业板':'sz159915','科创50':'sh588000','人工智能':'sh515070','半导体':'sh512480'}
BB_P=45;BB_S=2.0;RSI_P=14;RSI_L=30;RSI_H=70;ERS=65;BA=0.001
BARK_KEYS=['eoq8G58fJtDDFxHjhNueGH']  # 单推送防重复
REPO='sunran1996/my_candle'

def fetch():
    dfs={}
    for n,s in {**GROWTH,MAIN_NAME:MAIN_SYM}.items():
        df=ak.fund_etf_hist_sina(symbol=s); df['date']=pd.to_datetime(df['date'])
        dfs[n]=df[['date','open','high','low','close','volume']].sort_values('date').reset_index(drop=True)
    return dfs

def add_main(df):
    df=df.copy(); r=df['close'].pct_change().fillna(0); r[abs(r)>0.1]=0
    df['adj']=(1+r).cumprod()
    # BB用原始收盘价(K线坐标系)
    df['bb_ma']=df['close'].rolling(BB_P).mean(); df['bb_std']=df['close'].rolling(BB_P).std()
    df['up']=df['bb_ma']+BB_S*df['bb_std']; df['lo']=df['bb_ma']-BB_S*df['bb_std']
    d=df['adj'].diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/RSI_P,adjust=False).mean()/l.ewm(alpha=1/RSI_P,adjust=False).mean().replace(0,np.nan))
    return df

def add_growth(df):
    df=df.copy(); df['mom']=df['close']/df['close'].shift(10)-1
    e10=df['close'].ewm(span=10,adjust=False).mean(); e20=df['close'].ewm(span=20,adjust=False).mean()
    df['macd_h']=e10-e20-(e10-e20).ewm(span=7,adjust=False).mean()
    df['macd_line']=e10-e20; df['ma20']=df['close'].rolling(20).mean()
    return df

def send_bark(title,body,url=''):
    for bk in BARK_KEYS:
        try:
            data=json.dumps({'title':title,'body':body,'url':url}).encode()
            ur.urlopen(ur.Request(f'https://api.day.app/{bk}',data=data,
                       headers={'Content-Type':'application/json'}),timeout=10)
        except: pass
    print("已推送")

def main():
    try:
        print("获取数据...")
        raw=fetch(); df_main=add_main(raw[MAIN_NAME])
        dfs_g={n:add_growth(d) for n,d in raw.items() if n!=MAIN_NAME}

        # 实时行情
        is_weekend=pd.Timestamp.now().dayofweek>=5
        if not is_weekend:
            try:
                spot=ak.fund_etf_spot_em()
                for code,name in [('512890',MAIN_NAME),('159915','创业板')]:
                    s=spot[spot['代码']==code]
                    if len(s)>0:
                        rt=float(s['最新价'].iloc[0])
                        raw[name].loc[raw[name].index[-1],'close']=rt
                        raw[name].loc[raw[name].index[-1],'date']=pd.Timestamp.now()
                        print(f'  {name} {raw[name].loc[raw[name].index[-2],"close"]:.4f}→实时{rt:.4f}')
                df_main=add_main(raw[MAIN_NAME])
                dfs_g={n:add_growth(d) for n,d in raw.items() if n!=MAIN_NAME}
            except Exception as e: print(f'  实时行情失败: {e}')

        # 信号(用close-based BB)
        idx=-1; pos=len(df_main)+idx; row=df_main.iloc[pos]; date=row['date']
        price=row['close']; rsi=row['rsi']; lo=row['lo']; up=row['up']
        bb_pos=(price-lo)/(up-lo)*100 if up>lo else 50; main_px=price
        bb_buy=price<=lo; bb_sell=price>=up; rsi_buy=rsi<=RSI_L
        buy_ok=bb_buy or rsi_buy; sell_ok=(bb_sell and rsi>=ERS)

        # 成长排名
        g_idx=max(len(d)for d in dfs_g.values())-1
        scores={}
        for n in GROWTH:
            p2=min(g_idx,len(dfs_g[n])-1); v=dfs_g[n]['macd_h'].iloc[p2]
            if not pd.isna(v): scores[n]=v
        ranking=sorted(scores,key=scores.get,reverse=True)
        leader=ranking[0]if ranking else'—'; leader_macd=scores.get(leader,0)
        leader_px=raw[leader]['close'].iloc[min(len(raw[leader])-1,g_idx)]if leader!='—'else 0

        # 预警
        warn=''
        near_buy=(bb_pos<35 or rsi<45)and not buy_ok
        near_sell=(bb_pos>65 or rsi>60)and not sell_ok
        if near_sell: warn=' ⚠ 接近卖出'
        elif near_buy: warn=' ⚠ 接近买入'

        # 状态判断
        if buy_ok:
            action=f'🔴 买入红利低波'; detail=f'全仓{MAIN_NAME}@{main_px:.3f} RSI{rsi:.0f} BB{bb_pos:.0f}%'
        elif sell_ok and leader_macd>0:
            action=f'🟢 换仓{leader}'; detail=f'卖{MAIN_NAME}→买{leader}@{leader_px:.3f} MACD{leader_macd:+.3f}'
        elif sell_ok:
            action=f'⚫ 现金等待'; detail=f'{MAIN_NAME}已卖 MACD全负'
        else:
            action=f'⚪ 持有{warn}'; detail=f'{MAIN_NAME}@{main_px:.3f} RSI{rsi:.0f} BB{bb_pos:.0f}% | 副线{leader} MACD{leader_macd:+.3f}{" 可追" if leader_macd>0 else ""}'

        sub_rank=' > '.join(f'{n}({scores[n]:+.3f})'for n in ranking[:3])

        # ===== K线图 + 净值看板 =====
        lookback=120
        cn_c=mpf.make_marketcolors(up='#CC0000',down='#008800',edge='inherit',wick='inherit',volume='inherit')
        cn_s=mpf.make_mpf_style(marketcolors=cn_c,gridstyle='',rc={'font.sans-serif':[CN],'axes.unicode_minus':False})

        # 迷你回测(120日)
        main_close=raw[MAIN_NAME]['close'].tail(lookback).reset_index(drop=True)
        growth_close={n:raw[n]['close'].tail(lookback).reset_index(drop=True) for n in GROWTH if n in raw}
        m_sub=df_main.tail(lookback).reset_index(drop=True)
        g_sub={n:dfs_g[n].tail(lookback).reset_index(drop=True) for n in GROWTH if n in dfs_g}
        INIT=1_000_000; cash=0.0; sh_main=INIT/main_close.iloc[0]*(1-0.0003)
        sh_g={n:0.0 for n in GROWTH}; pos='MAIN'; peak=INIT; pbw2=None; sc2=2; ep2=main_close.iloc[0]
        hse=0; last_rb=None; stopped=False; navs=[]

        for i in range(lookback):
            mp=main_close.iloc[i]; row2=m_sub.iloc[i]
            adj=row2['adj']; rsi2=row2['rsi']; lo2=row2['lo']; up2=row2['up']
            nav=cash+sh_main*mp+sum(sh_g[n]*growth_close[n].iloc[i] for n in GROWTH if n in growth_close)
            if nav>peak: peak=nav
            if pd.isna(lo2)or pd.isna(rsi2): navs.append(nav); continue
            bw=(up2-lo2)/row2['bb_ma']if row2['bb_ma']>0 else 0.1
            exp=(pbw2 is not None and bw>pbw2); pbw2=bw
            bb_b=mp<=lo2; bb_s=mp>=up2; rsi_b=rsi2<=30
            if exp: sell_ok2=(bb_s and rsi2>=65); buy_ok2=(bb_b or rsi_b)
            else: buy_ok2=(bb_b or rsi_b); sell_ok2=(bb_s or rsi2>=70)
            if buy_ok2 or sell_ok2: sc2+=1
            if sc2<2: navs.append(nav); continue
            if buy_ok2 and pos!='MAIN':
                if pos in sh_g: cash+=sh_g[pos]*growth_close[pos].iloc[i]*(1-0.0003); sh_g[pos]=0
                sh_main=cash/mp*(1-0.0003); cash=0; pos='MAIN'; ep2=mp; stopped=False
            elif sell_ok2 and pos=='MAIN':
                cash+=sh_main*mp*(1-0.0003); sh_main=0; pos=None
            elif pos is None and not stopped:
                best_n=None; best_v=-99
                for n in GROWTH:
                    if n in g_sub and i<len(g_sub[n]):
                        v=g_sub[n]['macd_h'].iloc[i]
                        if not pd.isna(v)and v>best_v: best_v=v; best_n=n
                macd_ok=best_v>0; bull_ok=best_n and i<len(g_sub.get(best_n,[]))and growth_close.get(best_n,main_close).iloc[i]>g_sub[best_n]['ma20'].iloc[i]if best_n else False
                if best_n and macd_ok and bull_ok:
                    val=min(cash,nav); sh_g[best_n]=val/growth_close[best_n].iloc[i]*(1-0.0003); cash-=val; pos=best_n; hse=growth_close[best_n].iloc[i]; last_rb=i
            elif pos in GROWTH and pos in growth_close:
                cp=growth_close[pos].iloc[i]
                if cp>hse: hse=cp
                if cp<hse*0.9: cash+=sh_g[pos]*cp*(1-0.0003); sh_g[pos]=0; pos=None; stopped=True
            navs.append(cash+sh_main*mp+sum(sh_g[n]*growth_close[n].iloc[i] for n in GROWTH if n in growth_close))
        navs=np.array(navs); navs=navs/navs[0]

        # 用YH05风格布局
        fig=plt.figure(figsize=(6,10),facecolor='#FAFAFA')
        gs=fig.add_gridspec(3,1,height_ratios=[1.0,2.5,1.2],hspace=0.2,left=0.06,right=0.94,top=0.96,bottom=0.03)

        # P0: 信息栏
        ax0=fig.add_subplot(gs[0]); ax0.axis('off')
        ax0.text(0,0.8,f'YH04 {date.strftime("%Y-%m-%d")}',fontsize=15,fontweight='bold',color='#1A1A1A')
        ax0.text(0,0.4,f'{action}',fontsize=16,fontweight='bold',color='#E67E22'if warn else'#1A1A1A')
        ax0.text(0,0.1,f'{detail} | 副线: {sub_rank}',fontsize=9,color='#888')

        if sell_ok and leader_macd>0 and leader in raw:
            # 持有成长: K线+MACD
            plot_name=leader; plot_raw=raw[leader]
            ohlc=plot_raw.tail(lookback).copy()
            ohlc=ohlc.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            ohlc=ohlc.set_index('date')[['Open','High','Low','Close','Volume']]
            ax1=fig.add_subplot(gs[1])
            mpf.plot(ohlc,type='candle',ax=ax1,volume=False,style=cn_s)
            ax1.set_title(f'{plot_name} MACD{leader_macd:+.3f}',fontsize=10,loc='left',color='#9B59B6')
            ax1.tick_params(labelsize=7); ax1.grid(True,alpha=0.12)
        else:
            # 持有红利低波: K线+BB
            ohlc=raw[MAIN_NAME].tail(lookback).copy()
            ohlc=ohlc.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            ohlc=ohlc.set_index('date')[['Open','High','Low','Close','Volume']]
            bb_df=df_main.tail(lookback)
            ax1=fig.add_subplot(gs[1])
            ap_ma=mpf.make_addplot(bb_df['bb_ma'].values,color='#888',width=0.8,linestyle='--',ax=ax1)
            ap_up=mpf.make_addplot(bb_df['up'].values,color='#9B59B6',width=0.6,linestyle='--',ax=ax1)
            ap_lo=mpf.make_addplot(bb_df['lo'].values,color='#9B59B6',width=0.6,linestyle='--',ax=ax1)
            mpf.plot(ohlc,type='candle',ax=ax1,volume=False,style=cn_s,addplot=[ap_ma,ap_up,ap_lo])
            ax1.set_title(f'{MAIN_NAME} RSI{rsi:.0f} BB{bb_pos:.0f}%',fontsize=10,loc='left',color='#CC2222')
            ax1.tick_params(labelsize=7); ax1.grid(True,alpha=0.12)

        # P2: 净值曲线
        ax2=fig.add_subplot(gs[2]); ax2.set_facecolor('#FFFFFF')
        lc='#CC0000'if navs[-1]>=1 else'#008800'
        ax2.fill_between(range(len(navs)),1,navs,alpha=0.08,color=lc)
        ax2.plot(range(len(navs)),navs,color=lc,lw=2.0)
        ax2.axhline(y=1,color='#AAA',lw=0.8,ls='--')
        ax2.set_title(f'策略净值 {(navs[-1]-1)*100:+.1f}% (120日)',fontsize=10,loc='left',color=lc)
        ax2.tick_params(labelsize=7); ax2.grid(True,alpha=0.12)

        buf=io.BytesIO(); fig.savefig(buf,dpi=150,bbox_inches='tight',facecolor='#FAFAFA'); plt.close(fig)
        img_bytes=buf.getvalue()

        # 上传 + 推送
        token=os.environ.get('GH_TOKEN','')
        if not token:
            for p in ['../github_token.txt','github_token.txt','d:/策略/github_token.txt']:
                try: token=open(p).read().strip(); break
                except: pass
        chart_url=''
        if token:
            ts=pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            ctx2=ssl._create_unverified_context()
            h2={'Authorization':'Bearer '+token,'User-Agent':'YH04','Content-Type':'application/json'}
            api2=f'https://api.github.com/repos/{REPO}/contents/YH04/chart_{ts}.png'
            body2=json.dumps({'message':'YH04 chart','content':base64.b64encode(img_bytes).decode('ascii'),'branch':'main'}).encode()
            ur.urlopen(ur.Request(api2,data=body2,headers=h2,method='PUT'),timeout=15,context=ctx2)
            chart_url=f'https://cdn.jsdelivr.net/gh/{REPO}@main/YH04/chart_{ts}.png'

        body=(f'{action}\n{detail}\n副线: {sub_rank}')
        if warn: body+=f'\n{warn}'
        send_bark(f'YH04 {action}',body,chart_url)
        with open('_preview.png','wb')as f: f.write(img_bytes)
        print(f"完成! 图表: _preview.png")
    except Exception as e:
        print(f"失败: {e}"); import traceback; traceback.print_exc()
        send_bark('YH04信号失败',str(e)[:200])

if __name__=='__main__': main()
