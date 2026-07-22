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

MAIN_SYM='sh512890'; MAIN_NAME='红利低波'
GROWTH={'创业板':'sz159915','科创50':'sh588000','人工智能':'sh515070','半导体':'sh512480'}
BB_P=45;BB_S=2.0;RSI_P=14;RSI_L=30;RSI_H=70;ERS=65;BA=0.001
BARK_KEYS=['eoq8G58fJtDDFxHjhNueGH','WtAJhZtoGpU44fAiJCfJmb']
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
    df['ma']=df['adj'].rolling(BB_P).mean(); df['std']=df['adj'].rolling(BB_P).std()
    df['up']=df['ma']+BB_S*df['std']; df['lo']=df['ma']-BB_S*df['std']
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

        # 信号
        idx=-1; pos=len(df_main)+idx; row=df_main.iloc[pos]; date=row['date']
        adj,rsi,lo,up=row['adj'],row['rsi'],row['lo'],row['up']
        bb_pos=(adj-lo)/(up-lo)*100 if up>lo else 50; main_px=raw[MAIN_NAME]['close'].iloc[pos]
        bb_buy=adj<=lo; bb_sell=adj>=up; rsi_buy=rsi<=RSI_L
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

        # ===== K线图(动态) =====
        lookback=120
        cn_c=mpf.make_marketcolors(up='#CC0000',down='#008800',edge='inherit',wick='inherit',volume='inherit')
        cn_s=mpf.make_mpf_style(marketcolors=cn_c,gridstyle='')

        # 判断画哪个标的
        if sell_ok and leader_macd>0 and leader in raw:
            # 持有成长: 画创业板K线+MACD
            plot_name=leader; plot_raw=raw[leader]
            ohlc=plot_raw.tail(lookback).copy()
            ohlc=ohlc.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            ohlc=ohlc.set_index('date')[['Open','High','Low','Close','Volume']]
            # MACD数据
            g_df=dfs_g[leader].tail(lookback)
            macd_data=g_df['macd_h'].values; macd_line=g_df['macd_line'].values
            macd_sig=g_df['macd_s'].values if'macd_s'in g_df.columns else macd_line
            ap_macd=mpf.make_addplot(macd_line,panel=1,color='#CC2222',width=1.2,ylabel='MACD')
            ap_hist=mpf.make_addplot(macd_data,panel=1,type='bar',color='#CC2222',alpha=0.5)
            ap_zero=mpf.make_addplot(np.zeros(len(macd_data)),panel=1,color='#888',width=0.5)
            panels=dict(panel_ratios=(2,1))
            fig,axes=mpf.plot(ohlc,type='candle',volume=False,style=cn_s,addplot=[ap_macd,ap_hist,ap_zero],
                              returnfig=True,figsize=(6,8),panel_ratios=(2,1))
            axes[0].set_title(f'{plot_name} MACD{leader_macd:+.3f}',fontsize=10,loc='left',color='#9B59B6')
            axes[0].tick_params(labelsize=7); axes[0].grid(True,alpha=0.12)
            fig.suptitle(f'YH04 {date.strftime("%Y-%m-%d")}  {action}',fontsize=13,fontweight='bold',y=0.98)
        else:
            # 持有红利低波: 画K线+BB双轨
            ohlc=raw[MAIN_NAME].tail(lookback).copy()
            ohlc=ohlc.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            ohlc=ohlc.set_index('date')[['Open','High','Low','Close','Volume']]
            # BB轨道
            bb_adj=df_main['adj'].tail(lookback).values
            bb_ma=df_main['ma'].tail(lookback).values; bb_up=df_main['up'].tail(lookback).values; bb_lo=df_main['lo'].tail(lookback).values
            ap_ma=mpf.make_addplot(bb_ma,color='#888',width=0.8,linestyle='--')
            ap_up=mpf.make_addplot(bb_up,color='#9B59B6',width=0.6,linestyle='--')
            ap_lo=mpf.make_addplot(bb_lo,color='#9B59B6',width=0.6,linestyle='--')
            fig,axes=mpf.plot(ohlc,type='candle',volume=False,style=cn_s,addplot=[ap_ma,ap_up,ap_lo],
                              returnfig=True,figsize=(6,8))
            axes[0].set_title(f'{MAIN_NAME} RSI{rsi:.0f} BB{bb_pos:.0f}%',fontsize=10,loc='left',color='#CC2222')
            axes[0].tick_params(labelsize=7); axes[0].grid(True,alpha=0.12)
            fig.suptitle(f'YH04 {date.strftime("%Y-%m-%d")}  {action}',fontsize=13,fontweight='bold',y=0.98)

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
