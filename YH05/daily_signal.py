# -*- coding: utf-8 -*-
"""YH05 每日信号 + K线收益图"""
import sys, io, os, json, ssl, time, base64, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import akshare as ak, pandas as pd, numpy as np
import urllib.request as ur
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

_fonts=[f.name for f in fm.fontManager.ttflist]
CN='WenQuanYi Zen Hei' if 'WenQuanYi Zen Hei' in _fonts else ('SimHei'if'SimHei'in _fonts else'DejaVu Sans')
plt.rcParams['font.sans-serif']=[CN]; plt.rcParams['axes.unicode_minus']=False

MAIN_SYM='sh512890'; MAIN_NAME='红利低波'
GROWTH={'创业板':'sz159915'}
BB_P=45; BB_S=2.0; RSI_P=14; RSI_L=30; RSI_H=70; ERS=65; BA=0.001
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

def gen_chart(raw,df_main,dfs_g,both=False):
    """双K线图: 红利低波 + 创业板"""
    main=raw[MAIN_NAME].tail(120).copy()
    main=main.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
    main=main.set_index('date')[['Open','High','Low','Close','Volume']]

    cy=raw['创业板'].tail(120).copy()
    cy=cy.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
    cy=cy.set_index('date')[['Open','High','Low','Close','Volume']]

    # 主线信号
    r=df_main.iloc[-1]; adj=r['adj']; rsi=r['rsi']; lo=r['lo']; up=r['up']
    bb_pos=(adj-lo)/(up-lo)*100 if up>lo else 50
    main_px=raw[MAIN_NAME]['close'].iloc[-1]
    bb_buy=adj<=lo; bb_sell=adj>=up; rsi_buy=rsi<=RSI_L
    buy_ok=bb_buy or rsi_buy; sell_ok=(bb_sell and rsi>=ERS)
    main_sig='买入' if buy_ok else ('卖出' if sell_ok else '持有')

    # 成长MACD
    g_idx=len(dfs_g['创业板'])-1; g_row=dfs_g['创业板'].iloc[g_idx]
    macd_v=g_row['macd_h']; macd_line=g_row['macd_line']
    cy_px=raw['创业板']['close'].iloc[-1]; cy_mom=g_row['mom']

    # 预警
    warn=''
    if main_sig=='持有':
        if bb_pos<35 or rsi<45: warn=' ⚠ 接近买入'
        elif bb_pos>65 or rsi>60: warn=' ⚠ 接近卖出'

    # 副线状态
    if buy_ok:
        sub_state=f'🔴 清创业板,满仓红利'
    elif sell_ok and macd_v>0:
        above_zero=macd_line>0
        sub_state=f'🟢 换仓创业板 | {"水上满仓" if above_zero else "水下3成"}'
    elif not sell_ok and not buy_ok and macd_v>0:
        stop_px=cy_px*0.9
        sub_state=f'🟡 持有创业板@{cy_px:.3f} 止损{stop_px:.3f}'
    elif sell_ok:
        sub_state=f'⚫ 现金等待 | MACD全负'
    else:
        sub_state=f'⚪ 红利低波@{main_px:.3f} | 创业板MACD{macd_v:+.3f}'

    # ===== 画图 =====
    fig=plt.figure(figsize=(12,10),facecolor='#FAFAFA')

    # 顶部信息栏
    ax_info=fig.add_axes([0.05,0.88,0.9,0.10]); ax_info.axis('off')
    ax_info.text(0,0.7,f'YH05 {r["date"].strftime("%Y-%m-%d")}',fontsize=14,fontweight='bold',color='#1A1A1A')
    ax_info.text(0,0.2,f'{main_sig}{warn} | {sub_state}',fontsize=12,color='#E67E22' if warn else '#555')

    # 红利低波K线(上)
    ax1=fig.add_axes([0.05,0.48,0.9,0.37])
    mpf.plot(main,type='candle',ax=ax1,volume=False,style='charles')
    ax1.set_title(f'{MAIN_NAME}  RSI{rsi:.1f}  BB{bb_pos:.0f}%',fontsize=10,loc='left',color='#9B59B6')
    ax1.tick_params(labelsize=8); ax1.grid(True,alpha=0.15)
    # BB带
    ax1.axhline(y=main_px,color='#9B59B6',lw=0.5,ls='--',alpha=0.3)

    # 创业板K线(下)
    ax2=fig.add_axes([0.05,0.08,0.9,0.37])
    mpf.plot(cy,type='candle',ax=ax2,volume=False,style='charles')
    ax2.set_title(f'创业板  MACD{macd_v:+.3f}  动量{cy_mom:+.1%}',fontsize=10,loc='left',color='#E74C3C')
    ax2.tick_params(labelsize=8); ax2.grid(True,alpha=0.15)

    buf=io.BytesIO(); plt.savefig(buf,dpi=120,bbox_inches='tight',facecolor='#FAFAFA'); plt.close()
    return buf.getvalue(), main_sig, main_px, rsi, bb_pos, sub_state, warn

def upload_chart(token,img_bytes):
    ts=pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    fn=f'chart_{ts}.png'
    ctx=ssl._create_unverified_context()
    h={'Authorization':'Bearer '+token,'User-Agent':'YH05'}
    api=f'https://api.github.com/repos/{REPO}/contents/YH05/{fn}'
    sha=None
    try:
        r=json.loads(ur.urlopen(ur.Request(api,headers=h),timeout=10,context=ctx).read())
        sha=r.get('sha')
    except: pass
    body=json.dumps({'message':'YH05 chart','content':base64.b64encode(img_bytes).decode('ascii'),'branch':'main',**({'sha':sha}if sha else{})}).encode()
    ur.urlopen(ur.Request(api,data=body,headers={**h,'Content-Type':'application/json'},method='PUT'),timeout=15,context=ctx)
    return f'https://cdn.jsdelivr.net/gh/{REPO}@main/YH05/{fn}'

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
                for code,name in [('512890','红利低波'),('159915','创业板')]:
                    s=spot[spot['代码']==code]
                    if len(s)>0:
                        rt=float(s['最新价'].iloc[0])
                        old=raw[name]['close'].iloc[-1]
                        raw[name].loc[raw[name].index[-1],'close']=rt
                        raw[name].loc[raw[name].index[-1],'date']=pd.Timestamp.now()
                        print(f'  {name} {old:.4f}→实时{rt:.4f}')
                df_main=add_main(raw[MAIN_NAME])
                dfs_g={n:add_growth(d) for n,d in raw.items() if n!=MAIN_NAME}
            except Exception as e: print(f'  实时行情失败: {e}')

        print("生成图表...")
        img_bytes,sig,px,rsi,bb_pos,sub_state,warn=gen_chart(raw,df_main,dfs_g)

        token=os.environ.get('GH_TOKEN','')
        if not token:
            for p in ['../github_token.txt','github_token.txt','d:/策略/github_token.txt']:
                try: token=open(p).read().strip(); break
                except: pass
        chart_url=''
        if token:
            chart_url=upload_chart(token,img_bytes)
            print(f'  {chart_url}')

        body=(f'{sig}{warn}\n'
              f'{sub_state}\n'
              f'红利低波@{px:.3f} RSI{rsi:.1f} BB{bb_pos:.0f}%')
        send_bark(f'YH05 {sig}{warn}',body,chart_url)
        print("完成!")
    except Exception as e:
        print(f"失败: {e}"); import traceback; traceback.print_exc()
        send_bark('YH05信号失败',str(e)[:200])

if __name__=='__main__': main()
