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

    # ===== 120日迷你回测 =====
    lookback=120
    # 截取近120日数据
    main_sub=df_main.tail(lookback).reset_index(drop=True)
    cy_sub=dfs_g['创业板'].tail(lookback).reset_index(drop=True)
    main_close=raw[MAIN_NAME]['close'].tail(lookback).reset_index(drop=True)
    cy_close=raw['创业板']['close'].tail(lookback).reset_index(drop=True)

    INIT=1_000_000; cash=0.0; shares_main=INIT/main_close.iloc[0]*(1-0.0003)
    shares_cy=0.0; pos='MAIN'; peak=INIT; pbw=None; sc=2; ep=main_close.iloc[0]
    hse=0; last_rb=None; stopped=False; navs=[]; events=[]  # events记录换仓时间

    for i in range(lookback):
        mp=main_close.iloc[i]; cp=cy_close.iloc[i]
        row=main_sub.iloc[i]; crow=cy_sub.iloc[i]
        adj=row['adj']; rsi=row['rsi']; lo=row['lo']; up=row['up']

        nav=cash+shares_main*mp+shares_cy*cp
        if nav>peak: peak=nav
        dd=(nav-peak)/peak if peak>0 else 0
        if dd<-0.13 and (shares_main>0 or shares_cy>0):
            cash+=(shares_main*mp+shares_cy*cp)*(1-0.0003); shares_main=0; shares_cy=0
            peak=nav; pos=None; stopped=True

        if pd.isna(lo) or pd.isna(rsi): navs.append(nav); continue

        bw=(up-lo)/row['ma']if row['ma']>0 else 0.1
        exp=(pbw is not None and bw>pbw); pbw=bw
        bb_buy=(adj<=lo); bb_sell=(adj>=up); rsi_buy=(rsi<=30)
        if exp:
            sell_ok=(bb_sell and rsi>=65); buy_ok=(bb_buy or rsi_buy)
        else: buy_ok=(bb_buy or rsi_buy); sell_ok=(bb_sell or rsi>=70)

        if buy_ok or sell_ok: sc+=1
        if sc<2: navs.append(nav); continue

        if buy_ok and pos!='MAIN':
            if pos=='CY':
                cash+=shares_cy*cp*(1-0.0003); shares_cy=0
                events.append((i,'SELL_CY'))
            cash=max(cash,0); shares_main=cash/mp*(1-0.0003); cash=0; pos='MAIN'; ep=mp; stopped=False
            events.append((i,'BUY_MAIN'))
        elif sell_ok and pos=='MAIN':
            cash+=shares_main*mp*(1-0.0003); shares_main=0; pos=None
            events.append((i,'SELL_MAIN'))
        elif pos is None and not stopped:
            macd_ok=crow['macd_h']>0; bull_ok=cp>crow['ma20']
            days_since=(i-last_rb)if last_rb else 999
            if days_since>=5 and macd_ok and bull_ok:
                above_zero=crow['macd_line']>0; pos_pct=1.0 if above_zero else 0.3
                val=min(cash,nav*pos_pct)
                if val>100: shares_cy=val/cp*(1-0.0003); cash-=val; pos='CY'; hse=cp
                events.append((i,'BUY_CY'))
                last_rb=i
        elif pos=='CY':
            if cp>hse: hse=cp
            if cp<hse*0.9:
                cash+=shares_cy*cp*(1-0.0003); shares_cy=0; pos=None; stopped=True
                events.append((i,'STOP_CY'))

        navs.append(cash+shares_main*mp+shares_cy*cp)

    navs=np.array(navs); navs=navs/navs[0]

    # 打印交易明细
    print(f"\n  {'日期':<12} {'标的':<8} {'方向':<6} {'价格':>8} {'盈亏':>8}")
    entry_main=None; entry_cy=None
    for ei,etype in events:
        if etype=='BUY_MAIN':
            entry_main=(ei,main_close.iloc[ei])
            print(f"  {main_sub['date'].iloc[ei].strftime('%m-%d'):<12} {MAIN_NAME:<8} 买入   {main_close.iloc[ei]:>8.3f}  {'—':>8}")
        elif etype=='SELL_MAIN' and entry_main:
            pnl=(main_close.iloc[ei]/entry_main[1]-1)*100
            print(f"  {main_sub['date'].iloc[ei].strftime('%m-%d'):<12} {MAIN_NAME:<8} 卖出   {main_close.iloc[ei]:>8.3f}  {pnl:>+7.1f}%")
            entry_main=None
        elif etype=='BUY_CY':
            entry_cy=(ei,cy_close.iloc[ei])
            print(f"  {main_sub['date'].iloc[ei].strftime('%m-%d'):<12} 创业板    买入   {cy_close.iloc[ei]:>8.3f}  {'—':>8}")
        elif etype in ('SELL_CY','STOP_CY') and entry_cy:
            pnl=(cy_close.iloc[ei]/entry_cy[1]-1)*100
            tag='止损'if etype=='STOP_CY'else'卖出'
            print(f"  {main_sub['date'].iloc[ei].strftime('%m-%d'):<12} 创业板    {tag:<6} {cy_close.iloc[ei]:>8.3f}  {pnl:>+7.1f}%")
            entry_cy=None
    print(f"  近{lookback}日净值: {(navs[-1]-1)*100:+.1f}%")

    # ===== iPhone画图 6×12 =====
    fig=plt.figure(figsize=(6,12),facecolor='#FAFAFA')
    gs=fig.add_gridspec(4,1,height_ratios=[0.8,1.5,1.5,1.2],hspace=0.25,
                        left=0.06,right=0.94,top=0.97,bottom=0.03)

    # P0: 信息栏
    ax0=fig.add_subplot(gs[0]); ax0.axis('off')
    ax0.text(0,0.8,f'YH05 {r["date"].strftime("%Y-%m-%d")}',fontsize=15,fontweight='bold',color='#1A1A1A')
    ax0.text(0,0.3,f'{main_sig}{warn}',fontsize=16,fontweight='bold',color='#E67E22' if warn else '#1A1A1A')
    ax0.text(0,0.0,f'{sub_state}',fontsize=11,color='#555')

    # A股配色: 红涨绿跌
    cn_colors=mpf.make_marketcolors(up='#CC0000',down='#008800',edge='inherit',wick='inherit',volume='inherit')
    cn_style=mpf.make_mpf_style(marketcolors=cn_colors,gridstyle='')

    # P1: 红利低波K线(红)
    ax1=fig.add_subplot(gs[1])
    mpf.plot(main,type='candle',ax=ax1,volume=False,style=cn_style)
    ax1.set_title(f'{MAIN_NAME}  RSI{rsi:.1f}  BB{bb_pos:.0f}%',fontsize=11,loc='left',color='#CC2222')
    ax1.tick_params(labelsize=8); ax1.grid(True,alpha=0.15)
    # 买卖标记
    for ei,etype in events:
        if etype in ('BUY_MAIN','SELL_MAIN'):
            ax1.scatter(ei,main['Close'].iloc[ei]if etype=='BUY_MAIN' else main['High'].iloc[ei],
                       color='#CC2222'if etype=='BUY_MAIN'else'#008800',s=100,marker='^'if etype=='BUY_MAIN'else'v',
                       zorder=10,edgecolors='white',lw=1.5)

    # P2: 创业板K线(紫)
    ax2=fig.add_subplot(gs[2])
    mpf.plot(cy,type='candle',ax=ax2,volume=False,style=cn_style)
    ax2.set_title(f'创业板  MACD{macd_v:+.3f}  动量{cy_mom:+.1%}',fontsize=11,loc='left',color='#9B59B6')
    ax2.tick_params(labelsize=8); ax2.grid(True,alpha=0.15)
    # 买卖标记
    for ei,etype in events:
        if etype=='BUY_CY':
            ax2.scatter(ei,cy['Low'].iloc[ei],color='#CC0000',s=100,marker='^',zorder=10,edgecolors='white',lw=1.5)
        elif etype in ('STOP_CY','SELL_CY'):
            ax2.scatter(ei,cy['High'].iloc[ei],color='#008800',s=100,marker='v',zorder=10,edgecolors='white',lw=1.5)

    # P3: 策略收益曲线
    ax3=fig.add_subplot(gs[3])
    ax3.set_facecolor('#FFFFFF')
    nav_color='#CC2222' if navs[-1]>=1 else '#228B22'
    ax3.fill_between(range(len(navs)),1,navs,alpha=0.1,color=nav_color)
    ax3.plot(range(len(navs)),navs,color=nav_color,lw=1.8)
    ax3.axhline(y=1,color='#AAA',lw=0.8,ls='--')
    # 标记换仓事件(大号散点,红=红利,紫=创业)
    for ei,etype in events:
        y_pos=navs[ei]
        if etype=='BUY_CY':
            ax3.scatter(ei,y_pos,color='#9B59B6',s=90,marker='^',zorder=5,edgecolors='white',lw=1.5)
        elif etype=='STOP_CY':
            ax3.scatter(ei,y_pos,color='#9B59B6',s=90,marker='v',zorder=5,edgecolors='white',lw=1.5)
        elif etype=='BUY_MAIN':
            ax3.scatter(ei,y_pos,color='#CC2222',s=90,marker='^',zorder=5,edgecolors='white',lw=1.5)
        elif etype=='SELL_MAIN':
            ax3.scatter(ei,y_pos,color='#CC2222',s=90,marker='v',zorder=5,edgecolors='white',lw=1.5)
    ax3.set_xlim(-1,len(navs))
    ax3.set_title(f'策略净值 (近{lookback}日) {(navs[-1]-1)*100:+.1f}%',fontsize=11,loc='left',color=nav_color)
    ax3.tick_params(labelsize=8); ax3.grid(True,alpha=0.12)

    buf=io.BytesIO(); plt.savefig(buf,dpi=150,bbox_inches='tight',facecolor='#FAFAFA'); plt.close()
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
        if token: chart_url=upload_chart(token,img_bytes)

        body=(f'{sig}{warn}\n'
              f'{sub_state}\n'
              f'红利低波@{px:.3f} RSI{rsi:.1f} BB{bb_pos:.0f}%')
        send_bark(f'YH05 {sig}{warn}',body,chart_url)
        with open('_preview.png','wb') as f: f.write(img_bytes)
        print(f"完成! 图表: _preview.png")
    except Exception as e:
        print(f"失败: {e}"); import traceback; traceback.print_exc()
        send_bark('YH05信号失败',str(e)[:200])

if __name__=='__main__': main()
