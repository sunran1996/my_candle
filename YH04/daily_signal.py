# -*- coding: utf-8 -*-
"""YH04 每日信号 — 主线+副线全状态推送"""
import sys, io, os, json, ssl, time, base64, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import akshare as ak, pandas as pd, numpy as np
import urllib.request as ur
warnings.filterwarnings('ignore')

MAIN_SYM='sh512890'; MAIN_NAME='红利低波'
GROWTH={'创业板':'sz159915','科创50':'sh588000','人工智能':'sh515070','半导体':'sh512480'}
BB_P=45; BB_S=2.0; RSI_P=14; RSI_L=30; RSI_H=70; ERS=65; BA=0.001
BARK_KEY='eoq8G58fJtDDFxHjhNueGH'; REPO='sunran1996/my_candle'

def fetch():
    dfs={}
    for n,s in {**GROWTH,MAIN_NAME:MAIN_SYM}.items():
        df=ak.fund_etf_hist_sina(symbol=s); df['date']=pd.to_datetime(df['date'])
        dfs[n]=df[['date','close']].sort_values('date').reset_index(drop=True)
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
    return df

def send_bark(title,body):
    try:
        data=json.dumps({'title':title,'body':body}).encode()
        ur.urlopen(ur.Request(f'https://api.day.app/{BARK_KEY}',data=data,
                   headers={'Content-Type':'application/json'}),timeout=10)
        print("已推送到手机")
    except Exception as e: print(f"推送失败: {e}")

def main():
    try:
        print("获取数据...")
        raw=fetch(); df_main=add_main(raw[MAIN_NAME])
        dfs_g={n:add_growth(d) for n,d in raw.items() if n!=MAIN_NAME}
        idx=-1; pos=len(df_main)+idx; row=df_main.iloc[pos]; date=row['date']
        adj,rsi,lo,up=row['adj'],row['rsi'],row['lo'],row['up']
        bb_pos=(adj-lo)/(up-lo)*100 if up>lo else 50

        # YH02主线信号
        bb_buy=adj<=lo; bb_sell=adj>=up; rsi_buy=rsi<=RSI_L
        buy_ok=bb_buy or rsi_buy
        sell_ok=(bb_sell and rsi>=ERS)  # simplified expansion check
        if buy_ok: main_sig='买入'
        elif sell_ok: main_sig='卖出'
        else: main_sig='持有'

        # 成长MACD排名
        g_idx=max(len(d) for d in dfs_g.values())-1
        scores={}
        for n in GROWTH:
            p2=min(g_idx,len(dfs_g[n])-1)
            v=dfs_g[n]['macd_h'].iloc[p2]
            if not pd.isna(v): scores[n]=v
        ranking=sorted(scores,key=scores.get,reverse=True)
        leader=ranking[0] if ranking else '—'
        leader_macd=scores.get(leader,0)

        # 判断当前状态
        main_px=raw[MAIN_NAME]['close'].iloc[len(raw[MAIN_NAME])+idx]
        leader_px=raw[leader]['close'].iloc[min(len(raw[leader])-1,g_idx)] if leader!='—' else 0

        if main_sig in ('买入','持有'):
            state=f'🔵 满仓{MAIN_NAME}'
            suggest=f'{main_sig}{MAIN_NAME} @{main_px:.3f}'
            sub_info=f'副线待命 | 最强{leader} MACD{leader_macd:+.3f}'
        elif main_sig=='卖出':
            if leader_macd>0:
                state=f'🟢 买入{leader}'
                suggest=f'主线{MAIN_NAME}已卖出 → 买入{leader} @{leader_px:.3f} | 移动止损-10%'
                sub_info=f'MACD{leader_macd:+.3f} 动量{dfs_g[leader]["mom"].iloc[g_idx]:+.1%}'
            else:
                state=f'🔴 现金等待'
                suggest=f'{MAIN_NAME}已卖出 | 四指数MACD全负({leader_macd:+.3f}) | 持币观望'
                sub_info=f'等任一MACD翻红再进场'
        else:
            state='—'; suggest='—'; sub_info='—'

        # 终端输出
        print(f"\n{'='*60}")
        print(f"  YH04  {date.strftime('%Y-%m-%d')}  状态: {state}")
        print(f"{'='*60}")
        print(f"  主线 {MAIN_NAME}: {main_sig}  价格{main_px:.3f}  RSI{rsi:.1f}  BB{bb_pos:.0f}%")
        print(f"  建议: {suggest}")
        print(f"  {sub_info}")
        print(f"{'─'*60}")
        print(f"  成长MACD排名:")
        for i,n in enumerate(ranking[:4]):
            p=raw[n]['close'].iloc[min(len(raw[n])-1,g_idx)]
            m=dfs_g[n]['mom'].iloc[min(len(dfs_g[n])-1,g_idx)]
            macd=scores.get(n,0)
            tag='  ← 触发' if (main_sig=='卖出' and i==0 and macd>0) else ''
            print(f"    #{i+1} {n:<6} {p:.3f}  动量{m:+.1%}  MACD{macd:+.3f}{tag}")
        print(f"{'='*60}")

        # 推送
        emoji={'买入':'🔴','卖出':'🟢','持有':'⚪'}.get(main_sig,'⚪')
        body=(f'{emoji} {MAIN_NAME}: {main_sig}  价格{main_px:.3f}  RSI{rsi:.1f}\n'
              f'状态: {state}\n'
              f'{suggest}\n'
              f'{sub_info}')
        send_bark(f'YH04 {state[:12]}',body)
    except Exception as e:
        print(f"失败: {e}"); send_bark('YH04 信号失败',str(e)[:200])

if __name__=='__main__': main()
