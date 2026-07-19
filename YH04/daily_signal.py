# -*- coding: utf-8 -*-
"""YH04 每日自动信号 — 推送到手机Bark"""
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
        bb_buy=adj<=lo; bb_sell=adj>=up; rsi_buy=rsi<=RSI_L
        # 简化判断
        buy_ok=bb_buy or rsi_buy; sell_ok=bb_sell and rsi>=ERS
        if buy_ok: sig='买入'
        elif sell_ok: sig='卖出'
        else: sig='持有'

        # 成长排名
        g_idx=max(len(d) for d in dfs_g.values())-1
        scores={}
        for n in GROWTH:
            p2=min(g_idx,len(dfs_g[n])-1)
            v=dfs_g[n]['macd_h'].iloc[p2]
            if not pd.isna(v): scores[n]=v
        ranking=sorted(scores,key=scores.get,reverse=True)

        print(f"\n{MAIN_NAME}: {sig}  价格{row['close']:.3f}  RSI{rsi:.1f}  BB{bb_pos:.0f}%")
        print(f"成长MACD排名: {', '.join(f'{n}({scores[n]:.3f})' for n in ranking[:4])}")

        # 推送
        top=ranking[0] if ranking else '—'; top_macd=scores.get(top,0)
        rank_str = ', '.join(f'#{i+1}{n}' for i,n in enumerate(ranking[:3]))
        body=(f'{MAIN_NAME}: {sig}  价格{row["close"]:.3f}  RSI{rsi:.1f}\n'
              f'BB {bb_pos:.0f}%  |  成长MACD #{top_macd:+.3f}\n'
              f'排名: {rank_str}')
        send_bark(f'YH04 {MAIN_NAME} {sig}',body)
    except Exception as e:
        print(f"失败: {e}")
        send_bark('YH04 信号失败',str(e)[:200])

if __name__=='__main__': main()
