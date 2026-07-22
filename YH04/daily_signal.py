# -*- coding: utf-8 -*-
"""YH04 每日信号 — 三指令: 买入红利/换仓成长/不动"""
import sys, io, os, json, ssl, time, base64, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import akshare as ak, pandas as pd, numpy as np
import urllib.request as ur
warnings.filterwarnings('ignore')

MAIN_SYM='sh512890'; MAIN_NAME='红利低波'
GROWTH={'创业板':'sz159915','科创50':'sh588000','人工智能':'sh515070','半导体':'sh512480'}
BB_P=45; BB_S=2.0; RSI_P=14; RSI_L=30; RSI_H=70; ERS=65
BARK_KEYS=['eoq8G58fJtDDFxHjhNueGH','WtAJhZtoGpU44fAiJCfJmb']

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
    for bk in BARK_KEYS:
        try:
            data=json.dumps({'title':title,'body':body}).encode()
            ur.urlopen(ur.Request(f'https://api.day.app/{bk}',data=data,
                       headers={'Content-Type':'application/json'}),timeout=10)
        except Exception as e: print(f"推送{bk[:8]}失败: {e}")
    print("已推送")

def main():
    try:
        print("获取数据...")
        raw=fetch(); df_main=add_main(raw[MAIN_NAME])
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
                        if rt>0 and abs(rt-old)/old<0.05:
                            raw[name].loc[raw[name].index[-1],'close']=rt
                            raw[name].loc[raw[name].index[-1],'date']=pd.Timestamp.now()
                            print(f'  {name} 实时价: {rt:.4f}')
                df_main=add_main(raw[MAIN_NAME])
            except Exception as e: print(f'  实时行情失败: {e}')
        dfs_g={n:add_growth(d) for n,d in raw.items() if n!=MAIN_NAME}
        idx=-1; pos=len(df_main)+idx; row=df_main.iloc[pos]; date=row['date']
        adj,rsi,lo,up=row['adj'],row['rsi'],row['lo'],row['up']
        bb_pos=(adj-lo)/(up-lo)*100 if up>lo else 50
        main_px=raw[MAIN_NAME]['close'].iloc[len(raw[MAIN_NAME])+idx]

        # 主线信号
        bb_buy=adj<=lo; bb_sell=adj>=up; rsi_buy=rsi<=RSI_L
        buy_ok=bb_buy or rsi_buy; sell_ok=(bb_sell and rsi>=ERS)
        main_sig='买入' if buy_ok else ('卖出' if sell_ok else '持有')

        # 成长MACD
        g_idx=max(len(d) for d in dfs_g.values())-1
        scores={}
        for n in GROWTH:
            p2=min(g_idx,len(dfs_g[n])-1); v=dfs_g[n]['macd_h'].iloc[p2]
            if not pd.isna(v): scores[n]=v
        ranking=sorted(scores,key=scores.get,reverse=True)
        leader=ranking[0] if ranking else '—'
        leader_macd=scores.get(leader,0)
        leader_px=raw[leader]['close'].iloc[min(len(raw[leader])-1,g_idx)] if leader!='—' else 0

        # 三指令
        if buy_ok:
            action=f'买入红利低波'
            detail=f'全仓{MAIN_NAME} @{main_px:.3f} | RSI{rsi:.1f} BB{bb_pos:.0f}%'
        elif sell_ok and leader_macd>0:
            action=f'换仓{leader}'
            detail=f'卖{MAIN_NAME}@{main_px:.3f} → 买{leader}@{leader_px:.3f} | 止损-10% | MACD{leader_macd:+.3f}'
        elif sell_ok:
            action=f'卖出红利低波,持币'
            detail=f'卖{MAIN_NAME}@{main_px:.3f} | MACD全负 | 等翻红再进场'
        else:
            action=f'不动'
            detail=f'{MAIN_NAME}@{main_px:.3f} RSI{rsi:.1f} BB{bb_pos:.0f}% | 副线{leader} MACD{leader_macd:+.3f}{" 翻红可追" if leader_macd>0 else ""}'

        sub_rank=' > '.join(f'{n}({scores[n]:+.3f})' for n in ranking[:3])

        print(f"\n{'='*60}")
        print(f"  YH04  {date.strftime('%Y-%m-%d')}")
        print(f"  操作: {action}")
        print(f"  {detail}")
        print(f"{'─'*60}")
        for i,n in enumerate(ranking[:4]):
            p=raw[n]['close'].iloc[min(len(raw[n])-1,g_idx)]
            m=dfs_g[n]['mom'].iloc[min(len(dfs_g[n])-1,g_idx)]
            macd=scores.get(n,0)
            tag=' ← 可买' if (sell_ok and i==0 and macd>0) else ''
            print(f"  #{i+1} {n:<6} {p:.3f}  动量{m:+.1%}  MACD{macd:+.3f}{tag}")
        print(f"{'='*60}")

        body=(f'{action}\n{detail}\n副线: {sub_rank}')
        send_bark(f'YH04 {action}',body)
    except Exception as e:
        print(f"失败: {e}"); send_bark('YH04信号失败',str(e)[:200])

if __name__=='__main__': main()
