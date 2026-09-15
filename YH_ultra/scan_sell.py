# -*- coding: utf-8 -*-
"""分析亏损交易+测试改进卖出策略"""
import sys,io,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak,pandas as pd,numpy as np
warnings.filterwarnings('ignore')

STOCKS={'山东高速':'sh600350','渝农商行':'sh601077','皖通高速':'sh600012','江苏银行':'sh600919'}
INIT=1_000_000;COMM=0.0003;SLIP=0.0001;MAX_POS=0.25

BUY_PARAMS={
    '山东高速':{'rsi':45,'bb':0.25},
    '渝农商行':{'rsi':30,'bb':0.10},
    '皖通高速':{'rsi':38,'bb':0.12},
    '江苏银行':{'rsi':35,'bb':0.10},
}

def fetch():
    dfs={}
    for n,s in STOCKS.items():
        df=ak.stock_zh_a_daily(symbol=s,adjust='qfq');df['date']=pd.to_datetime(df['date'])
        dfs[n]=df[['date','open','high','low','close','volume']].sort_values('date').reset_index(drop=True)
    return dfs

def add_indicators(df):
    df=df.copy();c=df['close']
    df['ma20']=c.rolling(20).mean()
    df['bb_ma']=c.rolling(20).mean();df['bb_std']=c.rolling(20).std()
    df['bb_up']=df['bb_ma']+2*df['bb_std'];df['bb_lo']=df['bb_ma']-2*df['bb_std']
    d=c.diff();g=d.clip(lower=0);l=(-d).clip(lower=0)
    df['rsi']=100-100/(1+g.ewm(alpha=1/14,adjust=False).mean()/l.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan))
    return df

def check_buy(row,name):
    if pd.isna(row['bb_lo'])or pd.isna(row['rsi']):return False,0
    rsi=row['rsi'];c=row['close'];lo=row['bb_lo'];up=row['bb_up']
    if up<=lo:return False,0
    dist=(c-lo)/(up-lo)
    bp=BUY_PARAMS.get(name,{'rsi':42,'bb':0.25})
    sc=(1 if rsi<=bp['rsi'] else 0)+(1 if dist<=bp['bb'] else 0)
    if rsi<=30:sc+=1
    return sc>=1,sc

def run_with_sell(name, sell_fn):
    """返回所有交易明细"""
    raw=fetch();dfs={n:add_indicators(d)for n,d in raw.items()}
    dates=sorted(set.intersection(*[set(d['date'])for d in dfs.values()]))
    dates=[d for d in dates if d>=pd.Timestamp('2019-01-01')]
    cash=INIT;shares={n:0.0 for n in STOCKS};entry={n:0.0 for n in STOCKS};high={n:0.0 for n in STOCKS}
    navs=[];all_trades=[]

    for date in dates:
        px={n:raw[n][raw[n]['date']==date]['close'].iloc[0]for n in STOCKS if len(raw[n][raw[n]['date']==date])>0}

        for n in STOCKS:
            if shares[n]<=0:continue
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            if cp>high[n]:high[n]=cp
            pnl=cp/entry[n]-1;dd=cp/high[n]-1
            do,why=sell_fn(r.iloc[0],pnl,dd)
            if do:
                cash+=shares[n]*cp*(1-COMM-SLIP)
                all_trades.append({'date':date,'name':n,'entry':entry[n],'exit':cp,'pnl':pnl*100,'peak':high[n]/entry[n]-1,'reason':why})
                shares[n]=0;entry[n]=0;high[n]=0

        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)

        for n in STOCKS:
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            if shares[n]>0:continue
            ok,sc=check_buy(r.iloc[0],n)
            if not ok:continue
            val=min(cash,nav*MAX_POS)
            if val>5000:
                qty=val/cp*(1-COMM-SLIP)
                shares[n]=qty;cash-=val
                entry[n]=cp;high[n]=cp

        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)
        navs.append(nav)

    ndf=pd.DataFrame(navs,columns=['nav']);final=ndf['nav'].iloc[-1]
    ret=(final/INIT-1)*100;ann=((1+ret/100)**(252/len(ndf))-1)*100
    dr=ndf['nav'].pct_change().dropna();vol=dr.std()*np.sqrt(252)*100
    sr=(ann-2)/vol if vol>0 else 0
    mdd=((ndf['nav']-ndf['nav'].cummax())/ndf['nav'].cummax()).min()*100

    sells=[t for t in all_trades if t['pnl']!=0]
    pnls=np.array([s['pnl'] for s in sells])
    wr=(pnls>0).sum()/len(pnls)*100 if len(pnls)>0 else 0
    aw=pnls[pnls>0].mean()if(pnls>0).sum()>0 else 0
    al=pnls[pnls<0].mean()if(pnls<0).sum()>0 else 0
    return ann,ret,sr,mdd,wr,len(sells),aw,al,all_trades

# ======== 基准卖出 ========
def sell_now(row,pnl,dd):
    if pnl<=-0.10:return True,'硬止损'
    if dd<=-0.08:return True,'移动止损'
    if pnl>=0.20:return True,'止盈'
    return False,''

print("="*80)
print("  亏损交易分析 (当前策略)")
print("="*80)
_,_,_,_,_,_,_,_,trades=run_with_sell('当前',sell_now)
losing=[t for t in trades if t['pnl']<0]
print(f"总交易{len([t for t in trades if t['pnl']!=0])}笔, 亏损{len(losing)}笔")
print(f"\n亏损原因分布:")
from collections import Counter
reasons=Counter(t['reason'] for t in losing)
for r,c in reasons.most_common():
    print(f"  {r}: {c}笔")

# 按峰值分组
print(f"\n按峰值回撤分组:")
for label, lo, hi in [('从未盈利',-99,0),('微盈0~3%',0,3),('小盈3~8%',3,8),('中等8~15%',8,15),('大盈>15%',15,999)]:
    group=[t for t in losing if lo<=t['peak']*100<hi]
    if group:
        avg_pnl=np.mean([t['pnl'] for t in group])
        print(f"  峰值{label}: {len(group)}笔, 均价亏{avg_pnl:+.1f}%")

# ======== 改进方案测试 ========
print(f"\n{'='*80}")
print("  改进卖出策略测试")
print(f"{'='*80}")

# S1: 只有止盈+硬止损, 不用移动止损
def sell_s1(row,pnl,dd):
    if pnl<=-0.10:return True,'硬止损'
    if pnl>=0.20:return True,'止盈'
    return False,''

# S2: 移动止损放宽到12%
def sell_s2(row,pnl,dd):
    if pnl<=-0.10:return True,'硬止损'
    if dd<=-0.12:return True,'移动止损'
    if pnl>=0.20:return True,'止盈'
    return False,''

# S3: 浮盈>5%才启用移动止损(8%), 否则只看硬止损
def sell_s3(row,pnl,dd):
    if pnl<=-0.10:return True,'硬止损'
    if pnl>0.05 and dd<=-0.08:return True,'移动止损'
    if pnl>=0.20:return True,'止盈'
    return False,''

# S4: 阶梯移动止损
def sell_s4(row,pnl,dd):
    if pnl<=-0.10:return True,'硬止损'
    if pnl>0.12: ts=0.04
    elif pnl>0.05: ts=0.06
    else: ts=0.10
    if dd<=-ts:return True,f'移动止损{ts*100:.0f}%'
    if pnl>=0.20:return True,'止盈'
    return False,''

# S5: 放宽硬止损+止盈
def sell_s5(row,pnl,dd):
    if pnl<=-0.12:return True,'硬止损'
    if dd<=-0.08:return True,'移动止损'
    if pnl>=0.25:return True,'止盈'
    return False,''

tests=[
    ('S0 当前(HS10/TS8/TP20)', sell_now),
    ('S1 去移动止损',           sell_s1),
    ('S2 TS放宽到12%',          sell_s2),
    ('S3 盈利5%后才启动TS',      sell_s3),
    ('S4 阶梯TS(10/6/4%)',      sell_s4),
    ('S5 HS12/TP25',            sell_s5),
]

print(f"{'策略':<24} {'年化':>7} {'夏普':>6} {'回撤':>7} {'胜率':>6} {'交易':>5} {'均盈':>7} {'均亏':>7}")
print('─'*80)
best_sr=0;best_name=''
for nm,sf in tests:
    ann,ret,sr,mdd,wr,nt,aw,al,_=run_with_sell(nm,sf)
    mark=' <' if sr>best_sr else ''
    print(f'{nm:<24} {ann:>+6.1f}% {sr:>5.2f} {mdd:>+6.1f}% {wr:>5.0f}% {nt:>5} {aw:>+6.1f}% {al:>+6.1f}%{mark}')
    if sr>best_sr:best_sr=sr;best_name=nm
print(f'\n最佳: {best_name} (SR={best_sr:.2f})')

# ======== 改进买点: RSI拐头确认 ========
print(f"\n{'='*80}")
print("  改进买点: 渝农商行要求score>=2")
print(f"{'='*80}")

def check_buy_v3(row,name):
    if pd.isna(row['bb_lo'])or pd.isna(row['rsi']):return False,0
    rsi=row['rsi'];c=row['close'];lo=row['bb_lo'];up=row['bb_up']
    if up<=lo:return False,0
    dist=(c-lo)/(up-lo)
    bp=BUY_PARAMS.get(name,{'rsi':42,'bb':0.25})
    sc=(1 if rsi<=bp['rsi'] else 0)+(1 if dist<=bp['bb'] else 0)
    if rsi<=30:sc+=1
    min_score=2 if name=='渝农商行' else 1  # 渝农商行要求同时满足RSI+BB
    if sc<min_score:return False,0
    return True,sc

def run_with_buy_v3(name,sell_fn):
    raw=fetch();dfs={n:add_indicators(d)for n,d in raw.items()}
    dates=sorted(set.intersection(*[set(d['date'])for d in dfs.values()]))
    dates=[d for d in dates if d>=pd.Timestamp('2019-01-01')]
    cash=INIT;shares={n:0.0 for n in STOCKS};entry={n:0.0 for n in STOCKS};high={n:0.0 for n in STOCKS}
    navs=[];trades=[]

    for date in dates:
        px={n:raw[n][raw[n]['date']==date]['close'].iloc[0]for n in STOCKS if len(raw[n][raw[n]['date']==date])>0}

        for n in STOCKS:
            if shares[n]<=0:continue
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            if cp>high[n]:high[n]=cp
            pnl=cp/entry[n]-1;dd=cp/high[n]-1
            do,why=sell_fn(r.iloc[0],pnl,dd)
            if do:
                cash+=shares[n]*cp*(1-COMM-SLIP)
                trades.append(pnl*100)
                shares[n]=0;entry[n]=0;high[n]=0

        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)

        for n in STOCKS:
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            if shares[n]>0:continue
            ok,sc=check_buy_v3(r.iloc[0],n)
            if not ok:continue
            val=min(cash,nav*MAX_POS)
            if val>5000:
                qty=val/cp*(1-COMM-SLIP)
                shares[n]=qty;cash-=val
                entry[n]=cp;high[n]=cp

        nav=cash+sum(shares[n]*px.get(n,0)for n in STOCKS)
        navs.append(nav)

    ndf=pd.DataFrame(navs,columns=['nav']);final=ndf['nav'].iloc[-1]
    ret=(final/INIT-1)*100;ann=((1+ret/100)**(252/len(ndf))-1)*100
    dr=ndf['nav'].pct_change().dropna();vol=dr.std()*np.sqrt(252)*100
    sr=(ann-2)/vol if vol>0 else 0
    mdd=((ndf['nav']-ndf['nav'].cummax())/ndf['nav'].cummax()).min()*100
    sells=np.array([t for t in trades if t!=0])
    wr=(sells>0).sum()/len(sells)*100 if len(sells)>0 else 0
    aw=sells[sells>0].mean()if(sells>0).sum()>0 else 0
    al=sells[sells<0].mean()if(sells<0).sum()>0 else 0
    return ann,ret,sr,mdd,wr,len(sells),aw,al

# 重新跑基准
ann0,ret0,sr0,mdd0,wr0,nt0,aw0,al0,_=run_with_sell('基准',sell_now)
ann2,ret2,sr2,mdd2,wr2,nt2,aw2,al2=run_with_buy_v3('渝农score2',sell_now)
print(f"  基准:           年化{ann0:+.1f}% 夏普{sr0:.2f} 回撤{mdd0:+.1f}% 胜率{wr0:.0f}% 交易{nt0}")
print(f"  渝农score>=2:   年化{ann2:+.1f}% 夏普{sr2:.2f} 回撤{mdd2:+.1f}% 胜率{wr2:.0f}% 交易{nt2}")

