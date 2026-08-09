# -*- coding: utf-8 -*-
"""YH-1.0 DCA 每月2万注入"""
import sys,io,warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import akshare as ak,pandas as pd,numpy as np
warnings.filterwarnings('ignore')

CORE_STOCKS={'山东高速':'sh600350','渝农商行':'sh601077','皖通高速':'sh600012','江苏银行':'sh600919'}
ETF_STOCKS={'创业板':'sz159915'}
ALL_STOCKS={**CORE_STOCKS,**ETF_STOCKS}
INIT=1_000_000;COMM=0.0003;SLIP=0.0001;MAX_POS=0.25;DCA=20000
TRAIL_STOP=0.07;HARD_STOP=0.10;COOLDOWN=20
BUY_PARAMS={
    '山东高速':{'rsi':45,'bb':0.25,'tp':0.15,'tp_hi':0.20},
    '渝农商行':{'rsi':30,'bb':0.10,'tp':0.20,'tp_hi':0.25},
    '皖通高速':{'rsi':38,'bb':0.12,'tp':0.20,'tp_hi':0.25},
    '江苏银行':{'rsi':35,'bb':0.10,'tp':0.15,'tp_hi':0.20},
    '创业板':{'tp':0.10,'tp_hi':0.15},
}

def fetch():
    dfs={}
    for n,sym in ALL_STOCKS.items():
        if sym.startswith('sz159')or sym.startswith('sh510'):
            df=ak.fund_etf_hist_sina(symbol=sym)
        else:
            df=ak.stock_zh_a_daily(symbol=sym,adjust='qfq')
        df['date']=pd.to_datetime(df['date'])
        dfs[n]=df[['date','open','high','low','close','volume']].sort_values('date').reset_index(drop=True)
    return dfs

def add_indicators(df):
    df=df.copy();c=df['close']
    df['bb_ma']=c.rolling(20).mean();df['bb_std']=c.rolling(20).std()
    df['bb_up']=df['bb_ma']+2*df['bb_std'];df['bb_lo']=df['bb_ma']-2*df['bb_std']
    df['bb_up_d2']=df['bb_up'].diff().diff()
    ema12=c.ewm(span=12,adjust=False).mean();ema26=c.ewm(span=26,adjust=False).mean()
    df['macd_dif']=ema12-ema26;df['macd_dea']=df['macd_dif'].ewm(span=9,adjust=False).mean()
    df['macd_hist']=2*(df['macd_dif']-df['macd_dea'])
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

def run(dca_mode):
    cash=INIT;shares={n:0.0 for n in ALL_STOCKS};entry={n:0.0 for n in ALL_STOCKS};high={n:0.0 for n in ALL_STOCKS}
    accel={n:False for n in ALL_STOCKS};cooldown={n:0 for n in ALL_STOCKS}
    navs=[];costs=[];trades=[];total_in=INIT;last_month=None

    for date in dates:
        if dca_mode:
            m=date.month
            if last_month is not None and m!=last_month:
                cash+=DCA;total_in+=DCA
            last_month=m

        sold_today={n:False for n in ALL_STOCKS}
        px={n:raw[n][raw[n]['date']==date]['close'].iloc[0]for n in ALL_STOCKS if len(raw[n][raw[n]['date']==date])>0}

        # 卖出
        for n in ALL_STOCKS:
            if shares[n]<=0:continue
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            if cp>high[n]:high[n]=cp
            pnl=cp/entry[n]-1;dd=cp/high[n]-1
            tp=BUY_PARAMS[n]['tp'];tp_hi=BUY_PARAMS[n]['tp_hi']
            do=False;sell_px=cp
            if pnl<=-HARD_STOP:do=True
            elif accel[n]:
                if pnl>=tp_hi:do=True
                elif dd<=-TRAIL_STOP:
                    floor=entry[n]*(1+tp);stop_px=max(high[n]*(1-TRAIL_STOP),floor)
                    if cp<=stop_px:do=True;sell_px=max(cp,floor)
            elif dd<=-TRAIL_STOP:do=True
            elif pnl>=tp:
                d2=r.iloc[0].get('bb_up_d2')
                if not pd.isna(d2) and d2>0:accel[n]=True
                else:do=True
            if do:
                cash+=shares[n]*sell_px*(1-COMM-SLIP)
                trades.append((sell_px/entry[n]-1)*100)
                shares[n]=0;entry[n]=0;high[n]=0;accel[n]=False;sold_today[n]=True
                if sell_px/entry[n]-1<=-HARD_STOP:cooldown[n]=COOLDOWN

        nav_val=cash+sum(shares[n]*px.get(n,0)for n in ALL_STOCKS)
        for n in ALL_STOCKS:
            if cooldown[n]>0:cooldown[n]-=1

        # 核心买入 + 换仓
        for n in CORE_STOCKS:
            if sold_today[n]:continue
            if shares[n]>0:continue
            if cooldown[n]>0:continue
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp<=0 or len(r)==0:continue
            ok,sc=check_buy(r.iloc[0],n)
            if not ok:continue
            target_val=nav_val*MAX_POS
            if cash<target_val:
                cy='创业板';cy_px=px.get(cy,0)
                if shares.get(cy,0)>0 and cy_px>0:
                    need=target_val-cash
                    if need>=5000:
                        cy_val=shares[cy]*cy_px
                        sell_val=min(need,cy_val);sell_qty=sell_val/cy_px
                        sell_cost=sell_val*(COMM+SLIP)
                        shares[cy]-=sell_qty;cash+=sell_val-sell_cost
                        if shares[cy]<1e-8:shares[cy]=0;entry[cy]=0;high[cy]=0;accel[cy]=False
            val=min(cash,target_val)
            if val>5000:
                qty=val/cp*(1-COMM-SLIP);shares[n]=qty;cash-=val
                entry[n]=cp;high[n]=cp

        # 创业板买入
        n='创业板';core_held=sum(1 for nn in CORE_STOCKS if shares[nn]>0)
        if core_held<2 and shares[n]<=0 and cooldown[n]<=0:
            cp=px.get(n,0);r=dfs[n][dfs[n]['date']==date]
            if cp>0 and len(r)>0:
                row=r.iloc[0]
                dif=row.get('macd_dif');dea=row.get('macd_dea');hist=row.get('macd_hist')
                if (not pd.isna(dif))and(not pd.isna(dea))and dif>dea:
                    hist_recent=dfs[n]['macd_hist'].iloc[-40:].dropna()
                    max_hist=hist_recent.abs().max()if len(hist_recent)>10 else 0
                    strength=abs(hist)/max_hist if max_hist>0 else 0
                    pos_frac=0.50 if strength>0.3 else 0.25
                    val=min(cash,nav_val*pos_frac)
                    if val>5000:
                        qty=val/cp*(1-COMM-SLIP);shares[n]=qty;cash-=val
                        entry[n]=cp;high[n]=cp

        nav_val=cash+sum(shares[n]*px.get(n,0)for n in ALL_STOCKS)
        navs.append(nav_val);costs.append(total_in)

    ndf=pd.DataFrame({'date':dates,'nav':navs,'cost':costs})
    return ndf,total_in,trades

print("获取数据...")
raw=fetch();dfs={n:add_indicators(d)for n,d in raw.items()}
dates=sorted(set.intersection(*[set(d['date'])for d in dfs.values()]))
dates=[d for d in dates if d>=pd.Timestamp('2019-01-01')]
print("dates: %s ~ %s, %d days" % (dates[0].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d'), len(dates)))

print("\n回测 纯策略...")
ndf_pure,total_pure,tr_pure=run(dca_mode=False)
fin_pure=ndf_pure['nav'].iloc[-1];ret_pure=(fin_pure/INIT-1)*100
ann_pure=((1+ret_pure/100)**(252/len(ndf_pure))-1)*100
dr_pure=ndf_pure['nav'].pct_change().dropna();vol_pure=dr_pure.std()*np.sqrt(252)*100
sr_pure=(ann_pure-2)/vol_pure if vol_pure>0 else 0
mdd_pure=((ndf_pure['nav']-ndf_pure['nav'].cummax())/ndf_pure['nav'].cummax()).min()*100
pnls_pure=np.array(tr_pure);wr_pure=(pnls_pure>0).sum()/len(pnls_pure)*100 if len(pnls_pure)>0 else 0

print("\n回测 +DCA 2万/月...")
ndf_dca,total_dca,tr_dca=run(dca_mode=True)
fin_dca=ndf_dca['nav'].iloc[-1];ret_dca=(fin_dca/total_dca-1)*100
dr_dca=ndf_dca['nav'].pct_change().dropna();vol_dca=dr_dca.std()*np.sqrt(252)*100
app_ann=((fin_dca/INIT)**(252/len(ndf_dca))-1)*100
sr_dca=(app_ann-2)/vol_dca if vol_dca>0 else 0
mdd_dca=((ndf_dca['nav']-ndf_dca['nav'].cummax())/ndf_dca['nav'].cummax()).min()*100
pnls_dca=np.array(tr_dca);wr_dca=(pnls_dca>0).sum()/len(pnls_dca)*100 if len(pnls_dca)>0 else 0
dca_roi=(fin_dca-total_dca)/(total_dca-INIT)*100

print("\n" + "="*65)
print("  YH-1.0  DCA每月2万注入")
print("  " + "─"*40)
print("  纯策略:")
print("    终值%.0f万  年化%+.1f%%  夏普%.2f  回撤%+.1f%%  交易%d笔  胜率%.0f%%" % (
    fin_pure/10000, ann_pure, sr_pure, mdd_pure, len(tr_pure), wr_pure))
print("  +DCA注入:")
print("    终值%.0f万  总投入%.0f万  净利%.0f万" % (fin_dca/10000, total_dca/10000, (fin_dca-total_dca)/10000))
print("    表观年化%+.1f%%  夏普%.2f  回撤%+.1f%%  交易%d笔  胜率%.0f%%" % (
    app_ann, sr_dca, mdd_dca, len(tr_dca), wr_dca))
print("    DCA增量ROI: %+.0f%%" % dca_roi)

# 按年
ndf_dca['year']=ndf_dca['date'].dt.year
print("\n  DCA 按年:")
for yr,grp in ndf_dca.groupby('year'):
    if len(grp)<10:continue
    yr_ret=(grp['nav'].iloc[-1]/grp['nav'].iloc[0]-1)*100
    yr_mdd=((grp['nav']-grp['nav'].cummax())/grp['nav'].cummax()).min()*100
    yr_in=grp['cost'].iloc[-1]-grp['cost'].iloc[0]
    print("    %d: %+7.1f%%  MaxDD%+6.1f%%  注资%.0f万" % (yr, yr_ret, yr_mdd, yr_in/10000))
