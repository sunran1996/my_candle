# -*- coding: utf-8 -*-
"""YH08 单标的 K线 + 买卖点 放大图
用法:
    python plot_trades.py --name 中国神华 --start 2021-01-01
    python plot_trades.py --all --start 2021-01-01        # 全部标的一次生成
    python plot_trades.py --name 长江电力 --start 2016-01-01
默认: 中国神华, 2021-01-01 起
"""
import sys, io, os, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import daily_signal as d

NAME2KEY = {'长江电力': 'changjiang', '招商银行': 'zhaoshang', '国电电力': 'guodian',
            '中国神华': 'shenhua', '创业板': 'chuangye'}


def _draw_one(name, start, end, raw, dfs, td):
    ohlc = raw[name][raw[name]['date'] >= pd.Timestamp(start)].copy()
    if end:
        ohlc = ohlc[ohlc['date'] <= pd.Timestamp(end)]
    ohlc = ohlc.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                                'close': 'Close', 'volume': 'Volume'})
    ohlc = ohlc.set_index('date')[['Open', 'High', 'Low', 'Close', 'Volume']]

    cn_c = mpf.make_marketcolors(up='#CC0000', down='#008800', edge='inherit',
                                 wick='inherit', volume='inherit')
    cn_s = mpf.make_mpf_style(marketcolors=cn_c, gridstyle='',
                              rc={'font.sans-serif': [d.CN], 'axes.unicode_minus': False})

    fig, ax = plt.subplots(figsize=(24, 9), facecolor='white')
    mpf.plot(ohlc, type='candle', ax=ax, volume=False, style=cn_s)

    bb = dfs[name][dfs[name]['date'] >= pd.Timestamp(start)]
    if end:
        bb = bb[bb['date'] <= pd.Timestamp(end)]
    x = range(len(ohlc))
    ax.plot(x, bb['bb_up'].values[-len(ohlc):], color='#9B59B6', lw=0.8, ls='--', alpha=0.6)
    ax.plot(x, bb['bb_lo'].values[-len(ohlc):], color='#9B59B6', lw=0.8, ls='--', alpha=0.6)
    ax.plot(x, bb['bb_ma'].values[-len(ohlc):], color='#888888', lw=0.7, ls='--', alpha=0.5)
    ax.plot(x, bb['ma60'].values[-len(ohlc):], color='#3498DB', lw=1.0, alpha=0.6)

    ohlc_dates = ohlc.index
    n_buy = n_sell = 0
    for _, t in td.iterrows():
        if t['name'] != name:
            continue
        td_d = pd.Timestamp(t['date'])
        for j, od in enumerate(ohlc_dates):
            if pd.Timestamp(od).date() == td_d.date():
                if t['dir'] == 'BUY':
                    ax.scatter(j, ohlc['Low'].iloc[j], color='#FF0000', s=170, marker='^',
                               zorder=10, edgecolors='white', lw=2.0)
                    ax.annotate(f"买 {t['price']:.2f}", (j, ohlc['Low'].iloc[j]),
                                textcoords='offset points', xytext=(0, -30),
                                fontsize=8, color='#CC0000', fontweight='bold', ha='center')
                    n_buy += 1
                else:
                    ax.scatter(j, ohlc['High'].iloc[j], color='#008800', s=170, marker='v',
                               zorder=10, edgecolors='white', lw=2.0)
                    ax.annotate(f"卖 {t['pnl']:+.1f}%", (j, ohlc['High'].iloc[j]),
                                textcoords='offset points', xytext=(0, 20),
                                fontsize=8, color='#008800', fontweight='bold', ha='center')
                    n_sell += 1
                break

    sells = td[(td['name'] == name) & (td['dir'] == 'SELL')]
    wins = (sells['pnl'] > 0).sum()
    wr = wins / len(sells) * 100 if len(sells) else 0
    ax.set_title(f"{name}  买卖点 ({start} 起, 不加DCA)  |  买{n_buy} 卖{n_sell}  胜率{wr:.0f}%",
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=8)

    out = os.path.join(d.SCRIPT, f'trades_{NAME2KEY.get(name, name)}.png')
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'图表: {out}')


def _load(start, end):
    print('获取数据...')
    raw = d.fetch()
    dfs = {n: d.add_indicators(x) for n, x in raw.items()}
    dates = sorted(set.intersection(*[set(x['date']) for x in dfs.values()]))
    dates = [x for x in dates if x >= pd.Timestamp(start)]
    if end:
        dates = [x for x in dates if x <= pd.Timestamp(end)]
    ndf, td, _, _ = d._simulate(raw, dfs, dates, False)  # 不加DCA
    return raw, dfs, td


def plot(name, start, end):
    raw, dfs, td = _load(start, end)
    _draw_one(name, start, end, raw, dfs, td)


def plot_all(start, end):
    raw, dfs, td = _load(start, end)
    for name in d.ALL_STOCKS:
        _draw_one(name, start, end, raw, dfs, td)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='中国神华')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--start', default='2021-01-01')
    ap.add_argument('--end', default=None)
    a = ap.parse_args()
    if a.all:
        plot_all(a.start, a.end)
    else:
        plot(a.name, a.start, a.end)
