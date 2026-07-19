# -*- coding: utf-8 -*-
"""YH02 多源数据层: akshare → baostock → 本地缓存"""
import os, pandas as pd, numpy as np
from datetime import datetime, timedelta

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _get_cache_path(symbol):
    return os.path.join(CACHE_DIR, f'{symbol}.csv')


def _save_cache(df, symbol):
    _ensure_cache_dir()
    df.to_csv(_get_cache_path(symbol), index=False)


def _load_cache(symbol, max_age_days=1):
    path = _get_cache_path(symbol)
    if not os.path.exists(path): return None
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    if (datetime.now() - mtime).days > max_age_days: return None
    df = pd.read_csv(path); df['date'] = pd.to_datetime(df['date'])
    return df


def fetch_akshare(symbol):
    """新浪源 (akshare)"""
    import akshare as ak
    df = ak.fund_etf_hist_sina(symbol=symbol)
    df['date'] = pd.to_datetime(df['date'])
    return df[['date', 'close']].sort_values('date').reset_index(drop=True)


def fetch_baostock(symbol):
    """baostock 源 (免费, 无需token)"""
    import baostock as bs
    bs.login()
    # sh512890 → bs code: sh.512890
    code = f'{symbol[:2]}.{symbol[2:]}'
    end = datetime.now().strftime('%Y-%m-%d')
    start = '2015-01-01'
    rs = bs.query_history_k_data_plus(code, 'date,close', start_date=start, end_date=end, frequency='d')
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    if not rows:
        raise Exception('baostock returned empty')
    df = pd.DataFrame(rows, columns=['date', 'close']).astype({'close': float})
    df['date'] = pd.to_datetime(df['date'])
    return df[['date', 'close']].sort_values('date').reset_index(drop=True)


def get_data(symbol='sh512890', force_refresh=False):
    """
    多源数据获取:
    1. 本地缓存(1天内) → 2. akshare → 3. baostock → 4. 缓存副本
    """
    # 1. 缓存
    if not force_refresh:
        cached = _load_cache(symbol)
        if cached is not None and len(cached) > 100:
            return cached

    # 2. akshare
    try:
        df = fetch_akshare(symbol)
        if len(df) > 100:
            _save_cache(df, symbol)
            return df
    except Exception as e:
        print(f"  akshare 失败: {e}, 尝试 baostock...")

    # 3. baostock
    try:
        df = fetch_baostock(symbol)
        if len(df) > 100:
            _save_cache(df, symbol)
            return df
    except Exception as e:
        print(f"  baostock 失败: {e}, 用本地缓存...")

    # 4. 旧缓存(不限天数)
    old = _load_cache(symbol)
    if old is not None and len(old) > 100:
        print(f"  使用旧缓存 ({len(old)}条)")
        return old

    raise RuntimeError(f'无法获取 {symbol} 数据, 所有数据源均失败')


if __name__ == '__main__':
    print("测试数据源...")
    for method, sym in [('akshare', 'sh512890'), ('baostock', 'sh512890')]:
        try:
            fn = fetch_akshare if method == 'akshare' else fetch_baostock
            df = fn(sym)
            print(f"  {method}: {len(df)}条 {df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"  {method}: 失败 - {e}")
