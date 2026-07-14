# -*- coding: utf-8 -*-
"""运行beta5策略, 推送交易信号"""
import subprocess, sys, os, json, urllib.request, argparse

BARK = 'https://api.day.app/eoq8G58fJtDDFxHjhNueGH'
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--reset', action='store_true', help='清空持仓')
args = parser.parse_args()
if args.reset:
    for f in ['YH01/positions.json']:
        try: os.remove(f); print(f'cleared: {f}')
        except: pass
    sys.exit(0)

out_file = 'd:\\策略\\_beta5_out.txt'
subprocess.run([sys.executable, 'YH01/monitor.py', '--output', out_file])
with open(out_file, 'r', encoding='utf-8') as f: out = f.read()

signals = []
for line in out.split('\n'):
    s = line.strip()
    if any(kw in s for kw in ['BUY', 'SELL', '买入', '卖出', '>>>', '建议']):
        signals.append(s)

if signals:
    body = '\n'.join(signals[:10])
    data = json.dumps({'title': 'beta5 Signal', 'body': body}).encode('utf-8')
    urllib.request.urlopen(urllib.request.Request(BARK, data=data,
        headers={'Content-Type': 'application/json'}), timeout=10)
    print(f'Pushed {len(signals)} signals')
else:
    print('No signals')
