# -*- coding: utf-8 -*-
"""YH01 + YH02 -> Bark"""
import subprocess, sys, os, json, urllib.request, argparse

BARK = 'https://api.day.app/eoq8G58fJtDDFxHjhNueGH'

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--reset', action='store_true', help='仅清空持仓')
args = parser.parse_args()

if args.reset:
    for f in ['YH01/positions.json', 'YH02/positions.json', 'YH-beta/positions.json']:
        try: os.remove(f); print(f'已清空: {f}')
        except: print(f'无需清空: {f}')
    print('重置完成.'); sys.exit(0)

def send_bark(title, body):
    data = json.dumps({'title': title, 'body': body}).encode('utf-8')
    urllib.request.urlopen(urllib.request.Request(BARK, data=data,
        headers={'Content-Type': 'application/json'}), timeout=10)

for label, script in [('YH01', 'YH01/monitor.py'), ('YH02', 'YH02/monitor.py'), ('YH-beta', 'YH-beta/monitor.py')]:
    out_file = f'd:\\策略\\_{label}_out.txt'
    subprocess.run([sys.executable, script, '--output', out_file])
    with open(out_file, 'r', encoding='utf-8') as f:
        out = f.read()

    # 推Bark（先于终端打印，避免emoji崩溃）
    lines = out.split('\n'); clean = []
    for line in lines:
        s = line.strip()
        if s and all(c in '=─-*#~═' for c in s):
            if clean and clean[-1] != '': clean.append(''); continue
        if s == '':
            if clean and clean[-1] != '': clean.append(''); continue
        if any(kw in s for kw in ['BUY', 'SELL']): s = '**' + s + '**'
        clean.append(s)
    body = '\n'.join(clean)
    if len(body) > 3800: body = body[:3800] + '\n\n...'
    send_bark(label, body)
    print(f'{label} -> Bark ({len(body)} chars)')

    # 终端打印
    try:
        print(out)
    except:
        print(f'[终端编码限制, 见 {out_file} 或 VS Code 运行]')

print('Done.')
