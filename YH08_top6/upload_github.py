# -*- coding: utf-8 -*-
"""把 YH08_top6 策略代码上传到 GitHub 仓库 sunran1996/my_candle 的 YH08_top6/ 目录.
用法: python upload_github.py
"""
import os, sys, json, ssl, base64, io
import urllib.request as ur

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = 'sunran1996/my_candle'
DEST = 'YH08_top6'
FILES = ['daily_signal.py', 'compare.py', 'tune_3.py', 'upload_github.py']


def load_token():
    for p in [os.environ.get('GH_TOKEN', ''), '../github_token.txt', 'github_token.txt', 'd:/策略/github_token.txt']:
        if not p:
            continue
        try:
            t = open(p).read().strip()
            if t:
                return t
        except Exception:
            continue
    return ''


def github_put(token, path, content_b64, msg):
    ctx = ssl._create_unverified_context()
    h = {'Authorization': 'Bearer ' + token, 'User-Agent': 'YH08_top6'}
    api = f'https://api.github.com/repos/{REPO}/contents/{path}'
    try:
        r = json.loads(ur.urlopen(ur.Request(api, headers=h), timeout=15, context=ctx).read())
        sha = r.get('sha')
    except Exception:
        sha = None
    body = json.dumps({'message': msg, 'content': content_b64, 'branch': 'main',
                       **({'sha': sha} if sha else {})}).encode()
    ur.urlopen(ur.Request(api, data=body, headers={**h, 'Content-Type': 'application/json'},
                          method='PUT'), timeout=20, context=ctx)
    return sha is not None


def main():
    token = load_token()
    if not token:
        print('未找到 GitHub token (GH_TOKEN 或 github_token.txt)')
        return
    for name in FILES:
        p = os.path.join(HERE, name)
        if not os.path.exists(p):
            print(f'  跳过(不存在): {name}')
            continue
        with open(p, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        existed = github_put(token, f'{DEST}/{name}', b64, f'{DEST}: {name}')
        print(f"  {'更新' if existed else '新建'} {DEST}/{name}")
    print('完成')


if __name__ == '__main__':
    main()
