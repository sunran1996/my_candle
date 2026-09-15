# -*- coding: utf-8 -*-
"""一次性推送 YH-1.2 代码/持仓状态/提醒状态到 GitHub"""
import os
import daily_signal as d

token = ''
for p in ['d:/策略/github_token.txt', '../github_token.txt', 'github_token.txt']:
    try:
        token = open(p, encoding='utf-8').read().strip()
        if token:
            break
    except Exception:
        pass
if not token:
    print('未找到 github token, 跳过推送')
else:
    d.push_code(token)
    d.push_state(token)
    d.push_alerts(token)
    print('推送完成')
