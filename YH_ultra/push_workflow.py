"""Push workflow file to GitHub"""
import json,ssl,base64,os,urllib.request as ur

REPO='sunran1996/my_candle'

token=os.environ.get('GH_TOKEN','')
if not token:
    for p in ['../github_token.txt','github_token.txt','d:/策略/github_token.txt']:
        try: token=open(p).read().strip();break
        except: pass

print(f"Token: {'OK' if token else 'MISSING'}")

with open('workflow.yml','rb') as f: raw=f.read()
content_b64=base64.b64encode(raw).decode('ascii')

path='.github/workflows/yh_ultra.yml'
ctx=ssl._create_unverified_context()
h={'Authorization':'Bearer '+token,'User-Agent':'YH_ultra'}
api=f'https://api.github.com/repos/{REPO}/contents/{path}'

sha=None
try:
    r=json.loads(ur.urlopen(ur.Request(api,headers=h),timeout=10,context=ctx).read())
    sha=r.get('sha'); print(f"File exists, sha={sha[:8]}...")
except: print("File is new, will create")

body=json.dumps({'message':'Add YH_ultra workflow with --live','content':content_b64,'branch':'main',
    **({'sha':sha} if sha else {})}).encode()
ur.urlopen(ur.Request(api,data=body,headers={**h,'Content-Type':'application/json'},method='PUT'),
    timeout=15,context=ctx)
print(f"Pushed to {path}")
