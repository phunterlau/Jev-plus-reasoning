import hashlib,json,os,urllib.request,zipfile
from pathlib import Path
root=Path(os.environ['JEV_ROOT'])/'data'/'raw';root.mkdir(parents=True,exist_ok=True)
url='https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip'
p=root/'proofwriter-dataset-V2020.12.3.zip'
if not p.exists():
    tmp=p.with_suffix('.partial')
    print('Downloading',url,flush=True)
    urllib.request.urlretrieve(url,tmp)
    tmp.replace(p)
h=hashlib.sha256(p.read_bytes()).hexdigest()
with zipfile.ZipFile(p) as z:
    names=z.namelist()
    picked=[n for n in names if '/OWA/' in n and 'depth-5/' in n and n.endswith('.jsonl')]
    if not picked: picked=[n for n in names if 'OWA' in n][:40]
    print(json.dumps({'bytes':p.stat().st_size,'sha256':h,'files':picked},indent=2),flush=True)
    for n in picked:
        if n.endswith('.jsonl') and 'meta-' in n:
            with z.open(n) as f: sample=json.loads(f.readline())
            print('SAMPLE',n,json.dumps(sample)[:28000],flush=True)
            break
(root/'source_manifest.json').write_text(json.dumps({'url':url,'sha256':h,'bytes':p.stat().st_size},indent=2))
