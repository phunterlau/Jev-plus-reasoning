"""Audit official OWA labels/depths on lp, then freeze disjoint cohorts."""
from collections import Counter,defaultdict
import hashlib,json,os,zipfile
from pathlib import Path
from jev_reasoning.proofwriter import *

ROOT=Path(os.environ['JEV_ROOT']); OUT=ROOT/'data/derived/owa-v2'
if OUT.exists(): raise RuntimeError('Create-only dataset output exists')
OUT.mkdir(parents=True)
archive=ROOT/'data/raw/proofwriter-dataset-V2020.12.3.zip'
source=json.loads((archive.parent/'source_manifest.json').read_text())
assert source['sha256']=='bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26'

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def rank(x):return hashlib.sha256(('17:'+x).encode()).hexdigest()
def band(d):return 'shallow' if d in (0,1) else str(d)

counts=Counter(); errors=[]; groups=defaultdict(set); buffers=defaultdict(list)
with zipfile.ZipFile(archive) as z:
 names=sorted(n for n in z.namelist() if re.search(r'/OWA/depth-[01235]/meta-(train|dev|test)\.jsonl$',n))
 for path in names:
  split=path.rsplit('-',1)[-1].split('.')[0]; theory_depth=int(path.split('/depth-')[1].split('/')[0])
  with z.open(path) as f:
   for line_number,line in enumerate(f):
    raw=json.loads(line);counts['theories']+=1
    try:
     parsed=parse_theory(raw);facts,rules,grounded=parsed
     ds,parents=closure(facts,grounded)
     if any(complement(a) in ds for a in ds): raise ValueError('Contradictory closure')
     group=canonical_hash(facts,rules);groups[group].add(split)
     counts['tilde_premises']+=sum(v['representation'].count('"~"') for v in raw['rules'].values())
     # Independent comparison with all supplied implication depths, not just queries.
     for detail in raw.get('proofDetails',[]):
      a=atom(detail['representation'])
      if ds.get(a)!=int(detail['QDep']): raise ValueError(f'Implication depth mismatch: {a} {ds.get(a)} {detail["QDep"]}')
      counts['implications_checked']+=1
     if set(ds)!={atom(x['representation']) for x in raw.get('proofDetails',[])}:raise ValueError('Closure differs from all supplied implications')
     for qid,q in raw['questions'].items():
      counts['questions']+=1
      a=atom(q['representation']);gold=source_label(q['answer']);actual=label(a,ds)
      if actual!=gold: raise ValueError(f'Label mismatch {qid}: {actual} != {gold}')
      depth=None if gold=='Unknown' else ds[a if gold=='True' else complement(a)]
      if depth is not None and depth!=int(q['QDep']):raise ValueError(f'Question depth mismatch {qid}')
      cohort=band(theory_depth if depth is None else depth)
      if cohort not in ['shallow','2','3','5']:continue
      subdepth=depth if cohort=='shallow' and depth is not None else None
      uid=f'{raw["id"]}/{qid}'
      meta={'id':uid,'theory_id':raw['id'],'group':group,'source_split':split,'source_file':path,'line_number':line_number,'question_id':qid,'gold':gold,'depth':depth,'raw_qdep':int(q['QDep']),'theory_depth':theory_depth,'cohort':cohort,'family':raw['id'].split('-')[0],'rank':rank(uid)}
      key=(split,cohort,gold,subdepth,meta['family'])
      buf=buffers[key];buf.append(meta)
      if len(buf)>160:buf.sort(key=lambda x:x['rank']);del buf[80:]
    except Exception as e:
     counts['errors']+=1
     if len(errors)<25:errors.append({'path':path,'id':raw['id'],'error':str(e)})
  print(path,dict(counts),flush=True)

summary={'source':source,'counts':dict(counts),'errors':errors,'group_count':len(groups),'cross_split_duplicate_groups':sum(len(v)>1 for v in groups.values()),'polarity_contract':'OWA premise ~ treated as explicit -, verified against official full closure, labels and depths','seed':17}
(OUT/'audit.json').write_text(json.dumps(summary,indent=2))
if errors:raise RuntimeError('Source audit failed; see audit.json. No cohorts emitted.')

# One row per canonical theory; cross-split duplicate theories excluded from all cohorts.
selected=[];used=set()
quotas={'test':{'shallow':(34,33,33),'2':(33,34,33),'3':(33,33,34),'5':(34,33,33)},'dev':{b:(10,10,10) for b in ['shallow','2','3','5']},'train':{b:(2,2,2) for b in ['shallow','2','3','5']}}
for split in ['test','dev','train']:
 for cohort,ns in quotas[split].items():
  for gold,n in zip(['True','False','Unknown'],ns):
   subs=[(0,(n+1)//2),(1,n//2)] if cohort=='shallow' and gold!='Unknown' else [(None,n)]
   for sd,need in subs:
    pools={family:sorted(buffers.get((split,cohort,gold,sd,family),[]),key=lambda x:x['rank']) for family in ['AttNeg','AttNoneg','RelNeg','RelNoneg']}
    got=0
    while got<need:
     progress=False
     for family,pool in pools.items():
      while pool:
       item=pool.pop(0)
       if item['group'] not in used and len(groups[item['group']])==1:break
      else:continue
      selected.append(item);used.add(item['group']);got+=1;progress=True
      if got==need:break
     if not progress:raise RuntimeError(f'Insufficient supply {split} {cohort} {gold} {sd}: {got}/{need}')

index=defaultdict(dict)
for item in selected:index[item['source_file']].setdefault(item['line_number'],[]).append(item)
rows=defaultdict(list)
with zipfile.ZipFile(archive) as z:
 for path,wanted in index.items():
  with z.open(path) as f:
   for line_number,line in enumerate(f):
    if line_number not in wanted:continue
    raw=json.loads(line);parsed=parse_theory(raw);ds,_=closure(parsed[0],parsed[2])
    for item in wanted[line_number]:
     q=raw['questions'][item['question_id']]
     oracle,proof=oracle_intermediates(raw,q,ds,parsed)
     row={k:v for k,v in item.items() if k!='rank'}
     row.update(state=raw['theory'],question=q['question'],query=atom(q['representation']),oracle=oracle,oracle_proof=proof,source_archive_sha256=source['sha256'],raw_theory_sha256=digest(raw),state_sha256=hashlib.sha256(raw['theory'].encode()).hexdigest())
     rows[item['source_split']].append(row)

manifest={'protocol':'owa-v2','seed':17,'source':source,'cohorts':{},'readme':'Required proof depth excludes Unknown. One row per canonical theory; all cross-split duplicates excluded.'}
for split,data in rows.items():
 name={'test':'pilot','dev':'validation','train':'smoke'}[split]
 data.sort(key=lambda x:x['id'])
 dest=OUT/f'{name}.jsonl'
 dest.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in data))
 manifest['cohorts'][name]={'rows':len(data),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'ids':[x['id'] for x in data],'labels':dict(Counter(x['gold'] for x in data)),'depths':dict(Counter(str(x['depth']) for x in data)),'families':dict(Counter(x['family'] for x in data)),'oracle_nonempty':sum(bool(x['oracle']) for x in data),'state_chars':{'min':min(len(x['state']) for x in data),'max':max(len(x['state']) for x in data)}}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:{a:b for a,b in v.items() if a!='ids'} for k,v in manifest['cohorts'].items()},indent=2),flush=True)
