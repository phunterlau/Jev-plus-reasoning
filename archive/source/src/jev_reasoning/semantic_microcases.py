"""Fresh R0 microcases; labels derived and checked with the existing OWA engine."""
import itertools,json,random,re
from .proofwriter import closure,label,complement,canonical_hash
from .prompts import sha
LABELS=['True','False','Unknown']

def make_cases():
 names=[a+' '+b for a in ['Ari','Bea','Cora','Dara','Eli','Faye','Gia','Hana','Ira','Jana','Kai','Lea'] for b in ['Alder','Birch','Cedar','Elm','Maple','Willow']]
 random.Random(17).shuffle(names)
 pairs=[('blue','round'),('green','quiet'),('red','kind'),('young','tall'),('rough','small'),('white','cold')]
 rows=[]
 def txt(a):return f'{a[0]} is '+('not ' if a[3]=='-' else '')+a[2]+'.'
 for i,(stage,gold,sign,v) in enumerate(itertools.product([0,1],LABELS,['+','-'],range(6))):
  name=names[i];target,premise=pairs[v];q=(name,'is',target,sign)
  head=q if gold=='True' else complement(q) if gold=='False' else (q if v%2 else complement(q))
  facts={'f0':(name,'is','awake','+')};rules={}
  if stage==0:
   facts['f1']=head if gold!='Unknown' else ('Morgan Oak','is',target,sign)
  else:
   facts['f1']=(name if gold!='Unknown' else 'Morgan Oak','is',premise,'-' if v%2 else '+')
   p=('someone','is',premise,'-' if v%2 else '+');h=('someone','is',target,head[3]);rules={'r1':((p,),h)}
  grounded=[(rid,tuple((name,*p[1:]) for p in ps),(name,*h[1:])) for rid,(ps,h) in rules.items()]
  depths,_=closure(facts,grounded)
  assert label(q,depths)==gold
  d=None if gold=='Unknown' else depths[q if gold=='True' else complement(q)]
  assert d is None or d==stage
  texts=[txt(a) for a in facts.values()]
  random.Random(100+i).shuffle(texts)
  rule_text=[]
  for ps,h in rules.values():
   p=ps[0];rule_text.append('If someone is '+('not ' if p[3]=='-' else '')+p[2]+', then they are '+('not ' if h[3]=='-' else '')+h[2]+'.')
  state='Facts:\n'+'\n'.join(texts)+'\nRules:\n'+ ('\n'.join(rule_text) or 'None.')
  rows.append({'id':f'r0-{i:03d}','group':canonical_hash(facts,rules),'state':state,'state_sha256':sha(state),'question':txt(q),'query':q,'gold':gold,'depth':d,'nominal_depth':stage,'query_polarity':sign,'variant':v,'family':'synthetic_r0','cohort':'fresh_microcases','oracle':[],'facts':facts,'rules':rules})
 assert len(rows)==72 and len({r['group'] for r in rows})==72
 return rows

def parse_semantic(text):
 m=re.fullmatch(r'\s*(True|False|Unknown)[.!]?\s*',text)
 return m.group(1) if m else None
