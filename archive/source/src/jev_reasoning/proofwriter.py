"""Official ProofWriter OWA parsing, monotone closure, and proof validation."""
from collections import defaultdict
import hashlib
import itertools
import json
import re

ATOM = re.compile(r'\(\s*"([^"\n]+)"\s+"([^"\n]+)"\s+"([^"\n]+)"\s+"([+~\-])"\s*\)')
VARIABLES = {'someone', 'something'}

def atom(text, premise=False):
    match=ATOM.fullmatch(text.strip())
    if not match: raise ValueError(f'Unsupported atom: {text}')
    a=match.groups()
    if a[3]=='~':
        if not premise: raise ValueError('Tilde outside premise')
        a=(*a[:3], '-')
    return a

def complement(a): return (*a[:3], '-' if a[3]=='+' else '+')

def rule(text):
    if text.count('->')!=1: raise ValueError('Expected one rule arrow')
    left,right=text.split('->')
    pre=[atom(m.group(),True) for m in ATOM.finditer(left)]
    post=[atom(m.group()) for m in ATOM.finditer(right)]
    if not pre or len(post)!=1: raise ValueError('Malformed rule')
    if re.sub(r'[\s()]','',ATOM.sub('',left)+ATOM.sub('',right)):
        raise ValueError('Unparsed rule text')
    return tuple(pre),post[0]

def parse_theory(raw):
    facts={k:atom(v['representation']) for k,v in raw['triples'].items()}
    rules={k:rule(v['representation']) for k,v in raw['rules'].items()}
    entities=set()
    for a in list(facts.values())+[a for ps,c in rules.values() for a in (*ps,c)]:
        entities.add(a[0])
        if a[1]!='is': entities.add(a[2])
    entities-=VARIABLES
    grounded=[]
    for rid,(premises,conclusion) in rules.items():
        terms={v for a in (*premises,conclusion) for v in (a[0],a[2]) if v in VARIABLES}
        for values in itertools.product(sorted(entities),repeat=len(terms)):
            subst=dict(zip(sorted(terms),values))
            def ground(a): return (subst.get(a[0],a[0]),a[1],subst.get(a[2],a[2]),a[3])
            grounded.append((rid,tuple(map(ground,premises)),ground(conclusion)))
    return facts,rules,grounded

def closure(facts,grounded):
    depths={a:0 for a in facts.values()}
    parents={a:None for a in facts.values()}
    changed=True
    while changed:
        changed=False
        for rid,pre,con in grounded:
            if all(p in depths for p in pre):
                d=1+max(depths[p] for p in pre)
                if con not in depths or d<depths[con]:
                    depths[con]=d; parents[con]=(rid,pre);changed=True
    return depths,parents

def canonical_hash(facts,rules):
    canonical={'facts':sorted(set(facts.values())), 'rules':sorted((tuple(sorted(p)),c) for p,c in rules.values())}
    return hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()

def label(query,depths):
    yes=query in depths; no=complement(query) in depths
    if yes and no: raise ValueError('Both query polarities derivable')
    return 'True' if yes else 'False' if no else 'Unknown'

def source_label(value):
    if value is True: return 'True'
    if value is False: return 'False'
    if value=='Unknown': return value
    raise ValueError(f'Unexpected answer {value!r}')

def sexpr(text):
    tokens=re.findall(r'\(|\)|[^\s()]+',text)
    stack=[]; root=[]; current=root
    for token in tokens:
        if token=='(':
            child=[];current.append(child);stack.append(current);current=child
        elif token==')':
            if not stack: raise ValueError('Unbalanced proof')
            current=stack.pop()
        else: current.append(token)
    if stack or len(root)!=1: raise ValueError('Malformed proof')
    return root[0]

def oracle_intermediates(raw,q,depths,parsed=None):
    query=atom(q['representation']); answer=source_label(q['answer'])
    if answer=='Unknown': return [],None
    target=query if answer=='True' else complement(query)
    facts,_,grounded=parsed or parse_theory(raw)
    candidates=[]
    for proof in q.get('proofsWithIntermediates',[]):
        ints=proof['intermediates'] or {}
        nodes={k:atom(v['representation']) for k,v in ints.items()}
        visited=[]
        def evaluate(node):
            while isinstance(node,list) and len(node)==1: node=node[0]
            if isinstance(node,str):
                if node not in facts: raise ValueError(f'Unknown proof leaf {node}')
                return facts[node],0
            if len(node)!=3 or node[1]!='->': raise ValueError(f'Unsupported proof node {node}')
            premises=[evaluate(x) for x in node[0]]
            rid,mark,iid=node[2]
            if mark!='%': raise ValueError('Missing proof intermediate marker')
            target_atom=nodes[iid]
            actual=tuple(p for p,d in premises)
            if not any(r==rid and c==target_atom and sorted(ps)==sorted(actual) for r,ps,c in grounded):
                raise ValueError(f'Invalid gold proof application {rid}')
            d=1+max(v for _,v in premises)
            if target_atom not in depths: raise ValueError('Gold intermediate not entailed')
            visited.append((iid,target_atom,d))
            return target_atom,d
        root,d=evaluate(sexpr(proof['representation']))
        if root!=target: raise ValueError('Proof has wrong root')
        candidates.append((d,len(proof['representation']),proof['representation'],visited,ints))
    if not candidates: raise ValueError('Provable question without proof')
    chosen=min(candidates,key=lambda x:x[:3])
    if chosen[0]!=depths[target]: raise ValueError('Minimum gold proof depth differs from closure')
    out=[]; seen=set(facts.values())|{query,complement(query)}
    for iid,a,d in chosen[3]:
        if a not in seen:
            out.append({'text':chosen[4][iid]['text'],'atom':a,'depth':d})
            seen.add(a)
    return out,chosen[2]
