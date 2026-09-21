"""Explicit probability metrics with invalid-output coverage."""
import math
from .prompts import LABELS

def metrics(rows):
 n=len(rows)
 if not n:return {'n':0}
 acc=sum(r['correct'] for r in rows)/n
 f1=[]
 for label in LABELS:
  tp=sum(r['gold']==label and r['prediction']==label and r['readout_valid'] for r in rows)
  fp=sum(r['gold']!=label and r['prediction']==label and r['readout_valid'] for r in rows)
  fn=sum(r['gold']==label and (r['prediction']!=label or not r['readout_valid']) for r in rows)
  f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
 nll=0.;brier=0.;bins=[[] for _ in range(10)]
 for r in rows:
  ps=r['probabilities']; labels=r['option_ids']; gold=labels.index(r['gold'])
  nll-=math.log(max(ps[gold],1e-12))
  brier+=sum((p-(i==gold))**2 for i,p in enumerate(ps))
  conf=max(ps);idx=max(range(len(ps)),key=lambda i:ps[i])
  bins[min(9,int(conf*10))].append((conf,labels[idx]==r['gold']))
 ece=sum(len(b)/n*abs(sum(p for p,_ in b)/len(b)-sum(c for _,c in b)/len(b)) for b in bins if b)
 times=sorted(r['total_seconds'] for r in rows)
 def quantile(q):
  i=(n-1)*q;lo=int(i);hi=min(lo+1,n-1);return times[lo]+(times[hi]-times[lo])*(i-lo)
 return {'n':n,'accuracy':acc,'valid_fraction':sum(r['readout_valid'] for r in rows)/n,'macro_f1':sum(f1)/3,'nll':nll/n,'brier_sum':brier/n,'ece_10_equal_bins':ece,'probability_metric_scope':'all restricted slot distributions, including invalid public readouts','latency_p50_seconds':quantile(.5),'latency_p95_seconds':quantile(.95),'examples_per_second':n/sum(times),'mean_reasoning_tokens':sum(r['sampled_reasoning_tokens'] for r in rows)/n,'natural_close_fraction':sum(r['natural_close'] for r in rows)/n,'forced_close_fraction':sum(r['forced_close'] for r in rows)/n,'max_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in rows)}
