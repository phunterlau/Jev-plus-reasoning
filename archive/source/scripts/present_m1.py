"""Readable presentation of frozen results, executed on lp; no inference."""
import csv,hashlib,json,os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(os.environ['JEV_ROOT'])/'results/m1-report-v1'
out=root/'presentation';out.mkdir(exist_ok=False)
r=json.loads((root/'report.json').read_text());s=r['summary']
plt.rcParams.update({'font.size':10,'figure.dpi':160})
fig,ax=plt.subplots(figsize=(8,4.5))
rows=[]
for c,offset in [('A-ref',(12,40)),('A',(12,-30)),('B',(12,-13)),('C32',(5,8)),('C128',(5,-18)),('C512',(-37,10))]:
 m=s[f'pilot/{c}/all'];x=m['latency_p50_seconds']*1000;y=m['accuracy']
 ax.scatter(x,y,label=c);ax.annotate(c,(x,y),xytext=offset,textcoords='offset points',arrowprops={'arrowstyle':'-','color':'gray','lw':.6})
 rows.append({'figure':'latency','condition':c,'depth':'all','n':m['n'],'accuracy':y,'median_ms':x})
ax.set(xlabel='Median warm request latency (ms, log scale)',ylabel='Accuracy',xscale='log',ylim=(.2,.65),title='Same 400 questions: accuracy and measured latency');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(out/'accuracy_latency.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.5))
for c in ['A','C32','C128','C512','D']:
 ds=[0,1,2,3,5];ys=[s[f'pilot/{c}/depth_{d}']['accuracy'] for d in ds]
 ax.plot(ds,ys,marker='s' if c=='D' else 'o',linestyle='--' if c=='D' else '-',markerfacecolor='none' if c=='D' else None,label=c)
 for d,y in zip(ds,ys):rows.append({'figure':'depth','condition':c,'depth':d,'n':s[f'pilot/{c}/depth_{d}']['n'],'accuracy':y,'median_ms':''})
ax.set(xlabel='Minimum proof depth (provable questions only)',ylabel='Accuracy',ylim=(0,1),xticks=ds,title='Qwen3-0.6B: direct A and oracle D coincide');ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(out/'accuracy_by_depth.png');plt.close(fig)
with (out/'figure_data.csv').open('x') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'provenance.json').write_text(json.dumps({'report_sha256':hashlib.sha256((root/'report.json').read_bytes()).hexdigest(),'job':os.environ['JEV_JOB'],'purpose':'Presentation only; original figures and measurements preserved.'},indent=2))
