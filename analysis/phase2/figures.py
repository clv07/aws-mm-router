"""Read generated metric tables and write attainment, breakdown and placement PNGs."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import read, groups


def main():
    """Plot each comparison trace/configuration separately to avoid mixing batches."""
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('directory',type=Path); args=p.parse_args()
    root=args.directory
    for i,(key,rows) in enumerate(groups([r for r in read(root/'headline.csv') if r.get('latest')=='True'],('source','configuration','output_mode','pattern','mix','deadline_rule')).items()):
        fig,ax=plt.subplots()
        for policy,rs in groups(rows,('policy',)).items():
            points=sorted((float(r['load']),float(r['attainment'])) for r in rs if r['load'] and r['attainment'])
            if points: ax.plot(*zip(*points),marker='o',label=policy[0])
        ax.set(xlabel='offered load',ylabel='SLO attainment',title=' / '.join(str(k) for k in (key[0],key[3],key[4])))
        if ax.lines: ax.legend()
        fig.savefig(root/f'attainment-{i}.png'); plt.close(fig)
    for i,(key,rows) in enumerate(groups([r for r in read(root/'breakdown.csv') if r.get('latest')=='True'],('batch','source','run_id','policy')).items()):
        fig,ax=plt.subplots(); bottom=[0.0]*len(rows)
        for field in ('router_wait_s','node_wait_net_s','node_wait_s','prefill_s','decode_s','service_s'):
            if not any(r.get(field) for r in rows): continue
            xs=[float(r.get(field) or 0) for r in rows]
            ax.bar([r['modality'] for r in rows],xs,bottom=bottom,label=field); bottom=[a+b for a,b in zip(bottom,xs)]
        ax.set(ylabel='mean seconds',title=' / '.join(key[1:])); ax.legend(fontsize=6)
        fig.savefig(root/f'breakdown-{i}.png'); plt.close(fig)
    for i,(key,rows) in enumerate(groups([r for r in read(root/'placement.csv') if r.get('latest')=='True'],('batch','source','run_id','policy')).items()):
        modalities=sorted({r['modality'] for r in rows}); nodes=sorted({r['node'] for r in rows})
        values=[]
        for m in modalities:
            total=sum(int(r['full_success']) for r in rows if r['modality']==m)
            values.append([sum(int(r['full_success']) for r in rows if r['modality']==m and r['node']==n)/total if total else 0 for n in nodes])
        fig,ax=plt.subplots(); im=ax.imshow(values,vmin=0,vmax=1); fig.colorbar(im,ax=ax)
        ax.set_xticks(range(len(nodes)),nodes,rotation=20); ax.set_yticks(range(len(modalities)),modalities); ax.set_title(' / '.join(key[1:]))
        fig.savefig(root/f'placement-{i}.png'); plt.close(fig)


if __name__=='__main__': main()
