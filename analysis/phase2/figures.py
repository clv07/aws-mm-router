"""Read generated metric tables and write attainment, breakdown and placement PNGs."""
import argparse
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import read, groups


MODALITY_ORDER = ('all', 'chat', 'text', 'image', 'audio')


def max_rate(points, target):
    """Highest rate with attainment >= target there and at every lower rate, interpolated between the
    last point above and the first below; (rate, True) if every measured rate is above."""
    best = None
    for i, (x, y) in enumerate(points):
        if y >= target:
            best = x
            continue
        if i > 0:
            x0, y0 = points[i - 1]
            best = x0 + (y0 - target) / (y0 - y) * (x - x0)
        return best, False
    return best, True


def grid(root, metric, target):
    """SLO attainment vs offered rate in the layout of the EPD papers: a column per modality (plus all),
    a line per policy, the target as a dashed line and each policy's highest sustained rate as a
    dashed vertical. x = the run's offered req/s in the steady-state window; seeds are averaged."""
    heads = [r for r in read(root/'headline.csv') if r['latest'] == 'True' and r.get(metric)]
    rate = {(r['batch'], r['source'], r['run_id'], r['policy']): float(r['offered']) for r in heads if r.get('offered')}
    rows = [dict(r, modality='all') for r in heads]
    rows += [r for r in read(root/'per_modality.csv') if r['latest'] == 'True' and r.get(metric)]
    comparison = ('source', 'cluster', 'configuration', 'output_mode', 'mix', 'pattern', 'deadline_rule', 'slo_mult')
    written = []
    for i, (key, members) in enumerate(groups(rows, comparison).items()):
        cells = defaultdict(list)   # (modality, policy, load label) -> [(rate, attainment)]
        for r in members:
            x = rate.get((r['batch'], r['source'], r['run_id'], r['policy']))
            if x is not None:
                cells[(r['modality'], r['policy'], r['load'] or r['trace'])].append((x, float(r[metric])))   # seeds of one load
        curves = defaultdict(list)
        for (modality, policy, _), pts in cells.items():
            curves[(modality, policy)].append((sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)))
        modalities = [m for m in MODALITY_ORDER if any(c[0] == m for c in curves)]
        policies = sorted({c[1] for c in curves})
        if not modalities:
            continue
        fig, axes = plt.subplots(1, len(modalities), sharey=True, squeeze=False, figsize=(2.8 * len(modalities), 2.8))
        for ax, modality in zip(axes[0], modalities):
            for j, policy in enumerate(policies):
                pts = sorted(curves.get((modality, policy), []))
                if not pts:
                    continue
                color = f'C{j}'
                ax.plot([x for x, _ in pts], [100 * y for _, y in pts], marker='os^v>D'[j % 6], color=color,
                        markersize=4, label=policy)
                best, above = max_rate(pts, target)
                if best is not None and not above:
                    ax.axvline(best, ymax=target / 1.05, color=color, linestyle='--', linewidth=1)
            ax.axhline(100 * target, color='grey', linestyle='--', linewidth=1)
            ax.set(ylim=(0, 105), title=modality, xlabel='offered rate (req/s)')
        axes[0][0].set_ylabel(('TTFT+TPOT' if metric == 'attainment_ttft_tpot' else 'end-to-end') + ' SLO attainment (%)')
        axes[0][-1].legend(fontsize=7)
        fig.suptitle(' / '.join(k for k in (key[0], key[1], key[5], key[4]) if k), fontsize=9)
        fig.tight_layout()
        path = root/f'slo-grid-{metric}-{i}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def main():
    """Plot each comparison trace/configuration separately to avoid mixing batches."""
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('directory',type=Path)
    p.add_argument('--target', type=float, default=0.9, help='attainment goal for the SLO grids (default 0.9)')
    args=p.parse_args()
    root=args.directory
    for metric in ('attainment', 'attainment_ttft_tpot'):
        grid(root, metric, args.target)
    spread = [r for r in read(root/'seed_spread.csv') if r['load'] and r['attainment_mean']]
    comparison = ('source','cluster','configuration','output_mode','mix','deadline_rule','slo_mult')
    for i, (key, rows) in enumerate(groups(spread, comparison).items()):
        patterns = groups(rows, ('pattern',))
        fig, axes = plt.subplots(1, len(patterns), sharey=True, squeeze=False,
                                 figsize=(5 * len(patterns), 4))
        for ax, ((pattern,), members) in zip(axes[0], patterns.items()):
            for (policy,), series in groups(members, ('policy',)).items():
                points = sorted((float(r['load']), float(r['attainment_mean'])) for r in series)
                ax.plot(*zip(*points), marker='o', label=policy)
            ax.set(xlabel='offered load', ylabel='SLO attainment', title=pattern)
            ax.legend()
        fig.suptitle(' / '.join(k for k in (key[0],key[1],key[4]) if k))
        fig.tight_layout()
        fig.savefig(root/f'attainment-{i}.png')
        plt.close(fig)
    for i,(key,rows) in enumerate(groups([r for r in read(root/'breakdown.csv') if r.get('latest')=='True'],('batch','source','run_id','policy')).items()):
        fig,ax=plt.subplots(); bottom=[0.0]*len(rows)
        for field in ('router_wait_s','node_wait_net_s','net_s','node_queue_s','node_wait_s','encode_s','prefill_s','decode_s','service_s'):
            if not any(r.get(field) for r in rows): continue
            xs=[float(r.get(field) or 0) for r in rows]
            ax.bar([r['modality'] for r in rows],xs,bottom=bottom,label=field); bottom=[a+b for a,b in zip(bottom,xs)]
        ax.set(ylabel='mean seconds',title=' / '.join(key[1:])); ax.legend(fontsize=6)
        fig.savefig(root/f'breakdown-{i}.png'); plt.close(fig)
    for i,(key,rows) in enumerate(groups([r for r in read(root/'placement.csv') if r.get('latest')=='True' and r['size']=='all'],('batch','source','run_id','policy')).items()):
        modalities=sorted({r['modality'] for r in rows}); nodes=sorted({r['node'] for r in rows})
        values=[]
        for m in modalities:
            total=sum(int(r['full_success']) for r in rows if r['modality']==m)
            values.append([sum(int(r['full_success']) for r in rows if r['modality']==m and r['node']==n)/total if total else 0 for n in nodes])
        fig,ax=plt.subplots(); im=ax.imshow(values,vmin=0,vmax=1); fig.colorbar(im,ax=ax)
        ax.set_xticks(range(len(nodes)),nodes,rotation=20); ax.set_yticks(range(len(modalities)),modalities); ax.set_title(' / '.join(key[1:]))
        fig.savefig(root/f'placement-{i}.png'); plt.close(fig)


if __name__=='__main__': main()
