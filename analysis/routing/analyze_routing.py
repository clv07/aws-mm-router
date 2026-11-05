"""Routing benchmark analysis: latency per policy, where each policy sent each request type,
throughput and cost, and how accurate ect's predictions were.

    python analysis/routing/analyze_routing.py

Reads results/routing/requests.csv and summary.csv, prints the tables in README.md and writes
routing_latency, routing_placement and routing_throughput_cost (.pdf and .png) next to this script.
Uses the latest run of each (trace, policy), so a repeated run replaces an earlier one.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / 'results' / 'routing'
REQUESTS, SUMMARY = RESULTS / 'requests.csv', RESULTS / 'summary.csv'

POLICIES = ['rr', 'least', 'ect']
TYPES = ['chat', 'text', 'image', 'audio']
LOADS = {'cluster-load60': '60% load', 'cluster-load90': '90% load'}
NODES = ['c7a.2xlarge', 'c7i.xlarge', 'm8g.xlarge']

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'legend.fontsize': 6.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.linewidth': 0.6, 'xtick.color': '#52514e',
                     'pdf.fonttype': 42})
POLICY_COLORS = {'rr': '#2a78d6', 'least': '#eb6834', 'ect': '#1baf7a'}
NODE_COLORS = {'c7a.2xlarge': '#6250d6', 'c7i.xlarge': '#eda100', 'm8g.xlarge': '#008300'}
INK2, GRID = '#52514e', '#e6e5e1'


def load():
    d = pd.read_csv(REQUESTS, dtype={'run': str})
    latest = d.groupby(['trace', 'policy']).run_id.max().reset_index()   # run_id is a UTC timestamp
    d = d.merge(latest, on=['trace', 'policy', 'run_id'])
    failed = d[d.status != 'ok']
    if len(failed):
        print(f'{len(failed)} failed requests (left out):')
        print(failed.groupby(['trace', 'policy', 'node']).size().to_string(), '\n')
    return d[d.status == 'ok'].copy()


def latency_table(d):
    """Median and p95 end-to-end latency per load, policy and request type ('all' = every request)."""
    groups = pd.concat([d, d.assign(modality='all')])
    t = groups.groupby(['trace', 'policy', 'modality']).e2e_s.agg(
        n='size', median='median', p95=lambda s: s.quantile(.95)).reset_index()
    return t


def placement_table(d):
    """Share of each request type sent to each node, per load and policy."""
    t = d.groupby(['trace', 'policy', 'modality', 'node']).size().rename('n').reset_index()
    t['share'] = t.n / t.groupby(['trace', 'policy', 'modality']).n.transform('sum')
    return t


def prediction_table(d):
    """How far the latency ect's model predicted for the chosen node was from the measured one.
    Logged for every policy, so it also shows how predictable each policy's placements are."""
    d = d.assign(err=d.e2e_s - d.predicted_e2e_s)
    d['abs_rel'] = d.err.abs() / d.e2e_s
    return d.groupby(['trace', 'policy']).agg(n=('err', 'size'), median_err_s=('err', 'median'),
                                              median_abs_rel=('abs_rel', 'median')).reset_index()


def throughput_cost_table(d):
    """Offered vs served requests/s and cost per 1k requests, per load and policy, from summary.csv."""
    s = pd.read_csv(SUMMARY)
    s = s[(s.group == 'all') & s.run_id.isin(set(d.run_id))]
    return s[['trace', 'policy', 'offered_req_s', 'req_s', 'usd_per_1k_requests', 'p99_e2e_s']]


def fig_throughput_cost(t, path):
    """Served throughput (bars) against offered load (line), and cost per 1k requests, per policy."""
    loads = [l for l in LOADS if l in set(t.trace)]
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 1.5), gridspec_kw={'wspace': 0.35})
    w = 0.8 / len(POLICIES)
    x = np.arange(len(loads))
    for k, p in enumerate(POLICIES):
        s = t[t.policy == p].set_index('trace').reindex(loads)
        xs = x + (k - (len(POLICIES) - 1) / 2) * w
        for ax, col, fmt in [(axes[0], 'req_s', '{:.3f}'), (axes[1], 'usd_per_1k_requests', '${:.2f}')]:
            ax.bar(xs, s[col], w * 0.92, color=POLICY_COLORS[p], label=p)
            for xi, v in zip(xs, s[col]):
                ax.text(xi, v * 1.01, fmt.format(v), ha='center', va='bottom', fontsize=5, zorder=4,
                        bbox=dict(facecolor='white', edgecolor='none', pad=0.3))
    offered = t.groupby('trace').offered_req_s.first().reindex(loads)
    for xi, v in zip(x, offered):
        axes[0].hlines(v, xi - 0.45, xi + 0.45, color='#0b0b0b', linewidth=0.8, linestyle='--', zorder=2)
    axes[0].set_title('Served req/s (dashed: offered)', loc='left')
    axes[1].set_title('Cost per 1,000 requests', loc='left')
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([LOADS[l] for l in loads])
        ax.tick_params(axis='x', length=0)
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    for ax in axes:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    axes[1].legend(frameon=False, loc='lower right', bbox_to_anchor=(1, 1.08), ncol=len(POLICIES), handlelength=1)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(path.with_suffix('.png'), dpi=200, bbox_inches='tight')


def fig_latency(t, path):
    """Fig. 3: p95 latency per request type, policies side by side, one panel per load (stacked)."""
    loads = [l for l in LOADS if l in set(t.trace)]
    fig, axes = plt.subplots(len(loads), 1, figsize=(3.33, 1.0 * len(loads)), squeeze=False, sharex=True,
                             gridspec_kw={'hspace': 0.35})
    groups = TYPES + ['all']
    x = np.arange(len(groups))
    w = 0.8 / len(POLICIES)
    ymax = t.p95.max() * 1.18
    for ax, load in zip(axes[:, 0], loads):
        for k, p in enumerate(POLICIES):
            s = t[(t.trace == load) & (t.policy == p)].set_index('modality').reindex(groups)
            if s.p95.isna().all():
                continue
            xs = x + (k - (len(POLICIES) - 1) / 2) * w
            ax.bar(xs, s.p95.fillna(0), w * 0.92, color=POLICY_COLORS[p], label=p)
            ax.scatter(xs, s['median'], s=5, color='white', edgecolor='#0b0b0b', linewidth=0.5, zorder=3)
            for xi, v in zip(xs, s.p95):
                if not np.isnan(v):
                    ax.text(xi, v + ymax * 0.01, f'{v:.0f}', ha='center', va='bottom', fontsize=5)
        ax.set_title(LOADS[load], loc='left', pad=2)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.tick_params(axis='x', length=0)
        ax.set_ylim(0, ymax)
        ax.set_yticks([0, 100, 200])
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    fig.supylabel('p95 latency, s (dot: median)', color=INK2, fontsize=6.5, x=0.0)
    axes[0][0].legend(frameon=False, loc='lower right', bbox_to_anchor=(1, 0.98), ncol=len(POLICIES),
                      handlelength=1, columnspacing=1)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(path.with_suffix('.png'), dpi=200, bbox_inches='tight')


def fig_placement(t, path, load='cluster-load90'):
    """Fig. 4: for one load, the share of each request type each policy sent to each node."""
    t = t[t.trace == load]
    if t.empty:
        return
    fig, ax = plt.subplots(figsize=(3.33, 0.95))
    labels, x, pos = [], [], 0.0
    for m in TYPES:
        for p in POLICIES:
            s = t[(t.modality == m) & (t.policy == p)].set_index('node').reindex(NODES).share.fillna(0)
            bottom = 0.0
            for n in NODES:
                ax.bar(pos, s[n], 0.8, bottom=bottom, color=NODE_COLORS[n], edgecolor='white',
                       linewidth=0.8, label=n if (m, p) == (TYPES[0], POLICIES[0]) else None)
                bottom += s[n]
            labels.append(p)
            x.append(pos)
            pos += 1
        ax.text(pos - 2, -0.44, m, ha='center', fontsize=7, transform=ax.get_xaxis_transform())
        pos += 0.6
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels(['0', '50%', '100%'])
    ax.legend(frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.52), handlelength=1)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(path.with_suffix('.png'), dpi=200, bbox_inches='tight')


def main():
    d = load()
    pd.set_option('display.width', 200)
    lat, place, pred = latency_table(d), placement_table(d), prediction_table(d)

    print('requests per run:')
    print(d.groupby(['trace', 'policy']).size().unstack().reindex(columns=POLICIES).to_string(), '\n')
    print('latency, s (median / p95):')
    lat['cell'] = lat['median'].map('{:.0f}'.format) + ' / ' + lat.p95.map('{:.0f}'.format)
    print(lat.pivot_table(index=['trace', 'modality'], columns='policy', values='cell', aggfunc='first')
          .reindex(columns=POLICIES).to_string(), '\n')
    print('share of each request type per node:')
    print(place.pivot_table(index=['trace', 'policy', 'modality'], columns='node', values='share')
          .reindex(columns=NODES).fillna(0).round(2).to_string(), '\n')
    print('ect model prediction error (measured - predicted):')
    print(pred.round(2).to_string(index=False))

    tc = throughput_cost_table(d)
    print('\nthroughput and cost (all requests):')
    print(tc.round(3).to_string(index=False))

    fig_latency(lat, HERE / 'routing_latency.pdf')
    fig_placement(place, HERE / 'routing_placement.pdf')
    fig_throughput_cost(tc, HERE / 'routing_throughput_cost.pdf')
    print(f'\nwrote routing_latency, routing_placement and routing_throughput_cost (.pdf, .png) in {HERE}')


if __name__ == '__main__':
    main()
