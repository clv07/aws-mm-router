"""Read AWS/sim batches, traces and node profiles; write per-run metric CSVs and figures.

Missing measurements remain empty. Batch paths are part of every key; comparisons
retain configuration, output mode, trace/run seeds and deadline rule.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import main as runner
import router
from benchmark import median


HISTORICAL_RUNS = {
    '20261105T001442Z', '20261105T003211Z', '20261105T004841Z', '20261105T010506Z',
    '20261105T012551Z', '20261105T020543Z', '20261105T022419Z',
}


def number(row, key):
    """Return a numeric measurement, or None when unavailable."""
    return float(row[key]) if row.get(key) not in (None, '') else None


def read(path):
    """Read optional CSV input without inventing missing measurements."""
    return router.read_csv(path) if path.exists() else []


def unique(rows, keys):
    """Index rows and reject ambiguous joins before any metrics are computed."""
    out = {}
    for r in rows:
        key = tuple(r[k] for k in keys)
        if key in out:
            raise ValueError(f'duplicate {keys}: {key}')
        out[key] = r
    return out


def groups(rows, keys):
    """Partition rows by explicit comparison keys."""
    out = defaultdict(list)
    for r in rows:
        out[tuple(r.get(k, '') for k in keys)].append(r)
    return out


def stats(rows, field, prefix):
    """Use runner quantiles and median so historical summaries match exactly."""
    xs = [r[field] for r in rows if r.get(field) is not None]
    return {prefix + '_p50': median(xs), prefix + '_p95': runner.percentile(xs, .95),
            prefix + '_p99': runner.percentile(xs, .99)}


def intervals(rows, lo, hi, slots=None):
    """Integrate clipped busy intervals, including work present at window start."""
    if hi <= lo or not any(r.get('busy') is not None for r in rows):
        return None, None
    events = defaultdict(int)
    for r in rows:
        a, b = r.get('busy'), r.get('finish')
        if a is not None and b is not None and min(b, hi) > max(a, lo):
            events[max(a, lo)] += 1
            events[min(b, hi)] -= 1
    last, active, busy, area = lo, 0, 0, 0
    for t, delta in sorted(events.items()):
        busy += (t-last) * (active > 0)
        area += (t-last) * active
        last, active = t, active + delta
    return busy / (hi-lo), area / (hi-lo) / slots if slots else None


def metrics(rows, lo, hi, price):
    """Compute arrival-cohort latency and completion-window rates on [lo, hi)."""
    if hi <= lo:
        return {}
    cohort = [r for r in rows if lo <= r['arrival'] < hi]
    ok = [r for r in cohort if r['status'] == 'ok']
    done = [r for r in rows if r['status'] == 'ok' and lo <= r['finish'] < hi]
    deadlines = [r for r in cohort if r.get('deadline') is not None]
    has_deadlines = any(r.get('deadline') is not None for r in rows)
    met = [r for r in done if r.get('met')]
    # Failures stay in the attainment denominator; slowdown uses fastest solo baseline.
    result = {'requests': len(cohort), 'success': len(ok),
              'attainment': sum(bool(r.get('met')) for r in deadlines)/len(deadlines) if deadlines else None,
              'starvation': sum(r['latency'] > 2*r['deadline'] for r in ok if r.get('deadline') is not None) if has_deadlines else None,
              'completions': len(done), 'deadline_requests': len(deadlines), 'met_completions': len(met) if has_deadlines else None,
              'throughput': len(done)/(hi-lo), 'offered': len(cohort)/(hi-lo),
              'goodput': len(met)/(hi-lo) if has_deadlines else None,
              'usd_per_1k': price*(hi-lo)/3600*1000/len(done) if price and done else None,
              'usd_per_1k_met': price*(hi-lo)/3600*1000/len(met) if price and met else None}
    for field in ('latency', 'slowdown', 'ttft'):
        result.update(stats(ok, field, field))
    # Exact least-squares slope of the continuous backlog step function.
    events = defaultdict(int)
    events[lo] = events[hi] = 0
    for r in rows:
        start, stop = max(lo, r['arrival']), min(hi, r['finish'])
        if start < stop:
            events[start] += 1
            events[stop] -= 1
    integral, n, last = 0, 0, lo
    mid = (lo+hi)/2
    for t, change in sorted(events.items()):
        integral += n * ((t-mid)**2-(last-mid)**2)/2
        n += change
        last = t
    result['backlog_slope'] = integral / ((hi-lo)**3/12)
    return result


def normalize(batch, traces, costs):
    """Join side files from original requests, then derive comparable timestamps."""
    extra = unique(read(batch/'requests_extra.csv'), ('run_id','id'))
    rows = [dict(r, source='aws') for r in read(batch/'requests.csv')] + read(batch/'sim_requests.csv')
    unique([r for r in rows if r['source'] == 'aws'], ('run_id', 'id'))
    unique(rows, ('source','run_id','policy','id'))
    cache = {}
    for r in rows:
        if r['trace'] not in cache:
            paths = {p.resolve() for root in traces for p in root.rglob(r['trace']+'.csv')}
            if len(paths) != 1:
                raise ValueError(f'ambiguous/missing trace {r["trace"]}: {paths}')
            path = paths.pop()
            companion = path.with_suffix('.json')
            metadata = json.loads(companion.read_text()) if companion.exists() else {}
            meta = dict(metadata.get('params', {}), segments=metadata.get('segments'))
            cache[r['trace']] = (unique(read(path), ('id',)), meta)
        trace, meta = cache[r['trace']]
        q = trace[(r['id'],)]
        r['segments'] = meta['segments']
        r.setdefault('size', q.get('size',''))
        if r['source'] == 'aws':
            r.update(extra.get((r['run_id'],r['id']), {}))
            # docs/routing-benchmark.md section 2 records slots for the historical cluster.
            if (not r.get('node_slots') and r.get('cluster') == 'c7a-c7i-m8g'
                    and r['trace'] in ('cluster-load60', 'cluster-load90')
                    and r['run_id'] in HISTORICAL_RUNS):
                r['node_slots'] = {'c7a.2xlarge': 2, 'c7i.xlarge': 1, 'm8g.xlarge': 2}.get(r['node'])
            r['finish'] = float(r['sent_s']) + float(r['e2e_s'])
            r['sent'] = float(r['sent_s'])
            r['busy'] = r['sent'] + float(r['queue_s']) if r.get('queue_s') and r.get('server_prompt_ms') else None
            # AWS residual includes network; sim node wait is a separate measurement.
            r['node_wait_net_s'] = number(r,'queue_s')
            r['ttft'] = r['sent'] + float(r['ttft_s']) - float(r['arrival_s']) if r.get('ttft_s') else None
            r['prefill_s'] = number(r,'server_prompt_ms')
            r['decode_s'] = number(r,'server_predicted_ms')
            for key in ('prefill_s','decode_s'):
                if r[key] is not None: r[key] /= 1000
            r['prediction_error'] = float(r['e2e_s'])-float(r['predicted_e2e_s']) if r.get('predicted_e2e_s') else None
        else:
            r['status'] = 'ok'
            r['finish'], r['busy'] = float(r['finish_s']), float(r['start_s'])
            r['sent'] = r['busy'] if r['policy'] in router.QUEUE_POLICIES else float(r['arrival_s'])
            r['node_wait_s'] = r['busy']-r['sent']
            r['service_s'] = r['finish']-r['busy']
        r['arrival'] = float(r['arrival_s'])
        r['latency'] = r['finish']-r['arrival']
        r['router_wait_s'] = r['sent']-r['arrival']
        r['deadline'] = number(q,'deadline_s')
        factor = number(q,'slo_factor')
        baseline = r['deadline']/factor if r['deadline'] is not None and factor and factor > 0 else min(c[r['run']] for c in costs.values())
        r['slowdown'] = r['latency']/baseline
        r['met'] = r['status']=='ok' and r['latency'] <= r['deadline'] if r['deadline'] is not None else None
        r.update(batch=str(batch), burst=q.get('burst',''), trace_seed=meta.get('seed',''),
                 load=meta.get('load', {'cluster-load60':.6,'cluster-load90':.9}.get(r['trace'],'')),
                 pattern=meta.get('arrivals',''), mix=meta.get('mix',''), deadline_rule=meta.get('slo',''),
                 duration=meta.get('duration'), trace_count=len(trace))
    return rows


def write(path, rows):
    """Write a new CSV with union columns and explicit empty missing values."""
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ['batch']
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)


def analyze(args):
    """Produce per-run tables and seed spread without combining incompatible runs."""
    tables = defaultdict(list)
    for batch in args.batches:
        profile = args.nodes or (batch/'nodes.json' if (batch/'nodes.json').exists() else 'results/routing/nodes.json')
        nodes = router.load_nodes(profile)
        costs, _ = router.load_costs(nodes)
        rows = normalize(batch,args.traces,costs)
        decisions = {}
        for source, filename in [('sim', 'sim_decisions.csv'), ('aws', 'decisions.csv')]:
            calls = read(batch/filename)
            unique(calls, ('run_id','policy','n'))
            decisions[source] = groups(calls, ('run_id','policy'))
        samples_by_run = groups(read(batch/'node_samples.csv'), ('run_id',))
        for key, rs in groups(rows, ('source','run_id','policy')).items():
            first = rs[0]
            labels = {k:first.get(k,'') for k in ('batch','source','run_id','trace','policy','seed','trace_seed','cluster','output_mode','load','pattern','mix','deadline_rule','slo_mult')}
            # Resolved slots distinguish simulations with different concurrency limits.
            configuration = {n:dict(c, measured_slots=c.get('slots')) for n,c in nodes.items()}
            for n, members in groups(rs, ('node',)).items():
                slots = {number(r,'node_slots') for r in members if r.get('node_slots')}
                configuration[n[0]]['measured_slots'] = next(iter(slots)) if len(slots)==1 else None
            labels['configuration'] = json.dumps(configuration,sort_keys=True)
            # Only historical reruns replace earlier comparisons. Other seeds remain observations.
            labels['latest'] = (key[0] != 'aws' or key[1] not in HISTORICAL_RUNS or
                                key[1] == max(r['run_id'] for r in rows if r['source'] == 'aws'
                                              and r['run_id'] in HISTORICAL_RUNS
                                              and (r['trace'], r['policy']) == (first['trace'], key[2])))
            # Complete replays use the declared duration. Truncated runs stop at the last arrival.
            end = float(first['duration']) if first['duration'] and len(rs)==first['trace_count'] else max(r['arrival'] for r in rs)
            lo,hi = args.trim_start,end-args.trim_end
            price = sum(c.get('price_per_hour',0) for c in nodes.values()) if all(c.get('price_per_hour') for c in nodes.values()) else None
            head = {**labels, **metrics(rs,lo,hi,price), 'window_start':lo,'window_end':hi}
            if key[0] == 'sim' or key[2] in router.QUEUE_POLICIES:
                ds = [{'duration': float(d['duration_us'])} for d in decisions[key[0]][key[1:]]]
            else:
                ds = [{'duration': float(r['decision_us'])} for r in rs if r.get('decision_us')]
            head['decision_calls'] = len(ds)
            samples = samples_by_run[(key[1],)] if key[0] == 'aws' else []
            for sample in samples:
                t = float(sample['t_s'])
                burst = next((int(on) for start, stop, _, on in first['segments'] or []
                              if start <= t < stop), None)
                tables['sample_context'].append({**labels, **sample, 'burst': burst,
                    'backlog': sum(r['arrival'] <= t < r['finish'] for r in rs)})
            head.update(stats(ds,'duration','decision_us'))
            head['decision_us_max'] = max((d['duration'] for d in ds),default=None)
            tables['headline'].append(head)
            for table, keys in [('per_modality',('modality',)),('bursts',('burst',)),('placement',('modality','size','node')),('nodes',('node',)),('prediction',('node','modality')),('breakdown',('modality',)),('residuals',('node',))]:
                partitions = groups(rs,keys)
                if table == 'nodes':
                    # Include configured idle nodes so unused capacity remains visible.
                    for node in nodes:
                        partitions.setdefault((node,), [])
                if table == 'prediction':
                    partitions[('all','all')] = rs
                if table == 'placement':
                    for (modality,node), members in groups(rs,('modality','node')).items():
                        partitions[(modality,'all',node)] = members
                for group, members in partitions.items():
                    out = {**labels, **dict(zip(keys,group)), **metrics(members,lo,hi,price)}
                    ok = [r for r in members if r['status']=='ok']
                    out['full_success'] = len(ok)
                    if table == 'per_modality' and key[0] == 'aws':
                        # Legacy latency uses successful sends over the full run.
                        xs = [float(r['e2e_s']) for r in ok]
                        out.update(full_send_p50=median(xs), full_send_p95=runner.percentile(xs, .95))
                    if table=='placement':
                        out['share'] = len(ok)/sum(r['status']=='ok' and r['modality']==group[0] and (group[1]=='all' or r.get('size','')==group[1]) for r in rs) if ok else 0
                    if table=='nodes':
                        slots = {number(r,'node_slots') for r in members if r.get('node_slots')}
                        slot_count = next(iter(slots)) if len(slots)==1 else nodes[group[0]].get('slots')
                        out['busy_fraction'],out['slot_utilization'] = intervals(members,lo,hi,slot_count)
                        if not members and hi > lo:
                            out['busy_fraction'] = 0.0
                            out['slot_utilization'] = 0.0 if slot_count else None
                        out['first_failure_s'] = min((r['finish'] for r in members if r['status']=='error'),default=None)
                        out['failure_rate'] = (len(members)-len(ok))/len(members) if members else None
                        out['errors'] = json.dumps(dict((g[0],len(v)) for g,v in groups([r for r in members if r['status']!='ok'],('error',)).items()))
                        out['output_tokens_s'] = sum(number(r,'output_tokens') or 0 for r in members if r['status']=='ok' and lo<=r['finish']<hi)/(hi-lo) if hi>lo and key[0]=='aws' else None
                        observations = [r for r in samples if r['node']==group[0] and lo<=float(r['t_s'])<hi]
                        out['cpu_pct'] = median([float(r['cpu_pct']) for r in observations if r['cpu_pct']])
                        out['peak_ram_gb'] = max((float(r['ram_gb']) for r in observations if r['ram_gb']),default=None)
                    if table=='prediction':
                        out.update(stats(ok,'prediction_error','signed_error_s'))
                        measured=[r for r in ok if r.get('prediction_error') is not None]
                        out['prediction_n']=len(measured)
                        out['median_abs_rel']=median([abs(r['prediction_error'])/float(r['e2e_s']) for r in measured if float(r['e2e_s'])>0])
                    if table=='breakdown':
                        # Use the same successful arrival cohort as steady-state latency.
                        cohort = [r for r in ok if lo <= r['arrival'] < hi]
                        out['breakdown_requests'] = len(cohort) if hi > lo else None
                        for field in ('router_wait_s','node_wait_net_s','node_wait_s','prefill_s','decode_s','service_s'):
                            xs=[r[field] for r in cohort if r.get(field) is not None]; out[field]=sum(xs)/len(xs) if xs else None
                    if table=='residuals':
                        residual=[float(r['queue_s']) for r in ok if r.get('node_inflight')=='1' and r.get('queue_s')]; out.update(n=len(residual),median_queue_s=median(residual))
                    out['req_bytes'] = sum(float(r['req_bytes']) for r in members if r.get('req_bytes')) if any(r.get('req_bytes') for r in members) else None
                    tables[table].append(out)
            # Historical full-run values are retained verbatim, not relabeled as arrival latency.
            for s in read(batch/('summary.csv' if key[0]=='aws' else 'sim_summary.csv')):
                if s['run_id']==key[1] and s['policy']==key[2]: tables['full_run'].append(dict(s,batch=str(batch),source=key[0]))
    args.out.mkdir(parents=True,exist_ok=True)
    # Trace seed and run seed remain explicit; repeated draws on one trace are not independent traces.
    spread=[]
    keys=('source','cluster','configuration','output_mode','load','pattern','mix','deadline_rule','slo_mult','policy')
    for key,rows in groups([r for r in tables['headline'] if r['latest']],keys).items():
        out=dict(zip(keys,key)); out.update(runs=len(rows),trace_seeds=len({r['trace_seed'] for r in rows if r['trace_seed']!=''}),requests=sum(r.get('requests',0) for r in rows))
        for metric in ('attainment','goodput','slowdown_p95','latency_p95','usd_per_1k_met','decision_us_p99'):
            measured = [r for r in rows if r.get(metric) is not None]
            # Average repeats within a trace seed first, so one trace cannot outweigh another.
            by_seed = defaultdict(list)
            for r in measured:
                seed = r['trace_seed'] if r['trace_seed'] != '' else (r['trace'], r['seed'])
                by_seed[seed].append(r[metric])
            xs = [sum(values)/len(values) for values in by_seed.values()]
            count_field = {'attainment':'deadline_requests', 'goodput':'met_completions',
                           'slowdown_p95':'success', 'latency_p95':'success',
                           'usd_per_1k_met':'met_completions', 'decision_us_p99':'decision_calls'}[metric]
            out[metric+'_n'] = sum(r[count_field] for r in measured)
            for suffix,value in [('mean',sum(xs)/len(xs) if xs else None),('min',min(xs) if xs else None),('max',max(xs) if xs else None)]: out[metric+'_'+suffix]=value
        out['batches'] = json.dumps(sorted({r['batch'] for r in rows}))
        out['run_seeds'] = json.dumps(sorted({r['seed'] for r in rows}))
        spread.append(out)
        for row in rows:
            row.update({k:v for k,v in out.items() if k.endswith(('_mean','_min','_max'))})
    for name, rows in tables.items():
        write(args.out/(name+'.csv'), rows)
    write(args.out/'seed_spread.csv',spread)


def main():
    """Parse analysis-only options without changing runner commands or defaults."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches',type=Path,nargs='+',default=[Path('results/routing')])
    p.add_argument('--traces',type=Path,nargs='+',default=[Path('traces')])
    p.add_argument('--nodes', help='Override the node profile for all input batches')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--trim-start',type=float,default=60)
    p.add_argument('--trim-end',type=float,default=60)
    analyze(p.parse_args())


if __name__=='__main__': main()
