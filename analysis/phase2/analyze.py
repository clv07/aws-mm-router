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
import loadgen as lg
from benchmark import median


def number(row, key):
    """Return a numeric measurement, or None when unavailable."""
    return float(row[key]) if row.get(key) not in (None, '') else None


def read(path):
    """Read optional CSV input without inventing missing measurements."""
    return lg.read_csv(path) if path.exists() else []


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
    return {prefix + '_p50': median(xs), prefix + '_p95': lg.percentile(xs, .95),
            prefix + '_p99': lg.percentile(xs, .99)}


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
    met = [r for r in done if r.get('met')]
    # Failures stay in the attainment denominator; slowdown uses fastest solo baseline.
    result = {'requests': len(cohort), 'success': len(ok),
              'attainment': sum(bool(r.get('met')) for r in deadlines)/len(deadlines) if deadlines else None,
              'starvation': sum(r['latency'] > 2*r['deadline'] for r in ok if r.get('deadline') is not None),
              'throughput': len(done)/(hi-lo), 'offered': len(cohort)/(hi-lo),
              'goodput': len(met)/(hi-lo) if deadlines else None,
              'usd_per_1k': price*(hi-lo)/3600*1000/len(done) if price and done else None,
              'usd_per_1k_met': price*(hi-lo)/3600*1000/len(met) if price and met else None}
    for field in ('latency', 'slowdown', 'ttft'):
        result.update(stats(ok, field, field))
    # Exact least-squares slope of the continuous backlog step function.
    events = sorted({lo, hi, *[max(lo,min(hi,r[k])) for r in rows for k in ('arrival','finish')]})
    integral = 0
    mid = (lo+hi)/2
    for a,b in zip(events, events[1:]):
        n = sum(r['arrival'] <= a < r['finish'] for r in rows)
        integral += n * ((b-mid)**2-(a-mid)**2)/2
    result['backlog_slope'] = integral / ((hi-lo)**3/12)
    return result


def normalize(batch, traces, costs):
    """Join side files from original requests, then derive comparable timestamps."""
    extra = unique(read(batch/'requests_extra.csv'), ('run_id','id'))
    rows = [dict(r, source='aws') for r in read(batch/'requests.csv')] + read(batch/'sim_requests.csv')
    unique(rows, ('source','run_id','policy','id'))
    cache = {}
    for r in rows:
        if r['trace'] not in cache:
            paths = {p.resolve() for root in traces for p in root.rglob(r['trace']+'.csv')}
            if len(paths) != 1:
                raise ValueError(f'ambiguous/missing trace {r["trace"]}: {paths}')
            path = paths.pop()
            meta = json.loads(path.with_suffix('.json').read_text()).get('params', {}) if path.with_suffix('.json').exists() else {}
            cache[r['trace']] = (unique(read(path), ('id',)), meta)
        trace, meta = cache[r['trace']]
        q = trace[(r['id'],)]
        r.setdefault('size', q.get('size',''))
        if r['source'] == 'aws':
            r.update(extra.get((r['run_id'],r['id']), {}))
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
            r['sent'] = r['busy'] if r['policy'] in lg.QUEUE_POLICIES else float(r['arrival_s'])
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
    nodes = lg.load_nodes(args.nodes)
    costs,_ = lg.load_costs(nodes)
    tables = defaultdict(list)
    for batch in args.batches:
        rows = normalize(batch,args.traces,costs)
        decisions = read(batch/'sim_decisions.csv')
        for key, rs in groups(rows, ('source','run_id','policy')).items():
            first = rs[0]
            labels = {k:first.get(k,'') for k in ('batch','source','run_id','trace','policy','seed','trace_seed','cluster','output_mode','load','pattern','mix','deadline_rule','slo_mult')}
            # Resolved slots distinguish simulations with different concurrency limits.
            configuration = {n:dict(c) for n,c in nodes.items()}
            for n, members in groups(rs, ('node',)).items():
                slots = {number(r,'node_slots') for r in members if r.get('node_slots')}
                configuration[n[0]]['measured_slots'] = next(iter(slots)) if len(slots)==1 else None
            labels['configuration'] = json.dumps(configuration,sort_keys=True)
            labels['latest'] = first['run_id'] == max(r['run_id'] for r in rows if tuple(r.get(k,'') for k in ('source','trace','policy','seed','cluster','output_mode')) == tuple(first.get(k,'') for k in ('source','trace','policy','seed','cluster','output_mode')))
            end = float(first['duration']) if first['duration'] and len(rs)==first['trace_count'] else max(r['arrival'] for r in rs)
            lo,hi = args.trim_start,end-args.trim_end
            price = sum(c.get('price_per_hour',0) for c in nodes.values()) if all(c.get('price_per_hour') for c in nodes.values()) else None
            head = {**labels, **metrics(rs,lo,hi,price), 'window_start':lo,'window_end':hi}
            ds = [{'duration':float(d['duration_us'])} for d in decisions if (d['run_id'],d['policy'])==key[1:]] if key[0]=='sim' else [{'duration':float(r['decision_us'])} for r in rs if r.get('decision_us')]
            head.update(stats(ds,'duration','decision_us'))
            head['decision_us_max'] = max((d['duration'] for d in ds),default=None)
            tables['headline'].append(head)
            for table, keys in [('per_modality',('modality',)),('bursts',('burst',)),('placement',('modality','size','node')),('nodes',('node',)),('prediction',('node','modality')),('breakdown',('modality',)),('residuals',('node',))]:
                for group, members in groups(rs,keys).items():
                    out = {**labels, **dict(zip(keys,group)), **metrics(members,lo,hi,price)}
                    ok = [r for r in members if r['status']=='ok']
                    out['full_success'] = len(ok)
                    if table=='placement':
                        out['share'] = len(ok)/sum(r['status']=='ok' and r['modality']==group[0] and r.get('size','')==group[1] for r in rs) if ok else 0
                    if table=='nodes':
                        slots = {number(r,'node_slots') for r in members if r.get('node_slots')}
                        out['busy_fraction'],out['slot_utilization'] = intervals(members,lo,hi,next(iter(slots)) if len(slots)==1 else None)
                        out['first_failure_s'] = min((r['finish'] for r in members if r['status']=='error'),default=None)
                        out['failure_rate'] = (len(members)-len(ok))/len(members)
                        out['errors'] = json.dumps(dict((g[0],len(v)) for g,v in groups([r for r in members if r['status']!='ok'],('error',)).items()))
                        out['output_tokens_s'] = sum(number(r,'output_tokens') or 0 for r in members if r['status']=='ok' and lo<=r['finish']<hi)/(hi-lo) if hi>lo and key[0]=='aws' else None
                        samples = [r for r in read(batch/'node_samples.csv') if r['run_id']==key[1] and r['node']==group[0] and lo<=float(r['t_s'])<hi]
                        out['cpu_pct'] = median([float(r['cpu_pct']) for r in samples if r['cpu_pct']])
                        out['peak_ram_gb'] = max((float(r['ram_gb']) for r in samples if r['ram_gb']),default=None)
                    if table=='prediction':
                        out.update(stats(ok,'prediction_error','signed_error_s'))
                        measured=[r for r in ok if r.get('prediction_error') is not None]
                        out['prediction_n']=len(measured)
                        out['median_abs_rel']=median([abs(r['prediction_error'])/float(r['e2e_s']) for r in measured if float(r['e2e_s'])>0])
                    if table=='breakdown':
                        for field in ('router_wait_s','node_wait_net_s','node_wait_s','prefill_s','decode_s','service_s'):
                            xs=[r[field] for r in ok if r.get(field) is not None]; out[field]=sum(xs)/len(xs) if xs else None
                    if table=='residuals':
                        residual=[float(r['queue_s']) for r in ok if r.get('node_inflight')=='1' and r.get('queue_s')]; out.update(n=len(residual),median_queue_s=median(residual))
                    out['req_bytes'] = sum(float(r['req_bytes']) for r in members if r.get('req_bytes')) if any(r.get('req_bytes') for r in members) else None
                    tables[table].append(out)
            # Historical full-run values are retained verbatim, not relabeled as arrival latency.
            for s in read(batch/('summary.csv' if key[0]=='aws' else 'sim_summary.csv')):
                if s['run_id']==key[1] and s['policy']==key[2]: tables['full_run'].append(dict(s,batch=str(batch),source=key[0]))
    args.out.mkdir(parents=True,exist_ok=True)
    for name, rows in tables.items(): write(args.out/(name+'.csv'),rows)
    # Trace seed and run seed remain explicit; repeated draws on one trace are not independent traces.
    spread=[]
    keys=('source','configuration','output_mode','load','pattern','mix','deadline_rule','slo_mult','policy','trace')
    for key,rows in groups([r for r in tables['headline'] if r['latest']],keys).items():
        out=dict(zip(keys,key)); out.update(runs=len(rows),trace_seeds=len({r['trace_seed'] for r in rows if r['trace_seed']!=''}),requests=sum(r.get('requests',0) for r in rows))
        for metric in ('attainment','goodput','slowdown_p95','latency_p95','usd_per_1k_met','decision_us_p99'):
            xs=[r[metric] for r in rows if r.get(metric) is not None]
            for suffix,value in [('mean',sum(xs)/len(xs) if xs else None),('min',min(xs) if xs else None),('max',max(xs) if xs else None)]: out[metric+'_'+suffix]=value
        spread.append(out)
    write(args.out/'seed_spread.csv',spread)


def main():
    """Parse analysis-only options without changing runner commands or defaults."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches',type=Path,nargs='+',default=[Path('results/routing')])
    p.add_argument('--traces',type=Path,nargs='+',default=[Path('traces')])
    p.add_argument('--nodes',default='results/routing/nodes.json')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--trim-start',type=float,default=60)
    p.add_argument('--trim-end',type=float,default=60)
    analyze(p.parse_args())


if __name__=='__main__': main()
