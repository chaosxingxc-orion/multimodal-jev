"""Collect real shape receipts and compare monotone bucket schedules at measured boundaries."""
import argparse
import itertools
import json
import math
import statistics
from pathlib import Path


def check(ok, message):
    if not ok:
        raise ValueError(message)


def schedules(bands, records, widths):
    rates = {(r['band'], r['width']): r['rate'] for r in records}
    n = sum(b['population'] for b in bands)
    candidates = []
    for indices in itertools.combinations_with_replacement(range(len(widths)), len(bands)):
        w = [widths[i] for i in indices]
        if not all((i, cap) in rates for i, cap in enumerate(w)):
            continue
        sec = sum(b['population'] / rates[i, cap] for i, (b, cap) in enumerate(zip(bands, w)))
        buckets = [dict(max_tokens=bands[i]['upper'] if i+1<len(w) else None, batch_size_per_rank=cap)
                   for i, cap in enumerate(w) if i+1==len(w) or cap != w[i+1]]
        candidates.append((sec, dict(buckets=buckets, anchors_per_second=n/sec)))
    return {k:min((t for t in candidates if not required or len(t[1]['buckets'])==len(widths)),
                  key=lambda t:t[0])[1] for k, required in [('all_widths', True), ('optional_widths', False)]}


def collect(root, spec):
    read = lambda p: json.loads(p.read_bytes())
    d = read(root/'distribution.json')
    hardware = {r['cid']: r for r in read(root/'inspection.json')}
    policy = set()
    expected = list(itertools.product(spec['recipes'], spec['widths']))
    check(len(hardware) == len(expected), 'inspection count')
    for i, (recipe, width) in enumerate(expected):
        cid = (root/f'r2-cid{i}').read_text().strip()
        hw = hardware[cid]
        code = int((root/f'r2-exit{i}').read_text())
        check(hw['state']=='exited' and hw['code']==code and hw['device']==f'/dev/davinci{i}', 'terminal or device mismatch')
        lane = read(root/f'lane{i}.json')
        actual = lane.pop('real_lane')
        check(lane==spec and (actual['recipe'],actual['width'])==(recipe,width), 'lane configuration mismatch')
        data = d['recipes'][recipe]
        bands = data['bands']
        hist = data['anchors']['histogram']
        n = d['real'] + d['synthetic']
        check(sum(hist.values())==n and data['pairs']['n']==n*(1+sum(spec['recipes'][recipe])), 'population')
        check(all(sum(v for k,v in hist.items() if b['lower']<=int(k)<=b['upper'])==b['population'] for b in bands)
              and sum(b['population'] for b in bands)==n, 'band population')
        log=(root/f'r2-log{i}').read_text()
        rows=[json.loads(s[5:]) for s in log.splitlines() if s.startswith('BAND ')]
        check(len(rows)<=len(bands) and (code!=0 or len(rows)==len(bands)), 'incomplete successful lane')
        for j,r in enumerate(rows):
            check((r['band'],r['recipe'],r['width'],r['tokens'])==(j,recipe,width,bands[j]['upper']), 'wrong band')
            check(len(r['times'])==spec['repeats'] and all(math.isfinite(t) and t>0 for t in r['times'])
                  and math.isclose(width/statistics.median(r['times']),r['rate']), 'timing mismatch')
            policy.add(r.pop('policy'))
        data.setdefault('measurements', []).extend(rows)
        hw['recipe'], hw['width'] = recipe, width
        hw['oom_in_log'] = 'out of memory' in log.lower()
    check(len(policy)==1 and len({v['image'] for v in hardware.values()})==1,'mixed initial weights or image')
    for rec in d['recipes'].values():
        rec['candidates']=schedules(rec['bands'], rec['measurements'], spec['widths'])
    return dict(probe=spec, source_policy=policy.pop(), containers=list(hardware.values()), **d)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('receipts','spec','output'):
        p.add_argument(arg,type=Path)
    a=p.parse_args()
    result=collect(a.receipts,json.loads(a.spec.read_bytes()))
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
