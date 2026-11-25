"""Verify equal-input comparisons and summarize compute-only time per cohort."""
import json
import math
import statistics
import sys
from pathlib import Path


def collect(root, spec):
    hardware = {r['cid']:r for r in json.loads((root/'inspection.json').read_bytes())}
    cases = []
    grouped = {}
    for i,lane in enumerate(spec['lanes']):
        cid = (root/f'r2-cid{i}').read_text().strip()
        hw = hardware[cid]
        if (hw['state']!='exited' or hw['code'] or hw['device']!=f'/dev/davinci{i}' or
                int((root/f'r2-exit{i}').read_text())!=0):
            raise ValueError('unsuccessful device lane')
        line, = [s[8:] for s in (root/f'r2-log{i}').read_text().splitlines() if s.startswith('REAL_AB ')]
        r = json.loads(line)
        if (any(r[k]!=v for k,v in lane.items()) or r['panel_sha256']!=spec['panel_sha256']
                or r['pairs']!=1+sum(spec['contrast_samples'].values()) or
                len(r['lengths'])*r['width']!=r['anchors'] or len(r['runs'])!=spec['warmup']+spec['repeats']):
            raise ValueError('run differs from registered panel/geometry')
        times = [q['seconds'] for q in r['runs'][spec['warmup']:]]
        if any(len(row)!=3 or any(not math.isfinite(x) or x<=0 for x in row) for row in times):
            raise ValueError('incomplete timings')
        r['median_phase_seconds'] = list(map(statistics.median,zip(*times)))
        r['anchors_per_second'] = r['anchors']/statistics.median(map(sum,times))
        witnesses = (r['lengths'],[x['witness'] for x in r['runs']])
        grouped.setdefault((r['band'],r['width']),[]).append(witnesses)
        cases.append(dict(**r,container=cid))
    if (any(len(v)!=2 or v[0]!=v[1] for v in grouped.values()) or
            len({r['image'] for r in hardware.values()})!=1 or len({(r['policy'],r['anchors']) for r in cases})!=1):
        raise ValueError('graph comparison is not aligned')
    return dict(spec=spec, image=next(iter(hardware.values()))['image'],equal_witnesses=True,cases=cases)


if __name__=='__main__':
    root, config, output = map(Path,sys.argv[1:])
    output.write_text(json.dumps(collect(root,json.loads(config.read_bytes())),indent=2)+'\n')
