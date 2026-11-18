#!/usr/bin/env python3
"""Project fixed semantic input lengths from pinned audio metadata, not acceptance."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audio_tokens(samples):
    # Inspected CPU Qwen3ASRFeatureExtractor: raw mask sampled every160,
    # padded to480000 (divisible by160); min_length8000. No truncation.
    # Processor groups100 mel frames,13 tokens/full group, ceil(remainder/8).
    require(type(samples) is int and 8000 <= samples <= 480000,
            'projection restricted to original 0.5–30s audio at16kHz')
    frames = (samples + 159) // 160
    return (frames // 100) * 13 + ((frames % 100) + 7) // 8


def fixed_lengths(records):
    groups = defaultdict(list)
    for r in records:
        require(r['audio_tokens'] == audio_tokens(r['audio_samples']), 'audio length formula differs')
        groups[r['panel_id'], r['arm']].append(r)
    result = {}
    for key, pair in groups.items():
        require(len(pair) == 2 and len({r['audio_id'] for r in pair}) == 2,
                'two distinct measured audio anchors required')
        constants = {r['tokens'] - r['audio_tokens'] for r in pair}
        require(len(constants) == 1, 'non-audio length changed')
        positions = {tuple(p-r['audio_tokens'] for p in r['positions'][0]) for r in pair}
        require(len(positions) == 1, 'readout shift differs')
        require(len({r['template_sha256'] for r in pair}) == 1, 'template changed')
        result[key] = constants.pop()
    return result


def summarize(audio, lengths):
    require(bool(audio) and len({a['id'] for a in audio}) == len(audio), 'unique nonempty audio roster required')
    require(all(a['sample_rate'] == 16000 for a in audio), '16kHz only')
    counts = Counter(audio_tokens(a['samples']) for a in audio)
    n = len(audio)
    panels = []
    histogram = Counter()
    for panel, fixed in sorted(lengths.items()):
        h = Counter({fixed+k: count for k, count in counts.items()})
        histogram.update(h)
        panels.append(dict(panel_id=panel, non_audio_tokens=fixed, requests=n,
            min_tokens=min(h), max_tokens=max(h), total_tokens=sum(k*v for k,v in h.items()),
            overflow={str(c):sum(v for k,v in h.items() if k>c) for c in (8192,16384,32768,65536)}))
    total = sum(k*v for k,v in histogram.items())
    requests = sum(histogram.values())
    expanded = np.repeat(list(histogram), list(histogram.values()))
    return dict(recordings=n, panels=len(lengths), requests=requests,
        total_input_tokens=total, mean_input_tokens=total/requests,
        quantiles={str(q):float(np.quantile(expanded,q,method='linear')) for q in (0,.5,.9,.95,.99,1)},
        overflow={str(c):sum(v for k,v in histogram.items() if k>c) for c in (8192,16384,32768,65536)},
        audio_token_fraction=(sum(k*v for k,v in counts.items())*len(lengths))/total,
        length_histogram={str(k):v for k,v in sorted(histogram.items())}, per_panel=panels,
        dense_attention_pair_proxy=sum(k*k*v for k,v in histogram.items()),
        proxy_scope='Sum of sequence_length squared; not FLOPs, measured memory or latency. No padding, packing, prefix sharing or G multiplier.')


def project(probe, release, output):
    require(not output.exists(), 'output exists')
    config = read(probe/'config.json'); result = read(probe/'run/result.json')
    terminal = read(probe/'terminal.json'); state = terminal['container_state']
    require(state['Status']=='exited' and state['ExitCode']==0 and not state['OOMKilled'], 'completed probe required')
    require(result['config_sha256']==sha(probe/'config.json') and result['records_sha256']==sha(probe/'run/records.jsonl'), 'changed probe')
    pins = {str(probe/p):sha(probe/p) for p in ('config.json','run/result.json','run/records.jsonl','terminal.json','preflight.json')}
    for path,digest in {**config['pins'],**read(probe/'preflight.json')['source_pins']}.items():
        require(sha(path)==digest, 'changed dependency '+path)
        pins[path] = digest
    manifest = read(release/'MANIFEST.json')
    pins[str(release/'MANIFEST.json')] = sha(release/'MANIFEST.json')
    require(pins[str(release/'MANIFEST.json')]=='38e4273d9328e4f02db081315e46f97a17628b974be7ba175254882e95421482', 'unexpected population')
    records = rows(probe/'run/records.jsonl'); lengths = fixed_lengths(records)
    templates = rows(Path(config['templates']))
    require(set(lengths)=={(t['panel_id'],t['arm']) for t in templates} and len(lengths)==34, 'full measured template coverage required')
    populations = {}; rosters = {}
    for role in ('learning','selection'):
        for file in (role+'.audio.jsonl',role+'.jsonl'):
            require(sha(release/file)==manifest['files'][file], 'changed release file')
            pins[str(release/file)] = sha(release/file)
        audio = rows(release/(role+'.audio.jsonl')); native = rows(release/(role+'.jsonl'))
        ids = {r['id']:r for r in native}; rosters[role]=set(ids)
        require(len(ids)==len(native)==len(audio)==manifest['counts'][role]['recordings'], 'population mismatch')
        require({a['id'] for a in audio}==set(ids), 'audio roster mismatch')
        require(all(a['audio_sha256']==ids[a['id']]['audio_sha256'] and ids[a['id']]['role']==role for a in audio), 'audio binding mismatch')
        populations[role] = {arm:summarize(audio,{p:n for (p,a),n in lengths.items() if a==arm}) for arm in ('Direct','Q_input')}
    require(not rosters['learning'] & rosters['selection'], 'cross-role overlap')
    report = dict(schema='semantic-context-population-projection-v1', pins=pins,
        implementation_sha256=sha(__file__), populations=populations,
        context_accepted=False, training_dispatch_allowed=False, selected_context_limit=None,
        new_processor_requests=0, new_model_inference=False,
        formula='frames=ceil(samples/160); audio_tokens=13*floor(frames/100)+ceil((frames%100)/8); total=measured_non_audio_tokens+audio_tokens',
        assumptions=['CPU diagnostic processor/settings identical to measured68 requests',
            'one unaugmented mono16kHz audio,8000<=samples<=480000,padding max_length480000,no truncation',
            'canonical fixed templates and stable special-token replacement; Q_credit identical to Q_input',
            'length projection from pinned metadata; waveform files not reread or decoded in this audit'],
        scope='Full learning/selection metadata projection, not full processor execution, fixed-training-image or NPU acceptance. No labels used to select templates. No TEST data. G16 is not16 audio forwards.')
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=project(a.probe,a.release,a.output)
    print(json.dumps({role:{arm:{k:v for k,v in stats.items() if k not in ('length_histogram','per_panel')} for arm,stats in arms.items()} for role,arms in r['populations'].items()},indent=2))
