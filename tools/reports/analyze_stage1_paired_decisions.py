"""Consume existing audited Stage1 predictions; no model execution or selection."""
import hashlib
import json
from pathlib import Path
import argparse
from tools.reports.context_paired_decisions import summarize


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def run(comparison_path, initial_root, endpoint_root, release):
    comparison = json.loads(comparison_path.read_text())
    if (comparison['population'] != 9226 or not comparison['same_ordered_ids_processor_inputs_and_positions']
            or [s['step'] for s in comparison['states']] != [0, 634]):
        raise ValueError('Complete audited Stage1 pair required')
    sources = {str(comparison_path): sha(comparison_path)}
    manifest_path = release / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    sources[str(manifest_path)] = sha(manifest_path)
    for name in ('selection.jsonl', 'tasks.json'):
        if sha(release / name) != manifest['files'][name]:
            raise ValueError('Released selection labels or ontology changed')
    records = []
    for state, root in zip(comparison['states'], (initial_root, endpoint_root)):
        proof_path = Path(state['proof'])
        if sha(proof_path) != state['proof_sha256']:
            raise ValueError('Prior independent proof changed')
        proof = json.loads(proof_path.read_text())
        if (proof['state']['ExitCode'] != 0 or proof['state']['OOMKilled']
                or proof['inference_policy_id'] != state['policy_id']
                or proof['result']['release_sha256'] != sha(manifest_path)):
            raise ValueError('Successful bound inference required')
        p = root / 'selection/predictions.json'
        if sha(p) != proof['files']['selection/predictions.json']:
            raise ValueError('Audited prediction artifact changed')
        sources[str(proof_path)] = sha(proof_path); sources[str(p)] = sha(p)
        predictions = json.loads(p.read_text())['records']
        if len(predictions) != 9226 or len({p['id'] for p in predictions}) != 9226:
            raise ValueError('Complete unique prediction population required')
        for p in predictions:
            if p['policy_id'] != state['policy_id']:
                raise ValueError('Prediction policy mismatch')
        records.append(predictions)
    p = release / 'selection.jsonl'
    sources[str(p)] = sha(p)
    gold = [json.loads(line) for line in p.read_text().splitlines()]
    if len(gold) != 9226 or len({r['id'] for r in gold}) != 9226:
        raise ValueError('Complete unique selection labels required')
    p = release / 'tasks.json';sources[str(p)] = sha(p)
    candidates = [c['id'] for c in json.loads(p.read_text())['intent']['schema']['candidates']]
    if len(candidates) != 60 or candidates != proof['result']['metrics']['candidate_ids']:
        raise ValueError('Frozen candidate order mismatch')
    labels = {r['id']: r for r in gold}; paired = []
    if set(labels) != {r['id'] for r in records[0]}:
        raise ValueError('Prediction/label population mismatch')
    for a,b in zip(*records, strict=True):
        if a['id'] != b['id']:
            raise ValueError('Paired recording order differs')
        for key in ('inputs', 'positions'):
            if a['request_identity'][key] != b['request_identity'][key]:
                raise ValueError('Paired executed input differs')
        target = labels[a['id']]
        if target['role'] != 'selection':
            raise ValueError('Selection-only diagnostic')
        paired.append(dict(id=a['id'],source_group=target['source_group'],
                           gold=candidates.index(target['intent']),initial=a['prob'],selected=b['prob']))
    result = summarize(paired)
    for metric, original in [('top1','top1_event_accuracy'),('exact_set','exact_set'),('set_f1','query_macro_set_f1')]:
        for role,state in zip(('initial','selected'),comparison['states']):
            if abs(result['metrics'][metric]['recording_weighted'][role]-state['metrics'][original]) > 1e-12:
                raise ValueError('Recomputed metrics differ from prior independent score')
    result.update(sources=sources,stage='Stage1 adaptation only; not RL, TEST or native ASR zero-shot',
                  comparison_policy_ids=[s['policy_id'] for s in comparison['states']])
    for name in (__file__, str(Path(__file__).with_name('context_paired_decisions.py'))):
        result['sources'][name]=sha(Path(name))
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for key in ('comparison','initial_root','endpoint_root','release','output'):
        p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    a=p.parse_args()
    report=run(a.comparison,a.initial_root,a.endpoint_root,a.release)
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2))
