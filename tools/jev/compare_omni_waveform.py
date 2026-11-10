#!/usr/bin/env python3
"""Compare frozen actual/zero TRAIN outputs, separating sensitivity from accuracy."""
import argparse,json
from pathlib import Path
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.omni_probe_contract import select_learning
from multimodal_jev.waveform_control import transitions
from multimodal_jev.six_view_score import score_six_views


def load(root,condition):
    identity=json.loads((root/'identity.json').read_text());cfg=identity['config']
    result=json.loads((root/'result.json').read_text())
    if (json.loads((root/'terminal.json').read_text())['exit_code']!=0 or result['status']!='passed'
            or result['weight_updates']!=0 or result['model_files_unchanged'] is not True
            or result['audio_condition']!=condition or cfg['audio_condition']!=condition
            or len(cfg['source_groups'])!=64):raise ValueError('complete frozen64TRAIN condition required')
    if (file_hash(root/'predictions.jsonl')!=result['predictions_sha256']
            or file_hash(cfg['requests'])!=identity['requests_sha256']
            or file_hash(cfg['instruction'])!=identity['instruction_sha256']):raise ValueError('input/output changed')
    rows=select_learning(cfg,[json.loads(line) for line in Path(cfg['requests']).open()])
    pred=[json.loads(line) for line in (root/'predictions.jsonl').open()];indexed={r['example_id']:r for r in pred}
    selected=json.loads((root/'selection.json').read_text())['ids']
    if (len(indexed)!=len(pred) or len(pred)!=384 or result['predictions']!=384 or result['requests']!=384
            or len(selected)!=384 or set(selected)!=set(indexed) or set(indexed)!={r['id'] for r in rows}
            or any(r['replicate']!=0 or r['audio_condition']!=condition for r in pred)):
        raise ValueError('complete unique predictions required')
    return identity,rows,indexed


def compare(actual,zeros):
    ai,rows,a=load(actual,'actual');zi,zrows,z=load(zeros,'zeros')
    for key in ['requests_sha256','instruction_sha256','asset_audit_sha256','model_files','runner_sha256','source_sha256','versions']:
        if ai[key]!=zi[key]:raise ValueError('actor identity differs: '+key)
    ac,zc=dict(ai['config']),dict(zi['config'])
    for c in [ac,zc]:
        for k in ['audio_condition','instruction','asset_audit','partition']:c.pop(k)
    if ac!=zc or rows!=zrows:raise ValueError('only waveform condition may differ')
    targets=Path(ac['requests']).with_name('train.targets.jsonl');gold=[json.loads(line) for line in targets.open()]
    # Silence has no native task labels. Score only the real-audio condition.
    metrics=score_six_views(rows,gold,list(a.values()))
    return dict(schema='jev-omni-waveform-control-v1',actual_metrics=metrics,
        waveform_sensitivity=transitions(a,z),targets_sha256=file_hash(targets),
        actual_result_sha256=file_hash(actual/'result.json'),zeros_result_sha256=file_hash(zeros/'result.json'),
        scope='fixed64TRAIN learning groups; not TEST efficacy; zeros not assigned gold unknown labels')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['actual','zeros','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();report=compare(a.actual,a.zeros)
    with a.output.open('xb') as f:f.write(canonical(report))
    print(json.dumps(report,indent=2))
