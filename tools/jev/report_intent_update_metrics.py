#!/usr/bin/env python3
"""Reauthenticate one completed RL pair and report full-population update metrics."""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_update_metrics import summarize,training_metrics
from multimodal_jev.slurp_intent_pilot import load_pilot
from queue_span_native_actor_npu import read
from compare_intent_learning_update import run as compare


def run(training,suite):
    training,suite = Path(training),Path(suite)
    checked = compare(training,suite)  # Actual exits, restore/export and full gate first.
    cfg = read(training/'config.json'); old = Path(cfg['acceptance_root'])
    wire = 'D' if cfg['arm']=='D' else 'Q'
    items,_ = load_pilot(cfg['pilot'],cfg['manifest_sha256'],cfg['arm'])
    outputs = [{r['id']:r for r in read(p/'run/vllm/result.json')['records']} for p in (old,suite)]
    selected = read(training/'run/selection.json')['recording_ids']
    rows = [dict(id=i['id'],weight=i['target']['request_weight'],target=i['target']['targets'][-1],
        old=outputs[0][i['id']+'::'+wire]['candidate_logprobs'][-1],
        new=outputs[1][i['id']+'::'+wire]['candidate_logprobs'][-1]) for i in items]
    result = summarize(rows,selected)
    ranks = [training/'run'/f'rank-{rank}'/'result.json' for rank in range(4)]
    trace = training_metrics([read(p) for p in ranks])
    if (trace['updates'] != cfg['step']+1 or trace['adapter_id'] != checked['after_adapter_id']):
        raise ValueError('training trace differs from independently accepted policy')
    source = training/'source/tools/jev/train_intent_finite_npu.py'
    # This report's clip threshold is meaningful only for the executed source.
    if 'grad_clip=1.' not in source.read_text():
        raise ValueError('fixed gradient clipping contract missing')
    files = dict(checked['pins'])
    for p in ranks+[source,Path(__file__).resolve(),
                   Path(__file__).resolve().parents[2]/'src/multimodal_jev/intent_update_metrics.py',
                   Path(__file__).resolve().parents[2]/'src/multimodal_jev/span_policy_drift.py']:
        files[str(p)] = file_hash(p)
    return dict(result,training_root=str(training),suite_root=str(suite),training_arm=cfg['arm'],
                training_trace=trace,files=files)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('training','suite','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable report already exists')
    atomic_json(a.output,run(a.training,a.suite))
