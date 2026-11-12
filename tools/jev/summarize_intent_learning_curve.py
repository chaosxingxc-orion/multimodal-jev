#!/usr/bin/env python3
"""Re-score an explicitly supplied accepted prefix against the fixed protocol."""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_learning_curve import summarize
from queue_span_native_actor_npu import read,completed
from queue_span_dp4_resume import check_pins
from queue_intent_finite_npu import verify_final,verify_gate
from score_intent_learning_policy import run as score


def run(config):
    check_pins(config['pins']);protocol_path=Path(config['protocol']);protocol=read(protocol_path)
    if config['pins'].get(str(protocol_path))!=file_hash(protocol_path):raise ValueError('fixed protocol required')
    points=[];evidence={};previous_gate=None
    for index,entry in enumerate(config['points']):
        if entry['update']!=index:raise ValueError('contiguous explicit prefix required')
        gate=Path(entry['gate']);launch=read(gate/'launch.json')
        if not completed(launch['container_id'],config['image']) or launch['image']!=config['image']:
            raise ValueError('actual accepted gate container required')
        check_pins({str(gate/'config.json'):launch['config_sha256'],str(gate/'source-manifest.json'):launch['source_manifest_sha256']})
        verify_gate(gate,config['pilot'],config['manifest_sha256'])
        if index:
            verify_final(gate);verified=read(gate/'verified.json');training=Path(entry['training']);cfg=read(training/'config.json')
            training_launch=read(training/'launch.json')
            if not completed(training_launch['container_id'],config['image']):raise ValueError('training must complete')
            check_pins({str(training/'config.json'):training_launch['config_sha256']})
            selection=read(training/'run/selection.json')
            if (verified['root']!=str(training) or verified['suite']!=str(gate) or verified['updates']!=index
                    or cfg['step']!=index-1 or cfg['acceptance_root']!=str(previous_gate)
                    or cfg['arm']!=protocol['arm'] or cfg['seed']!=protocol['seed'] or cfg['lr']!=protocol['lr']
                    or cfg['pilot']!=config['pilot'] or cfg['manifest_sha256']!=config['manifest_sha256']
                    or selection['recording_ids']!=protocol['schedule'][index-1]):
                raise ValueError('checkpoint lineage or frozen schedule differs')
            for path in (training/'config.json',training/'run/selection.json',gate/'verified.json',gate/'ACCEPTED.json'):
                evidence[str(path)]=file_hash(path)
        pins={str(gate/p):file_hash(gate/p) for p in ('config.json','run/READY.json','run/terminal.json','run/vllm/result.json','run/vllm/terminal.json')}
        scored=score(dict(gate_root=str(gate),pilot=config['pilot'],manifest_sha256=config['manifest_sha256'],pins=pins))
        points.append(dict(update=index,score=scored));evidence.update(pins);previous_gate=gate
    result=summarize(points,protocol['schedule'],protocol['total_updates'])
    for path in (protocol_path,Path(__file__),Path(__file__).resolve().parents[2]/'src/multimodal_jev/intent_learning_curve.py',Path(__file__).resolve().parents[2]/'src/multimodal_jev/intent_policy_metrics.py'):
        evidence[str(path)]=file_hash(path)
    return dict(result,files=evidence,protocol_sha256=file_hash(protocol_path))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable curve already exists')
    atomic_json(a.output,run(read(a.config)))
