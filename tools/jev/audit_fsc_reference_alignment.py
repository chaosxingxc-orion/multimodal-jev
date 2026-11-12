#!/usr/bin/env python3
"""Authenticate a historical official baseline and align audio without labels."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.fsc_reference_alignment import align
from queue_span_dp4_resume import inspect, check_pins
from queue_span_native_actor_npu import read


def run(config):
    check_pins(config['pins'])
    for key in ('population','reference_receipt'):
        if config['pins'].get(config[key]) != file_hash(config[key]): raise ValueError('fixed reference assets required')
    receipt=read(config['reference_receipt']); root=Path(receipt['run']); population=read(config['population'])
    state=inspect([receipt['container_id']])[receipt['container_id']]
    if state['Image']!=config['image'] or state['State']['Status']!='exited' or state['State']['ExitCode']!=0:
        raise ValueError('actual completed official baseline required')
    identity=read(root/'identity.json'); result=read(root/'result.json'); predictions=root/'predictions.jsonl'
    if (read(root/'terminal.json')['exit_code']!=0 or file_hash(root/'identity.json')!=receipt['identity_sha256']
            or file_hash(predictions)!=receipt['result']['predictions_sha256']
            or result!=receipt['result'] or identity['config']['split']!='test'
            or identity['csv_sha256']!=population['source_csv_sha256']['test']):
        raise ValueError('historical run identity changed')
    for name, verification in identity['models'].items():
        model=Path(identity['config'][name])
        check_pins({str(model/e['file']):e['sha256'] for e in verification['files']})
    rows=[json.loads(line) for line in predictions.read_text().splitlines()]
    aligned=align(population,rows)
    if result['examples']!=aligned['recordings'] or result['predictions']!=aligned['recordings'] or result['missing']!=0:
        raise ValueError('historical denominator differs')
    aligned.update(reference_container_id=receipt['container_id'],actual_state=state['State'],
        population_sha256=file_hash(config['population']),predictions_sha256=file_hash(predictions),
        reference_receipt_sha256=file_hash(config['reference_receipt']),
        previously_verified_joint_correct=result['joint_correct'],
        previously_verified_joint_accuracy=result['joint_accuracy'],
        metric_source='Existing verified result only; no labels read or new score computed.',
        comparison_conditions='Task-trained SpeechBrain official weights, native semantics. ASR Jev is SLURP-trained transfer; no matched-training or binding-view equivalence claim.')
    return aligned


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable alignment exists')
    atomic_json(a.output,run(read(a.config)))
