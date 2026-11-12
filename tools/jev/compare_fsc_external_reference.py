#!/usr/bin/env python3
"""Compare external academic predictions only after complete transfer admission."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.fsc_external_comparison import compare_native
from multimodal_jev.fsc_transfer_population import scoring_targets
from score_fsc_span_transfer import run as score_matrix,read
from audit_fsc_reference_alignment import run as authenticate_reference


def run(protocol_path,protocol_sha256,reference_config):
    full=score_matrix(protocol_path,protocol_sha256)
    # This call precedes any published-reference label scoring as well.
    config=read(reference_config);alignment=authenticate_reference(config);protocol=read(protocol_path)
    if config['population']!=protocol['population'] or alignment['population_sha256']!=protocol['population_sha256']:
        raise ValueError('external reference population differs from frozen matrix')
    population=read(protocol['population']);task=read(protocol['task']);receipt=read(config['reference_receipt'])
    prediction_path=Path(receipt['run'])/'predictions.jsonl'
    predictions=[json.loads(line) for line in prediction_path.read_text().splitlines()]
    result=compare_native(task,population['records'],scoring_targets(population,task),predictions,full['policies'])
    baseline=result['reference'];expected=receipt['result']
    if baseline['correct']!=expected['joint_correct'] or baseline['recordings']!=expected['examples']:
        raise ValueError('recomputed external accuracy differs from accepted official score')
    files={**full['files'],**config['pins']}
    for p in [Path(reference_config),prediction_path,Path(__file__),Path(__file__).resolve().parents[2]/'src/multimodal_jev/fsc_external_comparison.py']:
        files[str(p.resolve())]=file_hash(p)
    return dict(result,protocol_sha256=protocol_sha256,files=files)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('protocol','protocol-sha256','reference-config'):p.add_argument('--'+name,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable comparison exists')
    atomic_json(a.output,run(a.protocol,a.protocol_sha256,a.reference_config))
