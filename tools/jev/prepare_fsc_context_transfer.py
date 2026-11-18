#!/usr/bin/env python3
"""Build label-free native FSC requests for the shared probability head.

No target labels, inference, model selection or semantic-presence supervision.
Existing native descriptors are retained verbatim as benchmark conventions.
"""
import argparse
import hashlib
import json
from pathlib import Path

from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_decision_contract import render_context_request
from multimodal_jev.fsc_span_transfer import validate
from multimodal_jev.slurp_training import content_hash


def compile_native(audio,task):
    validate(task)
    candidates=[dict(id=c['id'],definition=c['description']) for c in task['commands']]
    rules=[*task['thinking_requirements'],
        'The supplied action/object/location descriptors name the native FSC benchmark categories. '
        'Judge the assigned complete command, not independent semantic presence or execution feasibility.']
    return compile_candidates(audio,candidates,rules,semantics='exclusive_event')


def build(task_path,population_path,proof_path,output):
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    task_path,population_path,proof_path=map(Path,(task_path,population_path,proof_path))
    proof=json.loads(proof_path.read_text())
    if proof['files'].get(str(population_path))!=sha(population_path):
        raise ValueError('published input roster identity differs')
    population=json.loads(population_path.read_text());rows=population['records']
    if (population['schema']!='fsc-official-span-population-v1' or population['split']!='test' or
            population['role']!='frozen_transfer_evaluation_inputs' or
            population['recordings']!=len(rows) or len(rows)!=3793 or
            len({r['id'] for r in rows})!=len(rows) or
            population['speakers']!=len({r['speaker_id'] for r in rows}) or population['speakers']!=10):
        raise ValueError('complete official test input roster required')
    if any(set(r)!={'id','audio','audio_sha256','speaker_id'} for r in rows):
        raise ValueError('only label-free input roster fields permitted')
    task=json.loads(task_path.read_text());validate(task)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    target=output/'requests.jsonl'
    with target.open('x') as stream:
        for row in rows:
            request=compile_native(row['audio'],task)
            system,text=render_context_request(request)
            stream.write(json.dumps(dict(id=row['id'],audio_sha256=row['audio_sha256'],
                speaker_id=row['speaker_id'],request=request.model_input(),
                request_sha256=content_hash(request.model_input()),
                rendered_sha256=content_hash(dict(system=system,text=text))),separators=(',',':'))+'\n')
    report=dict(schema='fsc-context-exclusive-event-inputs-v1',recordings=len(rows),speakers=10,
        candidate_ids=[c['id'] for c in task['commands']],request_version='jev-context-exclusive_event-v1',
        task_sha256=sha(task_path),population_sha256=sha(population_path),proof_sha256=sha(proof_path),
        files={'requests.jsonl':sha(target)},training_dispatch_allowed=False,evaluation_dispatch_allowed=False,
        scope='Input-only full roster rendering; audio bytes and processor tensors not revalidated. '
              'No target gold, semantic-presence labels, model predictions or transfer claim.')
    (output/'MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('task','population','proof','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(build(a.task,a.population,a.proof,a.output),indent=2))
