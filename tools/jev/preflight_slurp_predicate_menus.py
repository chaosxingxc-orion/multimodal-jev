#!/usr/bin/env python3
"""CPU processor cost of fixed predicate menus, never gold-selected menus.

Uses current published intent descriptions without pretending they constitute a
new reviewed natural-language ontology. No supervision, training or model scores.
"""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash


def run(config, output):
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.slurp_official_training import load
    from multimodal_jev.decision_questions import Concept,compile_menu
    from multimodal_jev.semantic_spec import SemanticSpec,compile_definition
    from multimodal_jev.span_contract import SpanQuestion,encode_span_request
    from multimodal_jev.task_schema import DecisionSchema
    from multimodal_jev.audio_policy import load_audio
    if list(Path('/dev').glob('davinci[0-9]*')):
        raise ValueError('CPU-only input cost audit required')
    torch.set_num_threads(4)
    for p,h in config['pins'].items():
        if file_hash(p)!=h:raise ValueError('pinned input changed')
    data,task,_=load(config['release'],config['release_sha256'],'learning',purpose='optimize')
    # Fixed prefix of authenticated population, chosen without label predicates.
    selected=data[:16]
    schema=DecisionSchema.from_dict(task['schema'])
    concepts=tuple(Concept(c.id,c.description) for c in schema.candidates)
    if len(concepts)!=60 or len(selected)!=16:raise ValueError('expected full ontology and fixed 16-input cost audit')
    spec=SemanticSpec.from_dict(json.loads(Path(config['semantic_spec']).read_text()))
    definitions={}
    for size in (0,2,8,60):
        questions=() if size==0 else compile_menu('intent_label_member',concepts[:size]).questions
        definitions[size]=compile_definition((*questions,SpanQuestion('intent',schema)),'intent',spec)
    processor=Qwen3ASRProcessor.from_pretrained(config['model'],local_files_only=True)
    records=[]
    for row in selected:
        if file_hash(row['audio'])!=row['audio_sha256']:raise ValueError('audio changed')
        wave,_=load_audio(row['audio'],return_receipt=True)
        for size,definition in definitions.items():
            request=definition.for_audio(row['audio'])
            inputs,positions=encode_span_request(processor,request,wave)
            n=inputs['input_ids'].shape[-1]
            records.append(dict(id=row['id'],menu_size=size,spans=len(request.questions),tokens=n,
                positions=positions.tolist(),over_context=n>4096,semantic_spec_sha256=definition.semantic_spec_sha256))
        atomic_json(output/'progress.json',dict(recordings=len(records)//4,requests=len(records),total=64,updated_unix=time.time()))
    return dict(passed=True,recordings=16,requests=len(records),records=records,
        menus={str(size):dict(minimum_tokens=min(r['tokens'] for r in records if r['menu_size']==size),
            maximum_tokens=max(r['tokens'] for r in records if r['menu_size']==size),
            over_context=sum(r['over_context'] for r in records if r['menu_size']==size)) for size in definitions},
        context_limit=4096,definition_basis='existing published scenario/action descriptions, not newly reviewed semantic ontology',
        scope='Completed input-length audit, including overflow records. No claim all menus fit, no population-wide acceptance, no training/efficacy, no label-based input selection.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json',run(json.loads(a.config.read_text()),a.output))
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
