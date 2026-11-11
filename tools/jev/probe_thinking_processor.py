#!/usr/bin/env python3
"""Real CPU processor contract check; no model, teacher or dataset gold."""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.audio_policy import encode_prompt, hf_prompt, load_audio, teacher_forcing, vllm_request
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.consumption import canonical, request_from_example
from multimodal_jev.task_schema import DecisionSchema, JevExample
from multimodal_jev.thinking import ThinkingSpec


def run(config):
    import torch
    from transformers import Qwen3ASRProcessor
    torch.set_num_threads(2)
    for name, digest in config['source_hashes'].items():
        if file_hash(Path(config['source'])/name) != digest:
            raise ValueError('source changed: '+name)
    source = Path(config['source_requests'])
    if file_hash(source) != config['source_requests_sha256']:
        raise ValueError('source request release changed')
    selected = None
    with source.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['id'] == config['example_id']:
                if selected is not None:
                    raise ValueError('duplicate selected source')
                selected = row
    if selected is None or selected['role'] != 'learning' or selected['official_split'] != 'train':
        raise ValueError('registered learning source required')
    if file_hash(selected['input']['audio']) != selected['audio_sha256']:
        raise ValueError('audio changed')
    model = Path(config['processor_path'])
    for name, digest in config['processor_hashes'].items():
        if file_hash(model/name) != digest:
            raise ValueError('processor changed')
    processor = Qwen3ASRProcessor.from_pretrained(model, local_files_only=True)
    audio = load_audio(selected['input']['audio'])
    # The boolean is a synthetic serialization fixture, NOT an annotation for
    # this recording. Only the request source (no target file) is opened.
    base = JevExample(selected['id'], 'train', selected['input']['audio'],
        DecisionSchema('fixture','noul','Evaluate the configured proposition.'), True,
        {'source_group':selected['source_group'],'audio_sha256':selected['audio_sha256'],
         'transcript_reference':'FORBIDDEN_REFERENCE_SENTINEL'})
    configurations = [None,
        ThinkingSpec('evidence-v1',('Check negation scope using the complete audio.',)),
        ThinkingSpec('evidence-v2',('Check modifier attachment using the complete audio.',))]
    reports=[]; batches=[]; prompt_texts=[]
    for spec in configurations:
        example=replace(base,thinking=spec)
        text=encode_prompt(processor,example,audio)
        request=vllm_request(processor,example,audio)
        batch=hf_prompt(processor,example,audio)
        assert request['prompt']==text
        assert request['multi_modal_data']['audio'][0] is audio
        assert 'FORBIDDEN_REFERENCE_SENTINEL' not in text
        if spec is not None:
            assert all(r in text for r in spec.requirements)
        assert encode_prompt(processor,replace(example,target=False),audio)==text
        prepared,boundary=teacher_forcing(processor,example,audio)
        assert boundary==batch['input_ids'].shape[-1]
        assert torch.equal(prepared['input_ids'][:,:boundary],batch['input_ids'])
        assert torch.all(prepared['labels'][:,:boundary]==-100)
        assert torch.all(prepared['labels'][:,boundary:]!=-100)
        decoded=processor.tokenizer.decode(batch['input_ids'][0],skip_special_tokens=False)
        if spec is not None:
            assert all(r in decoded for r in spec.requirements)
        reports.append(dict(thinking=spec.to_dict() if spec else None,
            prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
            request_sha256=hashlib.sha256(canonical(request_from_example(example))).hexdigest(),
            input_ids_sha256=hashlib.sha256(batch['input_ids'].numpy().tobytes()).hexdigest(),
            prompt_tokens=boundary,response_tokens=int((prepared['labels']!=-100).sum()),
            audio_frames=int(batch['input_features_mask'].sum()),
            feature_shape=list(batch['input_features'].shape)))
        batches.append(batch);prompt_texts.append(text)
    assert len({r['input_ids_sha256'] for r in reports})==3
    assert len({r['request_sha256'] for r in reports})==3
    for batch in batches[1:]:
        assert torch.equal(batches[0]['input_features'],batch['input_features'])
        assert torch.equal(batches[0]['input_features_mask'],batch['input_features_mask'])
    return dict(status='passed',example_id=selected['id'],audio_sha256=selected['audio_sha256'],
        source_requests_sha256=config['source_requests_sha256'],processor_hashes=config['processor_hashes'],
        source_hashes=config['source_hashes'],profiles=reports,features_exact_across_specs=True,
        prompt_requirements_survive_tokenization=True,teacher_forcing_prompt_mask_verified=True,
        vllm_request_render_verified=True,vllm_engine_executed=False,model_loaded=False,
        target_files_opened=False,weight_updates=0,
        packages={n:importlib.metadata.version(n) for n in ('torch','transformers')},
        scope='Real CPU processor/tokenizer and synthetic response-label plumbing on one clean TRAIN learning waveform; no annotation claim, model reasoning, vLLM engine, NPU or RL validation.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'terminal.json').exists():raise FileExistsError('no automatic retry')
    start=time.monotonic()
    try:
        result=run(json.loads(args.config.read_text()))
        result['elapsed_seconds']=time.monotonic()-start
        result['config_sha256']=file_hash(args.config)
        atomic_json(args.output/'result.json',result)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(args.output/'terminal.json',dict(exit_code=0))
