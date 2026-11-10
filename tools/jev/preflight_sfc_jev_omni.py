#!/usr/bin/env python3
"""Actual CPU processor checks for every Jev decision in four TRAIN conversations."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc import source_identity
from multimodal_jev.sfc_audio import load_sfc_audio
from multimodal_jev.sfc_jev_runtime import messages, vllm_request, validate_bundle


def select(requests):
    groups = {}
    seen = set()
    for r in requests:
        validate_bundle(r)
        if r['split'] != 'train_id' or r['id'] in seen:
            raise ValueError('unique TRAIN requests required')
        seen.add(r['id'])
        level = r['source_group'].split('/')[0]
        groups.setdefault(level, set()).add(r['source_group'])
    if set(groups) != {'level_1','level_2','level_3_1','level_3_2'}:
        raise ValueError('all four SFC levels required')
    wanted = {min(v) for v in groups.values()}
    selected = [r for r in requests if r['source_group'] in wanted]
    for group in wanted:
        rows = [r for r in selected if r['source_group'] == group]
        lengths = sorted(len(r['input']['context']['audios']) for r in rows)
        if lengths != ([1,2] if group.startswith('level_3') else [1]):
            raise ValueError('complete chronological conversations required')
        for r in rows:
            paths = r['input']['context']['audios']
            # Verify input-side group/stage independently of targets.
            identity = source_identity([p[p.index('audio/final_data_audio/'):] for p in paths])
            if identity['source_group'] != group or identity['stage'] != len(paths):
                raise ValueError('input identity differs')
    if len(selected) != 6:
        raise ValueError('six TRAIN turns required')
    return selected


def expected_frames(lengths, hop):
    return [min((n+hop-1)//hop, max(lengths)//hop) for n in lengths]


def run(model, release, output):
    import torch
    from transformers import Qwen3OmniMoeProcessor
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessorKwargs
    torch.set_num_threads(4); start=time.monotonic()
    if Qwen3OmniMoeProcessorKwargs._defaults['audio_kwargs']['truncation'] is not False:
        raise ValueError('official nontruncating audio defaults required')
    manifest=json.loads((release/'MANIFEST.json').read_text())
    path=release/'train_id.requests.jsonl'
    if file_hash(path)!=manifest['splits']['train_id']['requests_sha256']:
        raise ValueError('TRAIN request file changed')
    # The target file is deliberately never opened.
    rows=select([json.loads(line) for line in path.open()])
    identity=dict(model=str(model),release=str(release),manifest_sha256=file_hash(release/'MANIFEST.json'),
        train_requests_sha256=file_hash(path),selected_ids=[r['id'] for r in rows],
        selection='lexicographically first complete TRAIN source group per each of four levels',
        versions={k:importlib.metadata.version(k) for k in ('transformers','torch','numpy','scipy','soundfile')},
        source_files={n:file_hash(Path(__file__).parents[2]/n) for n in
            ('tools/jev/preflight_sfc_jev_omni.py','src/multimodal_jev/sfc_jev_runtime.py',
             'src/multimodal_jev/sfc_jev.py','src/multimodal_jev/sfc_definitions.py','src/multimodal_jev/sfc_audio.py')},
        processor_assets={p.name:file_hash(p) for p in model.iterdir() if p.is_file() and p.suffix in ('.json','.txt','.jinja')},
        context_limit=4096,output_budget=1024)
    atomic_json(output/'identity.json',identity)
    processor=Qwen3OmniMoeProcessor.from_pretrained(model,local_files_only=True)
    total=sum(len(r['input']['decisions']) for r in rows); records=[]; swaps=[]
    for r in rows:
        waves=[];conversions=[]
        for audio,digest in zip(r['input']['context']['audios'],r['audio_sha256'],strict=True):
            if file_hash(audio)!=digest:raise ValueError('audio changed')
            wave,conversion=load_sfc_audio(audio);waves.append(wave);conversions.append(conversion)
        reference=None
        for decision in r['input']['decisions']:
            atomic_json(output/'progress.json',dict(phase='processor',done=len(records),total=total,heartbeat_unix=time.time()))
            did=decision['id']
            prompt=processor.apply_chat_template(messages(r,did,waves),tokenize=False,add_generation_prompt=True)
            engine_input=vllm_request(processor,r,did,waves)
            if engine_input['prompt']!=prompt:raise ValueError('HF and vLLM adapter prompt differs')
            if any(pair[0] is not wave or pair[1]!=16000 for pair,wave in zip(engine_input['multi_modal_data']['audio'],waves,strict=True)):
                raise ValueError('vLLM adapter waveform mapping differs')
            batch=processor(text=[prompt],audio=waves,sampling_rate=16000,return_tensors='pt',padding=True)
            expected=expected_frames([len(w) for w in waves],processor.feature_extractor.hop_length)
            observed=batch['feature_attention_mask'].sum(-1).tolist()
            if expected!=observed:raise ValueError('full waveform frame coverage failed')
            if not torch.isfinite(batch['input_features']).all():raise ValueError('nonfinite features')
            ids=batch['input_ids'][0]
            audio_id=processor.tokenizer.convert_tokens_to_ids('<|audio_pad|>')
            spans=[]
            for pos in (ids==audio_id).nonzero().flatten().tolist():
                if not spans or pos!=spans[-1][-1]+1:spans.append([])
                spans[-1].append(pos)
            if len(spans)!=len(waves):raise ValueError('audio segment count differs')
            if reference is None:
                reference=(batch['input_features'].clone(),batch['feature_attention_mask'].clone())
                if len(waves)==2:
                    swapped=processor(text=[prompt],audio=waves[::-1],sampling_rate=16000,return_tensors='pt',padding=True)
                    torch.testing.assert_close(reference[0],swapped['input_features'].flip(0),atol=1e-5,rtol=1e-5)
                    if not torch.equal(reference[1],swapped['feature_attention_mask'].flip(0)):
                        raise ValueError('swapped audio masks differ')
                    if torch.allclose(reference[0],swapped['input_features'],atol=1e-5,rtol=1e-5):
                        raise ValueError('distinct audio order control is degenerate')
                    swaps.append(dict(id=r['id'],maximum_error=float((reference[0]-swapped['input_features'].flip(0)).abs().max())))
            else:
                torch.testing.assert_close(batch['input_features'],reference[0],rtol=0,atol=0)
                if not torch.equal(batch['feature_attention_mask'],reference[1]):raise ValueError('decision changed audio masks')
            tokens=len(ids)
            if tokens+1024>4096:raise ValueError('registered context budget exceeded')
            records.append(dict(id=r['id'],decision_id=did,decision_type=decision['decision_type'],
                input_tokens=tokens,audio_token_lengths=list(map(len,spans)),expected_frames=expected,
                observed_frames=observed,conversions=conversions,
                prompt_sha256=__import__('hashlib').sha256(prompt.encode()).hexdigest(),
                input_ids_sha256=__import__('hashlib').sha256(canonical(ids.tolist())).hexdigest()))
    if len(records)!=total or len(swaps)!=2:raise ValueError('incomplete decision/order coverage')
    report=dict(status='passed',records=records,swap_controls=swaps,decisions=total,bundles=len(rows),
        min_input_tokens=min(r['input_tokens'] for r in records),max_input_tokens=max(r['input_tokens'] for r in records),
        identity_sha256=file_hash(output/'identity.json'),elapsed_seconds=time.monotonic()-start,
        scope='Actual CPU HF processor for all Jev decisions in six TRAIN turns/four complete conversations. '
              'Raw vLLM adapter prompt/payload checked, but no vLLM engine, model weights, generation, '
              'full dataset context fit or task efficacy evidence.')
    atomic_json(output/'result.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('model','release','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:
        result=run(a.model,a.release,a.output)
        print(json.dumps({k:v for k,v in result.items() if k not in ('records','swap_controls')},indent=2))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
