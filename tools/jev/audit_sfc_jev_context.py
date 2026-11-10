#!/usr/bin/env python3
"""Actual untruncated CPU HF processing of every released SFC-Jev decision."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_audio import load_sfc_audio
from multimodal_jev.sfc_jev_runtime import messages
from preflight_sfc_jev_omni import expected_frames


def audio_spans(ids, audio_id):
    spans=[]
    for pos, token in enumerate(ids):
        if token != audio_id:continue
        if not spans or pos != spans[-1][-1]+1:spans.append([])
        spans[-1].append(pos)
    return [len(s) for s in spans]


def check_coverage(request, ids, observed_frames, lengths, hop, audio_id):
    spans=audio_spans(ids,audio_id)
    expected=expected_frames(lengths,hop)
    if len(spans)!=len(request['audio_sha256']) or len(lengths)!=len(spans):
        raise ValueError('expanded audio count differs')
    if observed_frames!=expected:
        raise ValueError('full waveform frame coverage failed')
    return dict(input_tokens=len(ids),audio_token_lengths=spans,expected_frames=expected,
                observed_frames=observed_frames,context_overflow=len(ids)+1024>4096,
                input_ids_sha256=hashlib.sha256(canonical(ids)).hexdigest())


def run(model, release, output, *, candidate=None):
    render=messages;prompt_identity=None
    if candidate is not None:
        from multimodal_jev.sfc_candidate_context import load_candidate
        from multimodal_jev.sfc_prompt_candidates import messages as candidate_messages
        instruction,prompt_identity=load_candidate(**candidate)
        render=lambda row,decision,audios:candidate_messages(row,decision,audios,instruction)
    import torch
    from transformers import Qwen3OmniMoeProcessor
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessorKwargs
    torch.set_num_threads(4);start=time.monotonic()
    if Qwen3OmniMoeProcessorKwargs._defaults['audio_kwargs']['truncation'] is not False:
        raise ValueError('official nontruncating audio defaults required')
    manifest=json.loads((release/'MANIFEST.json').read_text())
    if set(manifest['splits'])!={'train_id','test_id','test_ood'}:
        raise ValueError('all official splits required')
    identity=dict(model=str(model),release=str(release),manifest_sha256=file_hash(release/'MANIFEST.json'),
        versions={k:importlib.metadata.version(k) for k in ('transformers','torch','numpy','scipy','soundfile')},
        source_files={n:file_hash(Path(__file__).parents[2]/n) for n in
            ('tools/jev/audit_sfc_jev_context.py','tools/jev/preflight_sfc_jev_omni.py',
             'src/multimodal_jev/sfc_jev_runtime.py','src/multimodal_jev/sfc_jev.py',
             'src/multimodal_jev/sfc_definitions.py','src/multimodal_jev/sfc_audio.py',
             'src/multimodal_jev/sfc_omni.py','src/multimodal_jev/sfc_consumption.py')},
        processor_assets={p.name:file_hash(p) for p in model.iterdir() if p.is_file() and p.suffix in ('.json','.txt','.jinja')},
        context_limit=4096,output_budget=1024,targets_read=False)
    if prompt_identity is not None:
        identity['prompt']=prompt_identity
        for name in ('src/multimodal_jev/sfc_candidate_context.py','src/multimodal_jev/sfc_prompt_candidates.py',
                     'src/multimodal_jev/teacher_candidates.py','src/multimodal_jev/sfc_teacher_wire.py',
                     'src/multimodal_jev/sfc_teacher_feedback.py'):
            identity['source_files'][name]=file_hash(Path(__file__).parents[2]/name)
    atomic_json(output/'identity.json',identity)
    processor=Qwen3OmniMoeProcessor.from_pretrained(model,local_files_only=True)
    audio_id=processor.tokenizer.convert_tokens_to_ids('<|audio_pad|>')
    reports={};audio_hashes={}
    for split in ('train_id','test_id','test_ood'):
        pin=manifest['splits'][split];path=release/f'{split}.requests.jsonl'
        if file_hash(path)!=pin['requests_sha256']:raise ValueError('request release changed')
        counts=dict(bundles=0,decisions=0,context_overflow=0,long_audio_bundles=0,
                    min_input_tokens=None,max_input_tokens=0,max_audio_seconds=0.)
        ids=set();groups=set();record_path=output/f'{split}.context.jsonl'
        with path.open() as source,record_path.open('xb') as handle:
            for line in source:
                row=json.loads(line);uid=row['id']
                if row['split']!=split or uid in ids:raise ValueError('duplicate or wrong split')
                ids.add(uid);groups.add(row['source_group']);waves=[]
                for audio,digest in zip(row['input']['context']['audios'],row['audio_sha256'],strict=True):
                    if audio not in audio_hashes:audio_hashes[audio]=file_hash(audio)
                    if audio_hashes[audio]!=digest:raise ValueError('audio changed')
                    waves.append(load_sfc_audio(audio)[0])
                lengths=[len(w) for w in waves];durations=[n/16000 for n in lengths]
                counts['long_audio_bundles']+=any(n>processor.feature_extractor.n_samples for n in lengths)
                counts['max_audio_seconds']=max(counts['max_audio_seconds'],*durations)
                for decision in row['input']['decisions']:
                    prompt=processor.apply_chat_template(render(row,decision['id'],waves),tokenize=False,add_generation_prompt=True)
                    batch=processor(text=[prompt],audio=waves,sampling_rate=16000,return_tensors='pt',padding=True)
                    if not torch.isfinite(batch['input_features']).all():raise ValueError('nonfinite audio features')
                    coverage=check_coverage(row,batch['input_ids'][0].tolist(),batch['feature_attention_mask'].sum(-1).tolist(),
                                            lengths,processor.feature_extractor.hop_length,audio_id)
                    count=coverage['input_tokens']
                    counts['min_input_tokens']=count if counts['min_input_tokens'] is None else min(counts['min_input_tokens'],count)
                    counts['max_input_tokens']=max(counts['max_input_tokens'],count)
                    counts['context_overflow']+=coverage['context_overflow'];counts['decisions']+=1
                    handle.write(canonical(dict(id=uid,decision_id=decision['id'],audio_seconds=durations,
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),**coverage)))
                counts['bundles']+=1;handle.flush()
                atomic_json(output/'progress.json',dict(phase='processor',split=split,**counts,
                    total_bundles=pin['bundles'],total_decisions=pin['decisions'],heartbeat_unix=time.time()))
        if (counts['bundles']!=pin['bundles'] or counts['decisions']!=pin['decisions']
                or len(groups)!=pin['source_groups']):raise ValueError('complete split coverage required')
        reports[split]=dict(**counts,source_groups=len(groups),records_sha256=file_hash(record_path))
        atomic_json(output/'completed-splits.json',reports)
    result=dict(status='passed',identity_sha256=file_hash(output/'identity.json'),splits=reports,
        all_inputs_fit=all(r['context_overflow']==0 for r in reports.values()),audio_files_hashed=len(audio_hashes),
        elapsed_seconds=time.monotonic()-start,
        scope='Actual CPU HF processor for every decision in all released splits, including audio longer than 30 seconds. '
              'Input-only context audit, no model weights, labels, generation, vLLM-engine parity or task-quality claim.')
    if candidate is not None:
        _,current=load_candidate(**candidate)
        if current!=prompt_identity:raise ValueError('candidate changed during processor audit')
        result['prompt']=prompt_identity
    atomic_json(output/'result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('model','release','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--staging',type=Path);p.add_argument('--staging-hash');p.add_argument('--candidate-id')
    a=p.parse_args();candidate=None
    if any(v is not None for v in (a.staging,a.staging_hash,a.candidate_id)):
        if not all((a.staging,a.staging_hash,a.candidate_id)):p.error('all candidate pins required')
        candidate=dict(staging=a.staging,staging_hash=a.staging_hash,candidate_id=a.candidate_id)
    a.output.mkdir(parents=True,exist_ok=False)
    try:print(json.dumps(run(a.model,a.release,a.output,candidate=candidate),indent=2))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
