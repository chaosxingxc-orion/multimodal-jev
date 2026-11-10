#!/usr/bin/env python3
"""Run independent scoring prompt reconstruction across every native input."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.audio2tool_acceptance import prompt_expander
from multimodal_jev.audio2tool_engine import prior_records
from multimodal_jev.audio2tool_inference import read_population,digest,write_once,INPUT_SHA,BUDGET_SHA
from multimodal_jev.baseline import atomic_json,file_hash


def run(args):
    from transformers import Qwen3OmniMoeProcessor
    start=time.monotonic();rows,budgets=read_population(args.inputs,args.budgets)
    prior,identity=prior_records(args.processor_run)
    for name,pin in identity['processor_assets'].items():
        if file_hash(args.model/name)!=pin:raise ValueError('fixed processor assets changed')
    processor=Qwen3OmniMoeProcessor.from_pretrained(args.model,local_files_only=True)
    if file_hash(inspect.getfile(type(processor)))!=identity['hf_processor_source_sha256']:
        raise ValueError('fixed actual HF processor required')
    expected=prompt_expander(processor,args.registry.read_bytes())
    previous={r['id']:r for r in prior.values()};matches=[];counts=Counter();unique=set();maximum=0
    prompt_hashes=set()
    with (args.output/'input-contexts.jsonl').open('x') as handle:
        for index,(row,budget) in enumerate(zip(rows,budgets,strict=True)):
            prompt,ids=expected(row,budget);sha=hashlib.sha256(prompt.encode()).hexdigest()
            if row['id'] in previous:
                saved=previous[row['id']]
                if (sha!=saved['prompt_sha256'] or len(ids)!=saved['input_tokens']
                        or hashlib.sha256(json.dumps(ids).encode()).hexdigest()!=saved['input_ids_sha256']):
                    raise ValueError('reconstruction differs from actual HF/vLLM processor case')
                matches.append(row['id'])
            record=dict(id=row['id'],tier=row['tier'],input_sha256=digest(row['input']),
                prompt_sha256=sha,input_tokens=len(ids),input_ids_sha256=digest(ids))
            handle.write(json.dumps(record)+'\n')
            prompt_hashes.add(sha);counts[row['tier']]+=1;unique.add(record['input_sha256']);maximum=max(maximum,len(ids))
            if index%1000==0:
                handle.flush()
                atomic_json(args.output/'progress.json',dict(phase='reconstruct',done=index+1,total=len(rows),heartbeat_unix=time.time()))
    if set(matches)!=set(previous):raise ValueError('all four actual processor comparisons required')
    write_once(args.output/'result.json',dict(status='passed',records=len(rows),unique_inputs=len(unique),
        counts_by_tier=dict(counts),prompt_layouts=len(prompt_hashes),max_input_tokens=maximum,
        actual_processor_cases_exact=matches,contexts_sha256=file_hash(args.output/'input-contexts.jsonl'),
        inputs_sha256=INPUT_SHA,budgets_sha256=BUDGET_SHA,elapsed_seconds=time.monotonic()-start,
        source_files={p:file_hash(Path(__file__).resolve().parents[2]/p) for p in (
            'tools/jev/check_audio2tool_acceptance_prompts.py','src/multimodal_jev/audio2tool_acceptance.py',
            'src/multimodal_jev/audio2tool_input.py','src/multimodal_jev/audio2tool_inference.py')},
        model_loaded=False,targets_read=False,
        scope='All native scoring-side prompts/token IDs reconstructed with actual fixed tokenizer and complete budgets; four cases exactly match prior real HF/vLLM processing. No full-population waveform processing, generation, actor acceptance or model score.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('inputs','budgets','model','registry','processor-run','output'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:run(args)
    except BaseException as exc:
        write_once(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    write_once(args.output/'terminal.json',dict(exit_code=0))
