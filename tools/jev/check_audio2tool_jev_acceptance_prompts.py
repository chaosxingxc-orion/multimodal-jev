#!/usr/bin/env python3
"""Real-tokenizer audit of all Jev scoring templates and audio expansion bounds."""
import argparse
from collections import defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.audio2tool_engine import prior_records
from multimodal_jev.audio2tool_inference import digest, write_once
from multimodal_jev.audio2tool_jev_acceptance import prompt_expander
from multimodal_jev.audio2tool_jev_inference import population, layout
from multimodal_jev.audio2tool_jev_engine import CONTEXT_SHA
from multimodal_jev.baseline import atomic_json, file_hash


def run(args):
    from transformers import Qwen3OmniMoeProcessor
    begin=time.monotonic();registry=args.registry.read_bytes()
    rows,bank,budgets,profiles=population(args.release,args.context_run,registry,args.inputs,args.budgets)
    _,identity=prior_records(args.native_processor_run)
    for name,pin in identity['processor_assets'].items():
        if file_hash(args.model/name)!=pin:
            raise ValueError('fixed processor assets required')
    processor=Qwen3OmniMoeProcessor.from_pretrained(args.model,local_files_only=True)
    if file_hash(inspect.getfile(type(processor)))!=identity['hf_processor_source_sha256']:
        raise ValueError('fixed real HF processor source required')
    expected=prompt_expander(processor,bank,registry,profiles);groups=defaultdict(list)
    for i,r in enumerate(rows):groups[layout(r['input']['context'])].append(i)
    totals=0;maximum=0;checked=0
    with (args.output/'profiles.jsonl').open('x') as handle:
        for number,(key,indices) in enumerate(groups.items()):
            low=min(indices,key=lambda i:sum(budgets[i]['audio_tokens']))
            high=max(indices,key=lambda i:sum(budgets[i]['audio_tokens']))
            audio_sum=sum(sum(budgets[i]['audio_tokens']) for i in indices)
            for d in bank['decisions']:
                if checked%100==0:
                    handle.flush();atomic_json(args.output/'progress.json',dict(phase='profile',done=checked,total=2742,heartbeat_unix=time.time()))
                checks=[]
                for label,index in [('minimum_actual_audio',low),('maximum_actual_audio',high)]:
                    row,budget=rows[index],budgets[index]
                    prompt,ids=expected(row,budget,d['id'])
                    pieces=prompt.split('<|audio_pad|>')
                    expanded=pieces[0]+''.join('<|audio_pad|>'*n+tail for n,tail in zip(budget['audio_tokens'],pieces[1:],strict=True))
                    actual=processor.tokenizer(expanded,add_special_tokens=False)['input_ids']
                    if actual!=ids:
                        raise ValueError('numeric expansion differs from full-string real tokenization')
                    checks.append(dict(selection=label,id=row['id'],input_tokens=len(ids),input_ids_sha256=digest(ids),
                                       prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
                p=profiles[(key,d['id'])];totals+=len(indices)*p['text_tokens']+audio_sum
                maximum=max(maximum,checks[1]['input_tokens'])
                if checks[1]['input_tokens']!=p['max_input_tokens']:
                    raise ValueError('full profile maximum differs')
                handle.write(json.dumps(dict(layout=number,decision_id=d['id'],records=len(indices),checks=checks))+'\n');checked+=1
    actual_cases=json.loads((args.context_run/'result.json').read_text())['real_processor_cases']
    by_id={r['id']:i for i,r in enumerate(rows)};matches=[]
    for c in actual_cases:
        i=by_id[c['id']];prompt,ids=expected(rows[i],budgets[i],c['decision_id'])
        if (digest(ids)!=c['input_ids_sha256'] or len(ids)!=c['input_tokens']
                or hashlib.sha256(prompt.encode()).hexdigest()!=c['prompt_sha256']):
            raise ValueError('real HF/vLLM audio processor boundary differs')
        matches.append(c['case'])
    if checked!=2742 or len(matches)!=12 or len(groups)!=6 or totals!=98211900808 or maximum!=18168:
        raise ValueError('registered complete context workload differs')
    source=Path(__file__).resolve().parents[2]
    write_once(args.output/'result.json',dict(status='passed',records=len(rows),decisions=len(rows)*457,
        profiles=checked,full_string_tokenizer_cases=checked*2,real_processor_boundaries_exact=matches,
        aggregate_input_tokens=totals,max_input_tokens=maximum,profiles_sha256=file_hash(args.output/'profiles.jsonl'),
        context_result_sha256=CONTEXT_SHA,model_loaded=False,targets_read=False,elapsed_seconds=time.monotonic()-begin,
        sources={name:file_hash(source/name) for name in ('tools/jev/check_audio2tool_jev_acceptance_prompts.py',
            'src/multimodal_jev/audio2tool_jev_acceptance.py','src/multimodal_jev/audio2tool_jev_inference.py',
            'src/multimodal_jev/audio2tool_jev_runtime.py')},
        scope='All 2742 templates: exact full-string/token-level expansion at smallest/largest actual audio budgets, and twelve prior real HF/vLLM boundaries. Full workload arithmetic, not 14M per-decision tokenizations, new waveforms, NPU capacity, actor acceptance or model performance.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('release','context-run','registry','inputs','budgets','model','native-processor-run','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:run(args)
    except BaseException as exc:
        write_once(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    write_once(args.output/'terminal.json',dict(exit_code=0))
