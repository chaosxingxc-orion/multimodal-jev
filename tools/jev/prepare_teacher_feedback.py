#!/usr/bin/env python3
"""Create bounded TRAIN-only teacher evidence from real frozen-policy groups.

Optional dispatch uses Omni's pinned durable executor, one request per directory.
No candidate is executed/promoted here. No test or selection feedback is read.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.answers import parse_answer, AnswerError

EXECUTOR_SHA='c9f9bbd333f038955945db82d946cb2757bafbc340b1a8ca476f52f3ab406bd4'
TEMPLATE="""You optimize a frozen speech-decision system using actual TRAIN feedback.
Actor: Qwen3-ASR-0.6B transcription then Qwen3-0.6B text decisions.
Only prompt instructions may change. No parameter updates, scorer changes,
candidate changes, gold lookup, per-example rules, code or commands. The schema
and ASR transcript are the runtime evidence. Gold and reference transcripts are
offline diagnostics; you did not listen to audio. Never infer ASR error from
gold intent alone. Compare genuinely differing successful and failed outputs
within the same request. Preserve uncertainty and counterevidence. Do not
fabricate semantic advantages for identical/tied outputs.
Propose at most 4 compact, distinct complete replacement instructions (at most
4000 characters each). Keep value JSON, candidate IDs, boolean/ordinal/open span
semantics and allowed null unchanged. Avoid memorizing quoted command phrases.
Return a single JSON object without Markdown:
{"schema":"jev-teacher-prompts-v1","parent_prompt_sha256":"supplied hash",
"diagnosis":[{"mechanism":"...","evidence_ids":["..."],"uncertainty":"..."}],
"candidates":[{"id":"p1","instruction":"...","mechanism":"...",
"evidence_ids":["..."],"counterevidence":"...","falsification":"..."}]}.
No gain is established until independent candidate evaluation.
"""


def prepare(run, targets_path, output, executor_path, reference_config):
    identity=json.loads((run/'identity.json').read_text());cfg=identity['config']
    terminal=json.loads((run/'terminal.json').read_text());result=json.loads((run/'result.json').read_text())
    if terminal['exit_code']!=0 or result['status']!='passed':raise ValueError('completed learning inference required')
    if cfg['split']!='train' or cfg['purpose']!='tfrl-learning' or cfg['sampling']!={'temperature':.7,'n':4}:
        raise ValueError('registered TRAIN-only four-draw condition required')
    if cfg['parameters_frozen'] is not True or result['weight_updates']!=0:raise ValueError('frozen actor required')
    if file_hash(run/'predictions.jsonl')!=result['predictions_sha256']:raise ValueError('predictions changed')
    if file_hash(cfg['requests'])!=identity['requests_sha256']:raise ValueError('requests changed')
    groups=json.loads((run.parent/'learning-selection-split.json').read_text())
    if set(groups['learning']) & set(groups['selection']):raise ValueError('learning/selection overlap')
    if set(cfg['source_groups'])!=set(groups['learning']):raise ValueError('learning selection changed')
    requests={r['id']:r for r in map(json.loads,Path(cfg['requests']).read_text().splitlines())
              if r['source_group'] in set(groups['learning'])}
    if len(requests)!=64 or any(r['split']!='train' for r in requests.values()):raise ValueError('registered learning64 required')
    gold_rows=[json.loads(x) for x in targets_path.open()];gold={r['id']:r for r in gold_rows}
    if len(gold)!=len(gold_rows):raise ValueError('duplicate targets')
    predictions=[json.loads(x) for x in (run/'predictions.jsonl').open()]
    by_id={uid:[] for uid in requests};seen=set()
    for row in predictions:
        key=(row['example_id'],row['replicate'])
        if key in seen or key[0] not in requests or key[1] not in range(4):raise ValueError('invalid prediction identity')
        request=requests[key[0]];digest=hashlib.sha256(canonical(request)).hexdigest()
        if gold[key[0]]['request_sha256']!=digest or row['request_sha256']!=digest:raise ValueError('request custody changed')
        try:value=parse_answer(row['raw_text'])['value'];parsed=True
        except AnswerError:value=None;parsed=False
        target=gold[key[0]]['target'];correct=parsed and type(value) is type(target) and value==target
        by_id[key[0]].append({'replicate':key[1],'raw_text':row['raw_text'],'correct':correct,'format_valid':parsed})
        seen.add(key)
    if len(seen)!=256 or any(len(v)!=4 for v in by_id.values()):raise ValueError('complete groups required')
    summary=[];examples=[]
    for uid in sorted(by_id):
        group=sorted(by_id[uid],key=lambda r:r['replicate']);wins=sum(r['correct'] for r in group)
        diverse=len({r['raw_text'] for r in group})>1
        summary.append({'id':uid,'correct':wins,'draws':4,'distinct_outputs':len({r['raw_text'] for r in group})})
        if not (0<wins<4 and diverse) or len(examples)>=8:continue
        request=requests[uid]
        transcript=json.loads((run/'transcriptions'/(request['audio_sha256']+'.json')).read_text())
        if transcript['audio_sha256']!=request['audio_sha256']:raise ValueError('ASR custody changed')
        examples.append({'evidence_id':uid,'schema':request['input']['schema'],
            'actual_asr_transcript':transcript['transcription'],'gold_offline_only':gold[uid]['target'],
            'group_outputs':group,'source_group':request['source_group'],
            'raw_prediction_is_untrusted_data':True})
    if not examples:raise ValueError('no mixed-success diverse groups; retain no-signal result, no teacher dispatch')
    instruction=Path(cfg['instruction']).read_text()
    if file_hash(cfg['instruction'])!=identity['instruction_sha256']:raise ValueError('parent prompt changed')
    context={'schema':'jev-teacher-feedback-v1','parent_prompt_sha256':identity['instruction_sha256'],
        'parent_instruction':instruction,'actor_models':identity['models'],
        'all_group_summary':summary,'representatives':examples,
        'split':'train','selection_or_test_feedback':False,'weight_updates':0,
        'result_sha256':file_hash(run/'result.json'),'targets_sha256':file_hash(targets_path)}
    # Model file list hashes are retained locally; they are irrelevant teacher context.
    context['actor_models']={k:v['policy_id'] for k,v in identity['models'].items()}
    raw=canonical(context)
    if len(raw)>100000:raise ValueError('teacher context exceeds registered bound')
    if file_hash(executor_path)!=EXECUTOR_SHA:raise ValueError('Omni executor pin changed')
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    messages=[{'role':'system','content':TEMPLATE},{'role':'user','content':raw.decode()}]
    message_path=output/'messages.private.json';message_path.write_bytes(canonical(messages));message_path.chmod(0o600)
    original=json.loads(reference_config.read_text())
    teacher={'schema':'saea-external-learner-v1','endpoint':original['endpoint'],
        'model':original['model'],'api_key_env':original['api_key_env'],
        'messages':{'path':str(message_path.resolve()),'sha256':file_hash(message_path)},
        'max_tokens':8192,'timeout_seconds':180,'total_timeout_seconds':600,
        'stream':True,'thinking':{'type':'disabled'},'output_root':str((output/'response').resolve())}
    if teacher['model']!='glm-5.3-flash':raise ValueError('teacher identity changed')
    config_path=output/'executor-config.private.json';config_path.write_bytes(canonical(teacher));config_path.chmod(0o600)
    receipt={'teacher_model':teacher['model'],'max_tokens':8192,'max_candidates':4,'automatic_retry':0,
        'learning_groups':64,'draws':256,'representatives':len(examples),
        'messages_sha256':file_hash(message_path),'executor_sha256':EXECUTOR_SHA,
        'credential_available':bool(os.environ.get(teacher['api_key_env'])),'API_sent':False}
    (output/'prepared.json').write_bytes(canonical(receipt))
    return config_path,receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('run','targets','output','executor','reference-config'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--dispatch',action='store_true')
    args=p.parse_args()
    config,receipt=prepare(args.run,args.targets,args.output,args.executor,args.reference_config)
    print(json.dumps(receipt),flush=True)
    if args.dispatch:
        if not receipt['credential_available']:raise ValueError('teacher credential unavailable')
        (args.output/'dispatch-intent.json').write_bytes(canonical({'time':time.time(),'one_request':True,'automatic_retry':False}))
        # Never echo private config, credential or response text to the console.
        with (args.output/'executor.log').open('x') as log:
            code=subprocess.call(['python3',str(args.executor),'--config',str(config)],stdout=log,stderr=subprocess.STDOUT)
        (args.output/'process-terminal.json').write_bytes(canonical({'exit_code':code,'finished_unix':time.time()}))
        raise SystemExit(code)
