#!/usr/bin/env python3
"""Inspect exact full Jev actor and raw ledger before accessing scorer targets."""
import argparse
import inspect
import json
from pathlib import Path
import re
import shlex
import subprocess
from types import SimpleNamespace

from multimodal_jev.audio2tool_acceptance import HEADERS_SHA, check_container
from multimodal_jev.audio2tool_engine import ASSET_AUDIT_SHA, ENGINE, VERSIONS, prior_records
from multimodal_jev.audio2tool_inference import INPUT_SHA, BUDGET_SHA, write_once
from multimodal_jev.audio2tool_jev_acceptance import audit_ledger, prompt_expander
from multimodal_jev.audio2tool_jev_capacity import MODEL, require_capacity
from multimodal_jev.audio2tool_jev_engine import CONTEXT_SHA, MANIFEST_SHA
from multimodal_jev.audio2tool_jev_inference import population
from multimodal_jev.baseline import file_hash
from score_audio2tool_jev import run as score_raw


def inspect_container(cid):
    if not re.fullmatch('[a-f0-9]{64}', cid):
        raise ValueError('exact container ID required')
    items=json.loads(subprocess.check_output(['ssh','-o','ConnectTimeout=8','npu',
        shlex.join(['docker','inspect',cid])],timeout=30))
    if len(items)!=1 or items[0]['Id']!=cid:
        raise ValueError('exact single live container inspection required')
    return items[0]


def configuration(item, plan, cid):
    check_container(item, plan, cid)
    source=Path(plan['source_root']);command=item['Config']['Cmd'];wall=plan['wall_seconds']
    if type(wall) is not int or wall<=0 or command[:7]!=[
        'timeout','--signal=TERM','--kill-after=30s',str(wall)+'s','python3','-B',
        str(source/'tools/jev/infer_audio2tool_jev.py')]:
        raise ValueError('explicit registered wall budget and full Jev runner required')
    keys={'release','context-run','registry','inputs','budgets','output','root','model','asset-audit',
          'native-processor-run','capacity-root','capacity-inspection','capacity-intent-hash',
          'capacity-inspection-hash','capacity-container-id'}
    opts=command[7:]
    if len(opts)!=2*len(keys) or set(opts[::2])!={'--'+k for k in keys}:
        raise ValueError('exact explicit full inference arguments required')
    cfg=dict(zip((k[2:] for k in opts[::2]),opts[1::2],strict=True));run=Path(cfg['output'])
    if not source.is_absolute() or not run.is_absolute():
        raise ValueError('absolute source and output paths required')
    binds=set(item['HostConfig']['Binds'])
    if f'{source}:{source}:ro' not in binds or f'{run.parent}:{run.parent}:rw' not in binds:
        raise ValueError('host-visible immutable source and experiment output mounts required')
    expected={str(p.relative_to(source)) for p in (source/'src/multimodal_jev').glob('*.py')}
    expected.add('tools/jev/infer_audio2tool_jev.py')
    if set(plan['source_files'])!=expected:
        raise ValueError('complete actor Python source manifest required')
    for name,pin in plan['source_files'].items():
        if file_hash(source/name)!=pin:
            raise ValueError('actor source snapshot changed')
    return cfg


def accept(args):
    if file_hash(args.plan)!=args.plan_hash:
        raise ValueError('pre-dispatch pinned actor plan required')
    plan=json.loads(args.plan.read_text());item=inspect_container(args.container_id)
    cfg=configuration(item,plan,args.container_id)
    write_once(args.output/'actor-inspection.json',item)
    run=Path(cfg['output']);result_pin=file_hash(run/'result.json')
    result=json.loads((run/'result.json').read_text());identity=json.loads((run/'identity.json').read_text())
    if (json.loads((run/'terminal.json').read_text())!={'exit_code':0} or result['status']!='passed'
            or result['identity_sha256']!=file_hash(run/'identity.json') or result['weight_updates']!=0
            or result['targets_read'] is not False or result['model_files_unchanged'] is not True
            or identity['sources']!=plan['source_files'] or identity['versions']!=VERSIONS or identity['engine']!=ENGINE
            or identity['sampling']!=dict(temperature=0.,max_tokens=1024,n=1,seed=17)
            or identity['manifest_sha256']!=MANIFEST_SHA or identity['context_sha256']!=CONTEXT_SHA
            or identity['inputs_sha256']!=INPUT_SHA or identity['budgets_sha256']!=BUDGET_SHA
            or identity['model']!=MODEL or cfg['model']!=MODEL
            or identity['asset_audit_sha256']!=ASSET_AUDIT_SHA or file_hash(cfg['asset-audit'])!=ASSET_AUDIT_SHA
            or identity['records']!=30733 or identity['decisions']!=14044981 or identity['shard_decisions']!=4096
            or identity['parameters_frozen'] is not True or identity['targets_read'] is not False
            or identity['duplicate_policy']!='independent_generation_for_every_release_record_and_decision'):
        raise ValueError('complete frozen actor identity and terminal required')
    assets=json.loads(Path(cfg['asset-audit']).read_text())
    if assets['model']!=MODEL or identity['model_files']!=assets['files']:
        raise ValueError('original complete model identity required')
    registry=Path(cfg['registry']).read_bytes()
    capacity_inspection=args.output/'capacity-inspection.json'
    write_once(capacity_inspection,inspect_container(cfg['capacity-container-id']))
    current=require_capacity(cfg['capacity-root'],cfg['capacity-intent-hash'],capacity_inspection,
        file_hash(capacity_inspection),cfg['capacity-container-id'],cfg['context-run'],cfg['release'],registry)
    recorded=require_capacity(cfg['capacity-root'],cfg['capacity-intent-hash'],cfg['capacity-inspection'],
        cfg['capacity-inspection-hash'],cfg['capacity-container-id'],cfg['context-run'],cfg['release'],registry)
    if identity['capacity']!=recorded or current['result_sha256']!=recorded['result_sha256']:
        raise ValueError('capacity predecessor changed')
    rows,bank,budgets,profiles=population(cfg['release'],cfg['context-run'],registry,cfg['inputs'],cfg['budgets'])
    if file_hash(args.headers)!=HEADERS_SHA:
        raise ValueError('independent pinned full audio header audit required')
    headers={r['path']:r for r in map(json.loads,args.headers.open())}
    prediction=run/'predictions.jsonl';pin=file_hash(prediction)
    complete=json.loads((run/'ledger/complete.json').read_text())
    if (result['ledger']!=complete or result['projection']!=dict(decisions=14044981,
            predictions_sha256=pin,actor_execution_verified=False)
            or json.loads((run/'ledger/manifest.json').read_text())['shard_decisions']!=4096):
        raise ValueError('complete unmodified raw ledger projection required')
    _,pidentity=prior_records(cfg['native-processor-run'])
    for name,sha in pidentity['processor_assets'].items():
        if file_hash(Path(MODEL)/name)!=sha:
            raise ValueError('tokenizer assets changed')
    from transformers import Qwen3OmniMoeProcessor
    processor=Qwen3OmniMoeProcessor.from_pretrained(MODEL,local_files_only=True)
    if file_hash(inspect.getfile(type(processor)))!=pidentity['hf_processor_source_sha256']:
        raise ValueError('fixed HF processor implementation required')
    summary=audit_ledger(run/'ledger',rows,bank,budgets,prediction,headers,
        prompt_expander(processor,bank,registry,profiles),
        lambda tokens:processor.tokenizer.decode(tokens,skip_special_tokens=True))
    if (summary['decisions']!=14044981 or summary['length_limited']!=complete['length_limited']
            or summary['predictions_sha256']!=pin or file_hash(run/'result.json')!=result_pin):
        raise ValueError('ledger and successful actor summary differ')
    configuration(inspect_container(args.container_id),plan,args.container_id)
    if file_hash(args.plan)!=args.plan_hash:
        raise ValueError('actor plan changed during acceptance')
    verifier=Path(__file__).resolve().parents[2]
    receipt=dict(container_id=args.container_id,plan_sha256=args.plan_hash,result_sha256=result_pin,
        identity_sha256=file_hash(run/'identity.json'),predictions_sha256=pin,ledger=summary,
        source_sha256={name:file_hash(verifier/name) for name in (
            'tools/jev/accept_and_score_audio2tool_jev.py','tools/jev/score_audio2tool_jev.py',
            'src/multimodal_jev/audio2tool_jev_acceptance.py','src/multimodal_jev/audio2tool_jev_ledger.py',
            'src/multimodal_jev/audio2tool_jev_capacity.py','src/multimodal_jev/audio2tool_jev_inference.py',
            'src/multimodal_jev/audio2tool_jev_runtime.py','src/multimodal_jev/audio2tool_acceptance.py')},
        targets_read=False,
        scope='Exact actor and capacity containers plus complete raw ledger/audio receipt arithmetic/prompt/output-token validation. Source-bound waveform verification, not full waveform recomputation; no score claims until targets are evaluated.')
    write_once(args.output/'acceptance.json',receipt)
    return cfg,prediction


def score(args):
    cfg,predictions=accept(args)
    # No target path or hash is read before actual actor acceptance succeeds.
    report=score_raw(SimpleNamespace(release=Path(cfg['release']),registry=Path(cfg['registry']),
        official_metrics=args.official_metrics,predictions=predictions))
    receipt=json.loads((args.output/'acceptance.json').read_text())
    if report['predictions_sha256']!=receipt['predictions_sha256']:
        raise ValueError('predictions changed after acceptance')
    report['actor_execution_verified']=True
    report['limitation']='Pinned full independent Jev actor acceptance plus local task metrics; not unchanged paper protocol, RL improvement or cross-task generalization evidence.'
    report['acceptance_sha256']=file_hash(args.output/'acceptance.json')
    write_once(args.output/'metrics.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('plan','headers','official-metrics','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    for key in ('plan-hash','container-id'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        score(args)
    except BaseException as exc:
        write_once(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    write_once(args.output/'terminal.json',dict(exit_code=0))
