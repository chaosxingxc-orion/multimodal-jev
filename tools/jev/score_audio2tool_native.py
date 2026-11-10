#!/usr/bin/env python3
"""Inspect full actor container and ledger before opening Audio2Tool targets."""
import argparse
import inspect
import json
from pathlib import Path
import shlex
import subprocess

from multimodal_jev.audio2tool_acceptance import HEADERS_SHA,TARGETS_SHA,audit_ledger,check_container,prompt_expander
from multimodal_jev.audio2tool_engine import ASSET_AUDIT_SHA,ENGINE,VERSIONS,prior_records
from multimodal_jev.audio2tool_inference import INPUT_SHA,BUDGET_SHA,read_population,require_capacity,write_once
from multimodal_jev.audio2tool_score import evaluate_projected,load_official
from multimodal_jev.baseline import file_hash


def inspect_container(cid):
    import re
    if not re.fullmatch('[a-f0-9]{64}',cid):raise ValueError('exact container ID required')
    return json.loads(subprocess.check_output(['ssh','-o','ConnectTimeout=8','npu',
        shlex.join(['docker','inspect',cid])],text=True,timeout=30))[0]


def accept(args):
    if file_hash(args.plan)!=args.plan_hash:raise ValueError('registered full launch plan changed')
    plan=json.loads(args.plan.read_text());item=inspect_container(args.container_id)
    check_container(item,plan,args.container_id)
    write_once(args.output/'actor-inspection.json',item)
    source=Path(plan['source_root']);argv=plan['argv']
    runner=str(source/'tools/jev/infer_audio2tool_native.py')
    command=argv[argv.index(plan['image'])+1:]
    if command[:6]!=['timeout','--signal=TERM','--kill-after=30s','86400s','python3','-B'] or command[6]!=runner:
        raise ValueError('registered bounded full runner command required')
    keys=('inputs','budgets','output','root','model','registry','asset-audit','processor-run',
          'capacity-root','capacity-inspection','capacity-inspection-hash','capacity-container-id')
    opts=command[7:]
    if len(opts)!=2*len(keys) or set(opts[::2])!={'--'+k for k in keys}:
        raise ValueError('explicit full inference arguments required')
    cfg=dict(zip((k[2:] for k in opts[::2]),opts[1::2],strict=True));run=Path(cfg['output'])
    expected_sources={str(p.relative_to(source)) for p in (source/'src/multimodal_jev').glob('*.py')}
    expected_sources.add('tools/jev/infer_audio2tool_native.py')
    if set(plan['source_files'])!=expected_sources:raise ValueError('complete actor source manifest required')
    for name,pin in plan['source_files'].items():
        if file_hash(source/name)!=pin:raise ValueError('actor source changed')
    result=json.loads((run/'result.json').read_text());identity=json.loads((run/'identity.json').read_text())
    if (json.loads((run/'terminal.json').read_text())['exit_code']!=0 or result['status']!='passed'
            or result['records']!=30733 or result['weight_updates']!=0 or result['targets_read'] is not False
            or result['model_files_unchanged'] is not True or result['identity_sha256']!=file_hash(run/'identity.json')
            or identity['source_files']!=plan['source_files'] or identity['versions']!=VERSIONS
            or identity['engine']!=ENGINE or identity['inputs_sha256']!=INPUT_SHA or identity['budgets_sha256']!=BUDGET_SHA
            or identity['sampling']!=dict(temperature=0.,max_tokens=1024,n=1,seed=17)
            or identity['duplicate_policy']!='execute_each_release_record_independently'
            or identity['parameters_frozen'] is not True or identity['targets_read'] is not False
            or file_hash(cfg['asset-audit'])!=ASSET_AUDIT_SHA or identity['asset_audit_sha256']!=ASSET_AUDIT_SHA):
        raise ValueError('complete frozen actor identity/result required')
    assets=json.loads(Path(cfg['asset-audit']).read_text())
    if identity['model']!=cfg['model'] or cfg['model']!=assets['model'] or identity['model_files']!=assets['files']:
        raise ValueError('registered original model identity required')
    capacity_item=inspect_container(cfg['capacity-container-id'])
    cp=args.output/'capacity-inspection.json';write_once(cp,capacity_item)
    current=require_capacity(cfg['capacity-root'],cp,file_hash(cp),cfg['capacity-container-id'],cfg['processor-run'])
    # Original actor admission must bind its recorded inspection; current live
    # inspect independently revalidates the same container/result.
    recorded=require_capacity(cfg['capacity-root'],cfg['capacity-inspection'],cfg['capacity-inspection-hash'],
                              cfg['capacity-container-id'],cfg['processor-run'])
    if identity['capacity']!=recorded or current['result_sha256']!=recorded['result_sha256']:
        raise ValueError('capacity predecessor changed')
    rows,budgets=read_population(cfg['inputs'],cfg['budgets'])
    if file_hash(args.headers)!=HEADERS_SHA:raise ValueError('pinned full waveform header audit required')
    headers={r['path']:r for r in map(json.loads,args.headers.open())}
    prediction_path=run/'ledger/predictions.jsonl'
    if file_hash(prediction_path)!=result['predictions_sha256']:raise ValueError('predictions changed')
    predictions=[json.loads(l) for l in prediction_path.open()]
    from transformers import Qwen3OmniMoeProcessor
    _,pidentity=prior_records(cfg['processor-run'])
    for name,pin in pidentity['processor_assets'].items():
        if file_hash(Path(cfg['model'])/name)!=pin:raise ValueError('tokenizer assets changed')
    processor=Qwen3OmniMoeProcessor.from_pretrained(cfg['model'],local_files_only=True)
    if file_hash(inspect.getfile(type(processor)))!=pidentity['hf_processor_source_sha256']:
        raise ValueError('accepted HF processor source required for independent rendering')
    expected=prompt_expander(processor,Path(cfg['registry']).read_bytes())
    summary=audit_ledger(run/'ledger',rows,budgets,predictions,headers,expected)
    if summary['records']!=result['records'] or summary['length_limited']!=result['length_limited']:
        raise ValueError('ledger summary differs from actor terminal')
    # Reinspect exact actor after potentially lengthy ledger work; no replay.
    check_container(inspect_container(args.container_id),plan,args.container_id)
    receipt=dict(container_id=args.container_id,launch_plan_sha256=args.plan_hash,
        result_sha256=file_hash(run/'result.json'),predictions_sha256=file_hash(prediction_path),
        ledger=summary,targets_read=False,
        verifier_source_files={p:file_hash(Path(__file__).resolve().parents[2]/p) for p in (
            'tools/jev/score_audio2tool_native.py','src/multimodal_jev/audio2tool_acceptance.py',
            'src/multimodal_jev/audio2tool_inference.py','src/multimodal_jev/audio2tool_input.py',
            'src/multimodal_jev/audio2tool_dialogue_input.py','src/multimodal_jev/audio2tool_inventory.py')},
        scope='Successful fixed actor container and full release ledger independently checked; prompts/tokens regenerated and header conversion arithmetic checked. Source/byte-check provenance, not full waveform recomputation.')
    write_once(args.output/'acceptance.json',receipt)
    return predictions


def score(args):
    predictions=accept(args)
    # First target access occurs only after successful runtime acceptance.
    if file_hash(args.targets)!=TARGETS_SHA:raise ValueError('registered scorer-only targets changed')
    targets=[json.loads(l) for l in args.targets.open()]
    report,details=evaluate_projected(targets,predictions,load_official(args.official_metrics))
    write_once(args.output/'metrics.json',report)
    write_once(args.output/'score-receipt.json',dict(targets_sha256=TARGETS_SHA,
        acceptance_sha256=file_hash(args.output/'acceptance.json'),metrics_sha256=file_hash(args.output/'metrics.json'),
        official_metrics_sha256=file_hash(args.official_metrics),runner_sha256=file_hash(__file__),
        scope='Local native JSON protocol with fixed official primitives and complete denominators; not historical paper reproduction.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('plan','headers','targets','official-metrics','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('plan-hash','container-id'):p.add_argument('--'+k,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        score(args)
    except BaseException as exc:
        write_once(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    write_once(args.output/'terminal.json',dict(exit_code=0))
