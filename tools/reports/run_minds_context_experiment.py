"""Frozen MINDS pair: package, parity/full inference, collect, then score.

Caller must supply the source-selected pair; no target checkpoint selection.
All outputs are immutable and any failure preserves original handles.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from tools.jev.context_rl_evaluation_plan import sha
from tools.jev.run_context_baseline_sequence import check_pins
from tools.reports.prepare_minds_context_package import PREFIX


def read(p):return json.loads(Path(p).read_text())


def write(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def validate(spec):
    check_pins(spec['pins'])
    contract=spec['scoring_contract']
    if sha(contract['path'])!=contract['sha256']:raise ValueError('predeclared scoring contract differs')
    c=read(contract['path'])
    if (c['schema']!='minds-native-descriptor-scoring-v1' or c['threshold']!=.5
            or c['target_tuning'] is not False):raise ValueError('frozen MINDS-only contract required')
    # Read identity metadata only; no target annotation lookup before inference.
    a=c['artifacts'];inputs=Path(spec['inputs'])
    for key,name in [(n.replace('.','_'),n) for n in ('MANIFEST.json','definitions.json','inputs.jsonl','conversion.jsonl')]:
        if sha(inputs/name)!=a[key]['sha256']:raise ValueError('predeclared MINDS inputs differ')
    for key in ('input_config',):
        if spec['pins'].get(spec[key])!=sha(spec[key]):raise ValueError('pinned processor audit required')
    for name in ('MANIFEST.json','inputs.jsonl'):
        p=Path(spec['processor_receipts_root'])/name
        if spec['pins'].get(str(p))!=sha(p):raise ValueError('pinned processor receipts required')
    if not re.fullmatch(re.escape(PREFIX)+r'minds-context-pair-[a-z0-9-]+',spec['remote_root']):
        raise ValueError('isolated MINDS runtime required')
    return c


def invoke(module,args,log):
    with Path(log).open('x') as stream:
        subprocess.run([sys.executable,'-m','tools.reports.'+module,*map(str,args)],
            stdout=stream,stderr=subprocess.STDOUT,check=True)


def run(spec,output,*,launch=False):
    validate(spec)
    if not launch:return dict(status='validated_MINDS_plan_no_dispatch',target_tuning=False)
    from tools.jev.prepare_fsc_context_source import verify as verify_source
    source=Path(spec['source_package']);source_digest=sha(source/'PACKAGE.json')
    verify_source(source,source_digest)
    output=Path(output).resolve()
    for p in (source,Path(spec['inputs']),Path(spec['processor_receipts_root'])):
        if output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output):raise ValueError('independent output required')
    output.mkdir(exist_ok=False)
    owner=(output/'owner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(output/'input.json',spec)
    def status(phase,**kw):
        p=output/'status.tmp';p.write_text(json.dumps(dict(phase=phase,unix=time.time(),**kw),indent=2)+'\n')
        p.replace(output/'status.json')
    def command(module,args,phase):
        validate(spec);status(phase);invoke(module,args,output/(phase+'.log'))
    try:
        package=output/'runtime-package'
        request={k:spec[k] for k in ('source_package','inputs','processor_receipts_root','input_config')}
        request['source_manifest_sha256']=source_digest
        write(output/'package-spec.json',request)
        command('prepare_minds_context_package',['--spec',output/'package-spec.json','--output',package,
            '--remote-root',spec['remote_root']],'package')
        digest=sha(package/'PACKAGE.json')
        command('launch_minds_context_pair',['--package',package,'--manifest-sha256',digest,'--launch'],'inference')
        launch_path=Path('.local/decision-objective-research')/(Path(spec['remote_root']).name+'-launch.json')
        collection=output/'collected';proof=output/'inference-audit.json'
        command('collect_minds_context_pair',['--package',package,'--manifest-sha256',digest,
            '--launch',launch_path,'--output',collection,'--report',proof],'collect')
        scoring=dict(package=str(package),digest=digest,collection=str(collection),launch=str(launch_path),
            observed=str(collection.with_suffix('.inspect.json')),transport=str(collection.with_suffix('.transport.json')),
            inference_report=str(proof),scoring_contract=spec['scoring_contract'])
        write(output/'score-spec.json',scoring)
        command('score_minds_context_pair',['--spec',output/'score-spec.json','--output',output/'scores.json'],'score')
        result=read(output/'scores.json')
        if result.get('recordings')!=8168 or result.get('decisions')!=14 or result.get('target_tuning') is not False:
            raise ValueError('full independently scored MINDS required')
        status('frozen_MINDS_collected_and_scored',scores_sha256=sha(output/'scores.json'),target_tuning=False)
        return read(output/'status.json')
    except BaseException as exc:
        status('stopped_preserve_handles',error=repr(exc),restart=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True)
    p.add_argument('--launch',action='store_true');a=p.parse_args()
    print(json.dumps(run(read(a.spec),a.output,launch=a.launch),indent=2))
