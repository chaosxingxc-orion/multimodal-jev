"""Source-only selection to frozen FSC transfer, using shared existing tools.

Default validates label-free identities only. Explicit launch invokes the
existing lease/exact-wait launcher. Failure preserves outputs without retries.
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
from tools.jev.prepare_fsc_context_package import PREFIX, CACHE
from tools.jev.run_context_baseline_sequence import check_pins


def read(path):return json.loads(Path(path).read_text())


def write(path, data):
    with Path(path).open('x') as stream:json.dump(data,stream,indent=2,allow_nan=False)


def validate(spec):
    check_pins(spec['pins'])
    required={'population','task','input_package_manifest','input_audit','input_receipts','requests'}
    if set(spec['artifacts']) != required:raise ValueError('fixed complete target artifacts required')
    for entry in spec['artifacts'].values():
        if set(entry) != {'path','sha256'} or sha(entry['path']) != entry['sha256']:
            raise ValueError('target artifact identity differs')
    a=spec['artifacts'];manifest=read(a['input_package_manifest']['path'])
    proof=read(a['input_audit']['path'])
    if (manifest['recordings'] != 3793 or manifest['task_sha256'] != a['task']['sha256']
            or manifest['population_sha256'] != a['population']['sha256']
            or manifest['files']['requests.jsonl'] != a['requests']['sha256']
            or proof['manifest']['request_manifest_sha256'] != a['input_package_manifest']['sha256']
            or proof['independent_receipts_sha256'] != a['input_receipts']['sha256']):
        raise ValueError('frozen target input population/definitions differ')
    if not re.fullmatch(re.escape(PREFIX)+r'fsc-context-pair-[a-z0-9-]+',spec['remote_root']):
        raise ValueError('isolated transfer remote directory required')
    return {k:v for k,v in a.items() if k != 'requests'}


def invoke(module, arguments, log):
    with Path(log).open('x') as stream:
        subprocess.run([sys.executable,'-m',('tools.reports.prepare_fsc_arithmetic_source' if module == 'prepare_fsc_context_source_v2' else 'tools.jev.'+module),*map(str,arguments)],
                       stdout=stream,stderr=subprocess.STDOUT,check=True)


def run(spec, output, *, launch=False):
    frozen=validate(spec)
    if not launch:return dict(status='validated_label_free_plan_no_dispatch',scoring_contract=frozen)
    output=Path(output).resolve();source_run=Path(spec['source_experiment']).resolve()
    for path in [source_run,*[Path(v['path']).resolve() for v in spec['artifacts'].values()]]:
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('independent output required')
    output.mkdir(exist_ok=False)
    owner=(output/'owner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(output/'input.json',spec)
    contract=output/'scoring-contract.json';write(contract,frozen)
    def status(phase,**extra):
        p=output/'status.tmp';p.write_text(json.dumps(dict(phase=phase,unix=time.time(),**extra),indent=2))
        p.replace(output/'status.json')
    def command(module,args,phase):
        validate(spec);status(phase);invoke(module,args,output/(phase+'.log'))
    try:
        source=output/'source-package';a=spec['artifacts']
        command('prepare_fsc_context_source_v2',['--experiment-directory',source_run,
            '--input-audit',a['input_audit']['path'],'--input-audit-sha256',a['input_audit']['sha256'],
            '--output',source],'source')
        inputs=dict(input_manifest=a['input_package_manifest'],input_audit=a['input_audit'],
                    requests=a['requests']['path'],input_receipts=a['input_receipts']['path'],cache_root=CACHE)
        request=dict(source_package=str(source),source_manifest_sha256=sha(source/'PACKAGE.json'),inputs=inputs)
        write(output/'package-spec.json',request);package=output/'runtime-package'
        command('prepare_fsc_context_package',['--spec',output/'package-spec.json','--output',package,
            '--remote-root',spec['remote_root']],'package')
        digest=sha(package/'PACKAGE.json')
        command('launch_fsc_context_pair',['--package',package,'--manifest-sha256',digest,'--launch'],'inference')
        launch_path=Path('.local/decision-objective-research')/(Path(spec['remote_root']).name+'-launch.json')
        collection=output/'collected';proof=output/'inference-audit.json'
        command('collect_fsc_context_pair',['--package',package,'--manifest-sha256',digest,
            '--launch',launch_path,'--output',collection,'--report',proof],'collect')
        scoring=dict(package=str(package),digest=digest,launch=str(launch_path),collection=str(collection),
            observed=str(collection.with_suffix('.inspect.json')),
            transport=str(collection.with_suffix('.transport.json')),inference_report=str(proof),
            scoring_contract=dict(path=str(contract),sha256=sha(contract)))
        write(output/'score-spec.json',scoring)
        command('score_fsc_context_pair',['--spec',output/'score-spec.json','--output',output/'scores.json'],'score')
        result=read(output/'scores.json')
        if result.get('runtime_provenance_verified') is not True:
            raise ValueError('independent runtime provenance required')
        status('frozen_transfer_collected_and_scored',scores_sha256=sha(output/'scores.json'),target_tuning=False)
        return read(output/'status.json')
    except BaseException as exc:
        status('stopped_preserve_handles',error=repr(exc),restart=False)
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True)
    p.add_argument('--output',required=True);p.add_argument('--launch',action='store_true')
    a=p.parse_args();print(json.dumps(run(read(a.spec),a.output,launch=a.launch),indent=2))
