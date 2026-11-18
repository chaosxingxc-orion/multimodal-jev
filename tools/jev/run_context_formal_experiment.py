"""Run one published RL experiment through collection and all 11 eval points.

No implicit retries or new handles. Existing launchers own device leases and
exact waits. A new local output directory records every command/spec/phase.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from tools.jev.prepare_context_formal_package import verify, sha, PREFIX
from tools.jev.prepare_context_stage2_source import verify as verify_source


def read(path):return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def plan(spec):
    package=Path(spec['package']);m=verify(package,spec['manifest_sha256'])
    c=read(package/'config.json');release=read(package/'RELEASE.json')
    initial=Path(spec['initial_package'])
    sm=verify_source(initial,release['source_package_manifest_sha256'])
    if sm['source_policy_id'] != c['source_policy_id']:
        raise ValueError('initial source differs from formal package')
    experiment=read(spec['experiment'])
    if (experiment != read(package/'evidence/experiment.json')
            or experiment['evaluation']['checkpoints'] != [0,*range(64,634,64),634]):
        raise ValueError('same frozen experiment and complete 11-point evaluation required')
    if read(spec['stage1_proof']) != read(package/'evidence/training_proof.json'):
        raise ValueError('common full Stage1 collection required')
    # Full artifact validation is repeated by the actual training/eval auditors.
    if not (Path(spec['stage1_root'])/'run/policy-0634/MANIFEST.json').is_file():
        raise ValueError('collected full Stage1 source missing')
    if read(spec['template'])['release_sha256'] != experiment['release_sha256']:
        raise ValueError('evaluation template data differs')
    schedule=package/'source/schedule'
    if sha(schedule/'MANIFEST.json') != c['schedule_sha256']:
        raise ValueError('scheduled learning batches changed')
    tag=Path(m['remote_root']).name
    if not tag.startswith('context-rl-'):raise ValueError('formal run tag required')
    return dict(run_id=tag,algorithm=c['algorithm'],training_package=str(package.resolve()),
        training_manifest_sha256=spec['manifest_sha256'],
        training_launch=str(Path('.local/decision-objective-research')/(tag+'-launch.json')),
        initial_policy=str((initial/'policy').resolve()),schedule=str(schedule.resolve()),
        evaluation_steps=experiment['evaluation']['checkpoints'],
        source_policy_id=c['source_policy_id'],status='planned_only_no_dispatch')


def invoke(module, arguments, log):
    with log.open('x') as stream:
        subprocess.run([sys.executable,'-m','tools.jev.'+module,*map(str,arguments)],
                       stdout=stream,stderr=subprocess.STDOUT,check=True)


def run(spec, output, *, launch=False):
    p=plan(spec)
    if not launch:return p
    output=Path(output).resolve()
    for root in (Path(spec['package']),Path(spec['initial_package']),Path(spec['stage1_root'])):
        if output.is_relative_to(root.resolve()) or root.resolve().is_relative_to(output):
            raise ValueError('independent experiment output required')
    output.mkdir(exist_ok=False)
    owner=(output/'owner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(output/'plan.json',p);write(output/'input.json',spec)
    def status(phase,**kw):
        temp=output/'status.tmp';temp.write_text(json.dumps(dict(phase=phase,unix=time.time(),**kw),indent=2)+'\n')
        temp.replace(output/'status.json')
    def command(module,args,name):
        status(name);invoke(module,args,output/(name+'.log'))
    try:
        command('launch_context_formal_training',['--package',p['training_package'],
            '--manifest-sha256',p['training_manifest_sha256'],'--launch'],'train')
        # Launcher returns on exact terminal, including nonzero exits. Collect
        # failure artifacts first; the independent audit then rejects them.
        training=output/'training-collected'
        command('collect_context_rl_training',['--launch',p['training_launch'],'--output',training],'collect-training')
        audit_spec=dict(collection=str(training),transport=str(training.with_suffix('.transport.json')),
            launch=p['training_launch'],experiment=spec['experiment'],schedule=p['schedule'],initial_policy=p['initial_policy'])
        write(output/'training-audit-spec.json',audit_spec)
        proof=output/'training-audit.json'
        command('audit_context_rl_training',['--spec',output/'training-audit-spec.json','--output',proof],'audit-training')
        common=dict(experiment=spec['experiment'],stage1_root=spec['stage1_root'],stage1_proof=spec['stage1_proof'],
            rl_root=str(training),rl_proof=str(proof),run_id=p['run_id'],algorithm=p['algorithm'])
        points=[]
        for step in p['evaluation_steps']:
            stem=f'step-{step:04d}';point=output/stem;point.mkdir()
            request=dict(common,template=spec['template'],step=step)
            write(point/'prepare-spec.json',request)
            package=point/'package';remote=PREFIX+'context-rl-eval-'+p['run_id'].removeprefix('context-rl-')+f'-step{step:04d}'
            command('prepare_context_rl_evaluation_package',['--spec',point/'prepare-spec.json',
                '--output',package,'--remote-root',remote],stem+'-prepare')
            command('launch_context_rl_evaluation',['--package',package,
                '--manifest-sha256',sha(package/'PACKAGE.json'),'--launch'],stem+'-launch')
            launch_path=Path('.local/decision-objective-research')/(Path(remote).name+'-launch.json')
            collection=point/'collected'
            command('collect_context_rl_selection',['--launch',launch_path,'--output',collection],stem+'-collect')
            entry=dict(step=step,collection=str(collection),transport=str(collection.with_suffix('.transport.json')),
                       expected_config_sha256=read(launch_path)['config_sha256'])
            write(point/'audit-spec.json',dict(common,**entry))
            command('audit_context_rl_selection',['--spec',point/'audit-spec.json','--output',point/'audit.json'],stem+'-audit')
            points.append(entry)
        write(output/'curve-spec.json',dict(common,evaluations=points))
        command('audit_context_rl_curve',['--spec',output/'curve-spec.json','--output',output/'curve.json'],'audit-curve')
        status('complete_training_and_selection_curve',run_id=p['run_id'],points=len(points),
               scope='One RL algorithm curve only; TEST, comparison and transfer remain separate')
        return read(output/'curve.json')
    except BaseException as exc:
        status('stopped_preserve_handles',error=repr(exc),restart=False)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--launch',action='store_true')
    a=parser.parse_args();print(json.dumps(run(read(a.spec),a.output,launch=a.launch),indent=2))
