"""Seal/launch finite DP8 InfoNCE using the shared exact-container lifecycle."""
import argparse
import json
from pathlib import Path
import shutil
import re
import subprocess
import sys
import time
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.context_topology_batch_plan import derive
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from multimodal_jev.context_replica_schedule import load_schedule
from multimodal_jev.slurp_portable_release import load as load_release
from tools.reports.context_infonce_trainer import generate,topology
from tools.reports import context_replica_training_operations as lifecycle


LEGACY_RUNNER = '''import json,subprocess,sys,time
from pathlib import Path
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import require_training_storage
from context_process_supervisor import stop_owned
r=Path(__file__).resolve().parent
require_training_storage(r)
c=json.loads((r/'config.json').read_text())
out=r/'supervision';out.mkdir(exist_ok=False)
argv=[sys.executable,'-m','torch.distributed.run','--standalone',
      '--nproc-per-node='+str(len(c['physical_devices'])),
      str(r/'source/tools/reports/context_infonce_trainer.py'),
      '--config',str(r/'config.json'),'--output',str(r/'run')]
begin=time.time();process=None;code=1
try:
 with (out/'actor.log').open('x') as log:
  process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  commit_json(out/'started.json',dict(pid=process.pid,argv=argv,started_unix=begin))
  code=process.wait()
finally:
 if process is not None and process.poll() is None:stop_owned(process,30)
 commit_json(out/'terminal.json',dict(exit_code=code,started_unix=begin,finished_unix=time.time()))
raise SystemExit(code)
'''

# --standalone advertises getfqdn() in this torch build; network=none cannot
# resolve the container hostname. Static single-node rendezvous uses loopback.
RUNNER = LEGACY_RUNNER.replace("'--standalone',", "'--nnodes=1','--node-rank=0',\n"
    "      '--master-addr=127.0.0.1','--master-port=29500','--max-restarts=0',")


def validate(c):
    if (not re.fullmatch(r'(?:[a-zA-Z0-9][a-zA-Z0-9_.-]*@)?[a-zA-Z0-9][a-zA-Z0-9_.-]*', c['host'])
            or not re.fullmatch('[a-z0-9][a-z0-9-]*', c['tag'])):
        raise ValueError('safe host alias and unique tag required')
    root=Path(c['package']);manifest=json.loads((root/'PREPARED.json').read_text())
    if file_hash(root/'PREPARED.json')!=c['manifest_sha256']:
        raise ValueError('instruction package manifest changed')
    actual={str(p.relative_to(root)):file_hash(p) for p in root.rglob('*')
            if p.is_file() and p!=root/'PREPARED.json'}
    if any(p.is_symlink() for p in root.rglob('*')) or actual!=manifest['files']:
        raise ValueError('instruction package content changed')
    config=json.loads((root/'config.json').read_text());remote=lifecycle.path(manifest['remote_root'])
    plan=derive(config['batch_config']);world=plan['dp_replicas']
    topology(config,world,['davinci'+str(d) for d in config['physical_devices']])
    if (manifest['image']!=lifecycle.IMAGE or manifest['status']!='instruction_finite_prepared_only'
            or config['start_update']!=0 or config['stop_update']!=2
            or config['engineering_prefix'] is not True or config['training_dispatch_allowed'] is not False
            or config['checkpoint_interval']!=1 or config.get('resume_root')
            or config['source_root']!=remote+'/source'
            or config['schedule']!=remote+'/source/schedule'
            or config['parent_schedule']!=remote+'/source/parent-schedule'
            or manifest['batch_plan']!=plan or manifest['training_devices']!=config['physical_devices']
            or manifest['inference_services']!=[]
            or manifest['layout']!=dict(configs=['config.json','PREPARED.json','run.py'],
                services={},directories=['run','supervision'])
            or (root/'run.py').read_text() not in (RUNNER,LEGACY_RUNNER)
            or file_hash(root/'source/schedule/MANIFEST.json')!=config['schedule_sha256']
            or file_hash(root/'source/parent-schedule/MANIFEST.json')!=config['parent_schedule_sha256']):
        raise ValueError('bounded DP8 instruction acceptance required')
    for name,h in config['pins'].items():
        lifecycle.path(name)
        if name.startswith(remote+'/') and actual.get(name[len(remote)+1:])!=h:
            raise ValueError('instruction dependency pin changed')
    generate(root/'source')
    evidence=lifecycle.release_evidence(c,config)
    if evidence and (Path(evidence[1]).is_relative_to(remote)
                    or Path(remote).is_relative_to(evidence[1])):
        raise ValueError('release evidence must be disjoint from run root')
    return manifest,config,{}


def prepare(c,provenance,*,tested_runtime=None):
    project=Path(c['project']);dest=Path(c['package']);remote=lifecycle.path(c['remote_root'])
    design=json.loads(Path(c['design']).read_text());sft=design['stage1']
    if design['training_dispatch_allowed'] is not False:
        raise ValueError('finite development design required')
    policy=Path(c['initial_policy_local']);m=json.loads((policy/'MANIFEST.json').read_text())
    bundle=read_execution_bundle(policy,expected_policy_id=c['initial_policy_id'],
        expected_contract=m['contract'],expected_base_id=c['base_id'])
    if m['contract']['token']['registration_status']!='admitted':
        raise ValueError('admitted initial decision marker required')
    stage_lora=None
    if 'lora' in sft:
        from multimodal_jev.context_stage_lora import bind_design
        stage_lora=bind_design(design,m['contract'])
    rows,task,_=load_release(c['release'],c['release_sha256'],'learning',purpose='optimize')
    schedule=Path(c['schedule']);parent=Path(c['parent_schedule'])
    batches,_=load_schedule(schedule,c['schedule_sha256'],release_sha256=c['release_sha256'],
        population={r['id']:r for r in rows},batch_config=sft['batch_config'],
        parent_root=parent,parent_manifest_sha256=c['parent_schedule_sha256'],
        parent_release_sha256=c['parent_release_sha256'])
    plan=derive(sft['batch_config'])
    if len(rows)!=sft['batch_config']['source_queries'] or len(batches)!=plan['total_updates']:
        raise ValueError('full declared-DP source schedule required')
    if 'stage1_layout' in c:
        from multimodal_jev.stage1_training_layout import Stage1TrainingLayout
        from tools.reports.context_infonce_trainer import SOURCE_SHA
        if file_hash(project/'tools/jev/train_context_instruction_npu.py')!=SOURCE_SHA:
            raise ValueError('legacy trainer cannot consume layout experiments')
        input_spec=c['stage1_layout']
        layout_reader=Stage1TrainingLayout.open(dict(stage='instruction',stage1_layout=input_spec,
            release_sha256=c['release_sha256'],instruction_objective=sft['instruction_objective']),
            [r['id'] for r in rows],task)
        from multimodal_jev.stage1_layout_cache import read_view
        cache_spec=c.get('layout_cache')
        if not cache_spec:raise ValueError('complete native and permuted TRAIN cache proof required')
        read_view(cache_spec,layout_reader,dict(c,batch_config=sft['batch_config'],pins=c['external_pins']))
        if json.loads(Path(input_spec['design']['path']).read_bytes())['stage1']['sha256']!=file_hash(c['design']):
            raise ValueError('native Stage1 design differs from the unified design')
    check_paths({'instruction_package':dest});dest.mkdir(parents=True,exist_ok=False)
    source=dest/'source'
    from multimodal_jev.stage1_package_sources import FINITE, copy_checked
    if tested_runtime is None:  # standalone development packaging
        names=[str(p.relative_to(project)) for p in (project/'src/multimodal_jev').glob('*.py')]+list(FINITE)
        for name in names:
            target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(project/name,target)
    else:
        names=[name for name in tested_runtime if name.startswith('src/') or name in FINITE]
        copy_checked(project,source,names,tested_runtime)
    shutil.copytree(schedule,source/'schedule');shutil.copytree(parent,source/'parent-schedule')
    generate(source)
    pins={remote+'/source/'+str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
    for p in policy.rglob('*'):
        if p.is_file():pins[c['initial_policy_remote']+'/'+str(p.relative_to(policy))]=file_hash(p)
    # Bind only needed reusable assets, not another experiment's output tree.
    pins.update(c['external_pins'])
    config=dict(model=c['model'],base_id=c['base_id'],runtime_profile=m['contract']['runtime_profile'],
        source_policy=c['initial_policy_remote'],source_policy_id=bundle['policy_id'],source_contract=m['contract'],
        source_root=remote+'/source',physical_devices=c['physical_devices'],
        schedule=remote+'/source/schedule',schedule_sha256=c['schedule_sha256'],
        parent_schedule=remote+'/source/parent-schedule',parent_schedule_sha256=c['parent_schedule_sha256'],
        schedule_release_sha256=c['parent_release_sha256'],release=c['release'],release_sha256=c['release_sha256'],
        encoded_audio_cache=c['encoded_audio_cache'],audio_mirror_root=c['audio_mirror_root'],
        stage='instruction',start_update=0,stop_update=2,engineering_prefix=True,training_dispatch_allowed=False,
        checkpoint_interval=1,resume_root=None,batch_config=sft['batch_config'],global_batch=sft['global_batch'],
        microbatch=sft['batch_config']['microbatch'],instruction_objective=sft['instruction_objective'],
        lr_schedule=sft['lr_schedule'],pins=pins)
    if stage_lora is not None:config['stage_lora_transition']=stage_lora
    if 'stage1_layout' in c:
        config['stage1_layout']={}
        for key,spec in dict(c['stage1_layout'],cache=c['layout_cache']).items():
            name='layout-'+key+'.json'
            target=source/name
            shutil.copyfile(spec['path'],target)
            if file_hash(target)!=spec['sha256']:raise ValueError('layout input changed during packaging')
            binding=dict(path=name,sha256=spec['sha256'])
            if key=='cache':config['layout_cache']=binding
            else:config['stage1_layout'][key]=binding
            pins[remote+'/source/'+name]=spec['sha256']
    if c.get('prompt_layout') is not None:
        from multimodal_jev.context_decision_contract import PREFIX_LAYOUT
        if c['prompt_layout']!=PREFIX_LAYOUT:raise ValueError('unknown instruction prompt layout')
        config['prompt_layout']=c['prompt_layout']
    commit_json(dest/'config.json',config)
    (dest/'run.py').write_text(RUNNER)
    manifest=dict(status='instruction_finite_prepared_only',image=lifecycle.IMAGE,remote_root=remote,
        training_devices=c['physical_devices'],inference_services=[],batch_plan=derive(sft['batch_config']),
        layout=dict(configs=['config.json','PREPARED.json','run.py'],services={},directories=['run','supervision']),
        source_initial_policy_id=bundle['policy_id'],design_sha256=file_hash(c['design']),provenance=provenance,
        files={str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()})
    commit_json(dest/'PREPARED.json',manifest)
    return dict(status=manifest['status'],manifest_sha256=file_hash(dest/'PREPARED.json'),
                source_initial_policy_id=bundle['policy_id'],updates=2,full_schedule_updates=len(batches))


def command(c,m,config):
    argv=lifecycle.command(c,m,config)
    mount=config['encoded_audio_cache']+':'+config['encoded_audio_cache']
    if argv.count(mount)!=1:raise ValueError('expected input cache mount required')
    argv[argv.index(mount)]=mount+':ro'
    return argv


def launch(c,provenance):
    # A local two-rank launch probes the same static rendezvous argv before
    # allocating eight NPUs; retain compatibility for auditing old packages.
    if (Path(c['package'])/'run.py').read_text()!=RUNNER:
        raise ValueError('obsolete hostname-dependent runner cannot be launched')
    rendezvous=json.loads(Path(c['rendezvous_report']).read_text())
    if (file_hash(c['rendezvous_report'])!=c['rendezvous_report_sha256']
            or rendezvous['image']!=lifecycle.IMAGE or rendezvous['state']['ExitCode']!=0
            or rendezvous['terminal']['exit_code']!=0
            or rendezvous['counts']!=dict(tests=1,failures=0,errors=0,skipped=0)):
        raise ValueError('isolated static rendezvous acceptance required')
    tested_runner=Path(rendezvous['collection']).parent/(Path(rendezvous['collection']).name.removesuffix('-collected')+'-package')/'tools/reports/context_instruction_operations.py'
    tested_manifest=Path(rendezvous['collection'])/'package/FILES.json'
    if (file_hash(tested_manifest)!=rendezvous['source_manifest_sha256']
            or file_hash(tested_runner)!=json.loads(tested_manifest.read_text())['tools/reports/context_instruction_operations.py']):
        raise ValueError('rendezvous source snapshot changed')
    import ast
    nodes=ast.parse(tested_runner.read_text());values={}
    for node in nodes.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('LEGACY_RUNNER','RUNNER'):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<tested-runner>','exec'),values)
    if values['RUNNER']!=RUNNER:
        raise ValueError('rendezvous test did not exercise this runner')
    report=json.loads(Path(c['cpu_report']).read_text())
    if (file_hash(c['cpu_report'])!=c['cpu_report_sha256']
            or report['image']!=lifecycle.IMAGE
            or report['state']['Status']!='exited' or report['state']['ExitCode']!=0
            or report['terminal']['exit_code']!=0
            or report['counts']!=dict(tests=c.get('cpu_expected_tests',32),failures=0,errors=0,skipped=0)):
        raise ValueError('independent native DP8 CPU acceptance required')
    training=json.loads((Path(c['package'])/'config.json').read_text())
    width=derive(training['batch_config'])['dp_replicas']
    if width!=8:
        import xml.etree.ElementTree as ET
        junit=Path(report['collection'])/'run/junit.xml'
        if file_hash(junit)!=report['files'].get('run/junit.xml'):
            raise ValueError('native CPU test evidence changed')
        cases=list(ET.parse(junit).getroot().iter('testcase'))
        expected='test_native_full_and_tail_update_with_fresh_resume['+str(width)+']'
        if not any(node.attrib.get('name')==expected and not list(node) for node in cases):
            raise ValueError('CPU acceptance must exercise declared DP width')
    if training['instruction_objective']['kind']=='candidate_infonce_probability_margin_v1':
        import xml.etree.ElementTree as ET
        junit=Path(report['collection'])/'run/junit.xml'
        if file_hash(junit)!=report['files'].get('run/junit.xml'):raise ValueError('margin CPU evidence changed')
        cases=list(ET.parse(junit).getroot().iter('testcase'))
        if not any(n.attrib.get('name')=='test_native_margin_full_tail_and_fresh_resume' and not list(n) for n in cases):
            raise ValueError('new margin objective requires native CPU gradient/replay acceptance')
    layout_sources=()
    if 'stage1_layout' in training:
        import xml.etree.ElementTree as ET
        junit=Path(report['collection'])/'run/junit.xml'
        if file_hash(junit)!=report['files'].get('run/junit.xml'):raise ValueError('layout CPU evidence changed')
        passed={n.get('name') for n in ET.parse(junit).getroot().iter('testcase') if not list(n)}
        if not {'test_actual_stage1_batch_rebinds_layout_on_prefetch_and_fresh_resume',
                'test_text_view_rejects_misbound_native_and_unknown_order'} <= passed:
            raise ValueError('actual layout consumer and cache rejection checks required')
        layout_sources=tuple('src/multimodal_jev/'+n+'.py' for n in
                ('stage1_candidate_layout','stage1_training_layout','stage1_layout_cache','cached_context_batches'))
    tested=Path(report['collection'])/'package/FILES.json'
    if file_hash(tested)!=report['source_manifest_sha256']:
        raise ValueError('CPU acceptance source manifest changed')
    sources=json.loads(tested.read_text());package=Path(c['package'])/'source'
    for name in (*layout_sources,'src/multimodal_jev/context_infonce.py','src/multimodal_jev/context_warmup.py',
                 'src/multimodal_jev/context_native_actor.py',
                 'src/multimodal_jev/readonly_encoded_audio_cache.py',
                 'tools/reports/context_infonce_actor.py','tools/reports/context_infonce_trainer.py',
                 'tools/jev/train_context_instruction_npu.py'):
        if sources.get(name)!=file_hash(package/name):
            raise ValueError('NPU candidate differs from accepted CPU runtime: '+name)
    runner=FunctionType(lifecycle.launch.__code__,dict(lifecycle.launch.__globals__,validate=validate,command=command),
                        lifecycle.launch.__name__)
    return runner(c,provenance)


def detach(template,local):
    c,_=load_config(template,local)
    validate(c)
    root=Path(c['operations_root'])/'driver'
    check_paths({'driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.context_instruction_operations',
          '--config',str(Path(template).resolve()),'--local',str(Path(local).resolve()),'--action','launch']
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True)
    record=dict(pid=child.pid,argv=argv,started_unix=time.time(),
                process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    commit_json(root/'started.json',record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','validate','launch','detach'],required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    if a.action=='launch':
        terminal=Path(c['operations_root'])/'driver'/'terminal.json'
        try:
            result=launch(c,provenance)
        except BaseException as exc:
            if terminal.parent.is_dir():
                commit_json(terminal,dict(exit_code=1,error=repr(exc),finished_unix=time.time()))
            raise
        else:
            if terminal.parent.is_dir():
                commit_json(terminal,dict(exit_code=0,result=result,finished_unix=time.time()))
    elif a.action=='detach':result=detach(a.config,a.local)
    elif a.action=='prepare':result=prepare(c,provenance)
    else:result={'status':'validated_only','manifest_sha256':c['manifest_sha256'],
                 'argv':command(c,*validate(c)[:2])}
    print(json.dumps(result,indent=2))
