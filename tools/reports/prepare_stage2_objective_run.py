"""Prepare tail/efficiency/formal runs from the frozen editable-objective prefix.

No forward tree changes. Formal/efficiency use the tested plugin with detailed
traces disabled; their default objective must be byte-identical to acceptance.
Tail uses a separately pinned generated trainer with the original tail schedule.
Preparation never grants dispatch; lifecycle requires real acceptance evidence.
"""
import argparse
import copy
import inspect
import json
from pathlib import Path
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports import prepare_objective_training_candidate as prefix
from tools.reports import context_cached_formal as formal
from tools.reports import context_cached_efficiency as efficiency
from tools.reports import prepare_context_cached_training as cached
from tools.reports.prepare_context_replica_training_candidate import rebase
from tools.reports.context_objective_bootstrap import OBJECTIVE


def read(p):return json.loads(Path(p).read_text())


def expected(c,provenance):
    prefix.audit(c['prefix_package'],provenance)
    parent=Path(c['prefix_package']['output']);old=read(parent/'PREPARED.json');remote=c['remote_root'];cached.path(remote)
    output=Path(c['output'])
    config=read(parent/'config.json');kind=c['kind']
    if (kind not in ('tail','efficiency','formal') or config['objective_plugin']['options']!={}
            or file_hash(parent/'objective-extension'/OBJECTIVE)!=file_hash(Path(__file__).parents[2]/OBJECTIVE)):
        raise ValueError('default accepted objective and explicit run kind required')
    if (output.resolve().is_relative_to(parent.resolve()) or parent.resolve().is_relative_to(output.resolve())
            or any(Path(remote).is_relative_to(p) or Path(p).is_relative_to(remote)
                for p in [old['remote_root'],config['reference_dataset']['root']])):
        raise ValueError('disjoint fresh run and readonly reference required')
    files={n:(parent/n).read_bytes() for n in old['files']}
    config=rebase(config,old['remote_root'],remote)
    if kind=='formal':
        config.update(experiment_kind='formal_source',engineering_prefix=False,training_dispatch_allowed=True,
            stop_update=634,evaluation_steps=list(formal.STEPS),objective_trace=False)
        config.pop('checkpoint_interval',None);formal.contract(config)
        runner=formal.runner(old)
    elif kind=='efficiency':
        config.update(experiment_kind='efficiency',stop_update=100,objective_trace=False)
        runner=efficiency.runner(old)
    else:
        config.update(experiment_kind='cached_reference_tail_admission',objective_trace=False)
        from multimodal_jev.context_cached_tail_window import contract
        contract(config);runner=cached.runner(old,'cached_reference_tail_admission')
        # Keep accepted extension modules byte-identical. This outer wrapper
        # derives the same plugin generator with the original tail generator.
        files['objective-tail-entry.py']=TAIL_ENTRY.encode()
    before="str(r/'source/tools/reports/"+('context_cached_tail_trainer.py' if kind=='tail' else 'context_cached_reference_trainer.py')+"')"
    if runner.count(before)!=1:raise ValueError('actor runner anchor changed')
    target="str(r/'objective-tail-entry.py')" if kind=='tail' else "str(r/'objective-extension/bootstrap.py')"
    files['run.py']=runner.replace(before,target).encode()
    names={'config.json',*old['layout']['services'].values()}
    pins={n:h for n,h in config['pins'].items() if not Path(n).is_relative_to(remote)}
    pins.update({remote+'/'+n:prefix.sha(raw) for n,raw in files.items() if n not in names});config['pins']=pins
    files['config.json']=prefix.encode(config)
    for name in old['layout']['services'].values():
        value=rebase(read(parent/name),old['remote_root'],remote);value['pins']=pins;files[name]=prefix.encode(value)
    m=copy.deepcopy(old);m.update(status='stage2_objective_run_prepared_only',remote_root=remote,
        parent_sha256=c['prefix_package']['manifest_sha256'],config_provenance=provenance,
        experiment=dict(kind=kind,updates=config['stop_update']),files={n:prefix.sha(raw) for n,raw in files.items()},
        numerical_admission=False,training_dispatch_allowed=False)
    if kind=='tail':m['layout']['configs'].append('objective-tail-entry.py')
    return files,m


TAIL_ENTRY='''import argparse,importlib.util,json
from pathlib import Path
from types import FunctionType
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
c=json.loads(Path(a.config).read_text());root=Path(c['objective_extension']['root'])
spec=importlib.util.spec_from_file_location('pinned_objective_bootstrap',root/'bootstrap.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);trainer=b.install(c)
from tools.reports.context_cached_tail_trainer import generate as tail_generate
from multimodal_jev.context_cached_tail_window import contract
contract(c)
def generate(source_root,formal=False):
 if formal:raise ValueError('tail cannot be formal')
 return tail_generate(source_root)
generator=FunctionType(trainer.generate.__code__,dict(trainer.generate.__globals__,original_generate=generate),argdefs=trainer.generate.__defaults__)
generator.__kwdefaults__=trainer.generate.__kwdefaults__
scope={'__name__':'objective_tail_derived'}
exec(compile(generator(c['source_root']),'<objective-tail>','exec'),scope)
scope['run'](c,Path(a.output))
'''


def prepare(c,provenance):
    root=Path(c['output'])
    if root.exists():raise ValueError('new run directory required')
    files,m=expected(c,provenance);check_paths({'stage2_package':root});root.mkdir(parents=True)
    for n,raw in files.items():
        p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    commit_json(root/'PREPARED.json',m)
    return dict(status=m['status'],manifest_sha256=file_hash(root/'PREPARED.json'),npu_executed=False)


def audit(c,provenance):
    root=Path(c['output']);manifest=root/'PREPARED.json'
    if root.is_symlink() or file_hash(manifest)!=c['manifest_sha256']:raise ValueError('stage2 manifest changed')
    m=read(manifest);files,expected_manifest=expected(c,m['config_provenance'])
    if (any(p.is_symlink() for p in root.rglob('*')) or m!=expected_manifest
            or {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file() and p!=manifest}!=files):
        raise ValueError('stage2 package differs from independent reconstruction')
    return dict(status='stage2_objective_run_independently_reconstructed',manifest_sha256=c['manifest_sha256'],
        kind=c['kind'],npu_executed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','audit'],default='prepare');a=p.parse_args();c,prov=load_config(a.config,a.local)
    print(json.dumps((prepare if a.action=='prepare' else audit)(c,prov),indent=2))
