"""Seal a sparse formal checkpoint from an exact terminal attempt, without launch.

Accepts a failed formal attempt only after its complete transport and committed
learning prefix pass independent checks. Resumed-segment composition remains
mandatory before reporting a complete run.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.context_policy_artifacts import read_execution_bundle, verify_native_adapter
from tools.reports.context_macro_formal_operations import validate as validate_parent, command
from tools.reports.context_macro_probe_operations import inspect_runtime
from tools.reports.context_macro_formal_candidate import contract, STEPS
from tools.reports.context_macro_formal_resume import expected_contract, validate_resume
from tools.reports.context_master_export_audit import inspect_committed
from tools.jev.audit_context_rl_training import verify_checkpoint
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.prepare_context_macro_replay import write
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.audit_context_macro_formal_prefix import audit_prefix


def checkpoint(c, parent):
    receipt=validate_parent(parent)
    step=c['step']
    if type(step) is not int or step not in STEPS[1:-1]:
        raise ValueError('committed nonterminal sparse checkpoint required')
    package=Path(parent['prepared'])/'package';base=Path(parent['operations_root'])
    collection=base/'collected';run=collection/'run'
    training=json.loads((package/'training.json').read_text())
    inspections=json.loads((base/'collected-inspect.json').read_text())
    if set(inspections)!={'preflight','client','service-0','service-1'}:
        raise ValueError('complete exact terminal attempt required')
    for role,observed in inspections.items():
        handle=json.loads((base/(role+'-launch.json')).read_text())
        if handle['argv']!=command(parent,role):raise ValueError('executed formal command differs')
        inspect_runtime(observed,handle)
        if observed['State']['Running'] or observed['State']['Status']!='exited':
            raise ValueError('do not snapshot a live formal attempt')
        if role=='preflight' and (observed['State']['ExitCode']!=0 or observed['State'].get('OOMKilled',False)):
            raise ValueError('successful formal preflight required for recovery')
    transport_path=base/'transport.json'
    if file_hash(transport_path)!=c['transport_sha256']:
        raise ValueError('explicit collected transport binding required')
    transport=json.loads(transport_path.read_text());files=transport['files']
    if file_hash(base/'results.tar')!=transport['archive_sha256']:
        raise ValueError('collected archive changed')
    if any(p.is_symlink() for p in collection.rglob('*')):
        raise ValueError('collection symlink forbidden')
    actual={str(p.relative_to(collection)):file_hash(p)
            for p in collection.rglob('*') if p.is_file()}
    if actual!=files:
        raise ValueError('complete collected prefix transport changed')
    def bound(name):
        rel=Path(name);p=collection/rel
        if (rel.is_absolute() or '..' in rel.parts or p.is_symlink()
                or not p.resolve().is_relative_to(collection.resolve())
                or files.get(name)!=file_hash(p)):
            raise ValueError('unbound collected checkpoint asset: '+name)
        return p
    commit_name=f'checkpoint-{step:04d}-committed.json'
    commit=json.loads(bound('run/'+commit_name).read_text())
    expected=expected_contract(training);expected['formal_experiment']=contract(training)
    required={commit_name,f'execution-{step:04d}.pt',
              *[f'rank-{rank}/checkpoint-{step:04d}.pt' for rank in range(2)]}
    if (commit['updates']!=step or commit['contract']!=expected
            or set(commit['rank_shards'])!=required-{commit_name,f'execution-{step:04d}.pt'}
            or commit['joint_policy_directory']!=f'policy-{step:04d}'):
        raise ValueError('checkpoint objective, sparse cursor or ranks differ')
    for directory in (f'policy-{step:04d}',f'adapter-{step:04d}'):
        actual={str(p.relative_to(run)) for p in (run/directory).rglob('*') if p.is_file()}
        declared={name[4:] for name in files if name.startswith('run/'+directory+'/')}
        if not actual or actual!=declared:raise ValueError('complete policy/adapter inventory required')
        required.update(actual)
    for name in required:bound('run/'+name)
    snapshot=run/f'policy-{step:04d}'
    if file_hash(snapshot/'MANIFEST.json')!=commit['joint_policy_manifest_sha256']:
        raise ValueError('checkpoint policy manifest differs')
    bundle=read_execution_bundle(snapshot,expected_policy_id=commit['joint_policy']['policy_id'],
        expected_contract=training['source_contract'],expected_base_id=training['base_id'])
    verify_native_adapter(run/f'adapter-{step:04d}',bundle)
    verify_checkpoint(run,step,expected,bundle,world_size=2)
    masters=inspect_committed(dict(run_root=str(run),step=step,world_size=2,
                                  commit_sha256=file_hash(run/commit_name)))
    prefix=audit_prefix(package,collection,step)
    if (prefix['status']!='formal_dp2_prefix_outputs_audited'
            or prefix['start_update']!=0 or prefix['stop_update']!=step
            or len(prefix['updates'])!=step):
        raise ValueError('independent complete learning prefix required')
    return training,commit,required,dict(parent_manifest_sha256=receipt['files_sha256'],
        transport_sha256=c['transport_sha256'],checkpoint=masters,
        containers={role:obs['Id'] for role,obs in inspections.items()},
        container_evidence_sha256=file_hash(base/'collected-inspect.json'),
        recovery_files={name:files['run/'+name] for name in sorted(required)},
        prefix_audit=prefix,prefix_learning_history_verified=True)


def prepare(c,provenance):
    parent,_=load_config(c['parent_config'],c['parent_local'])
    original,commit,required,evidence=checkpoint(c,parent)
    output=Path(c['output']);source=Path(parent['prepared'])/'package'
    old=Path(parent['remote_root']);remote=Path(c['remote_root']);step=c['step']
    if (output.exists() or not output.is_absolute()
            or output.resolve().is_relative_to(source.resolve())
            or source.resolve().is_relative_to(output.resolve())
            or not remote.is_absolute() or '..' in remote.parts or ':' in str(remote)
            or remote.is_relative_to(old) or old.is_relative_to(remote)
            or len(str(remote/'ipc').encode())+8>107):
        raise ValueError('new disjoint bounded recovery paths required')
    check_paths({'recovery_package':output})
    output.mkdir(parents=True);dest=output/'package';shutil.copytree(source,dest)
    (dest/'FILES.json').unlink()
    run=Path(parent['operations_root'])/'collected/run'
    for name in sorted(required):
        target=dest/'resume'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(run/name,target)
        if file_hash(target)!=evidence['recovery_files'][name]:raise ValueError('recovery copy changed')
    runtime=json.loads((source/'RUNTIME-SOURCES.json').read_text())
    if {str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*.py')}!=runtime:
        raise ValueError('recovery must retain exact frozen runtime sources')
    config=relocate(original,str(old),str(remote))
    config.update(start_update=step,resume_root=str(remote/'package/resume'),
        initial_adapter=str(remote/'package/resume'/f'adapter-{step:04d}'))
    config['macro_client']['attempt_id']=f'formal-resume-{step:04d}'
    write(dest/'RECOVERY.json',dict(evidence,source_operations=parent,step=step,
        initial_behavior_policy_id=commit['joint_policy']['policy_id'],config_provenance=provenance))
    config['pins'].update({str(remote/'package/resume'/name):evidence['recovery_files'][name] for name in required})
    config['pins'][str(remote/'package/RECOVERY.json')]=file_hash(dest/'RECOVERY.json')
    write(dest/'training.json',config)
    for i in range(2):
        service=relocate(json.loads((source/f'service-{i}.json').read_text()),str(old),str(remote))
        service['pins']={p:h for p,h in config['pins'].items() if Path(p).is_relative_to(remote/'package')}
        service['policy'].update(snapshot=str(remote/'package/resume'/f'policy-{step:04d}'),
            adapter=config['initial_adapter'],policy_id=commit['joint_policy']['policy_id'])
        write(dest/f'service-{i}.json',service)
    validate_resume(relocate(config,str(remote/'package'),str(dest.resolve())))
    write(dest/'FILES.json',{str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()})
    result=json.loads((source.parent/'PREPARED.json').read_text())
    result.update(status='macro_sparse_resume_prepared_only',files_sha256=verify(dest),
        parent_manifest_sha256=evidence['parent_manifest_sha256'],step=step,
        remote_root=str(remote),ipc_root=str(remote/'ipc'),
        selected_microbatch=config['microbatch'],formal_launch_admitted=False,
        recovery_sha256=file_hash(dest/'RECOVERY.json'),
        training_dispatch_allowed=False,prefix_learning_history_verified=True,
        pending=['resumed segment independent audit and composition with verified prefix',
                 'formal resume deployment admission and actual NPU recovery'])
    write(output/'PREPARED.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    torch.set_num_threads(1);print(json.dumps(prepare(c,provenance),indent=2))
