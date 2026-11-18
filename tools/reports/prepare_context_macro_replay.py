"""Seal one fresh-process DP2 step from an independently audited prefix.

Copies only the first-step LoRA/head/FP32 optimizer shards. Runtime Python,
base model, full TRAIN schedule and fixed reference identity remain unchanged.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports.audit_context_macro_training import audit_collected, expected_contract
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.run_context_batch_probe_candidate import verify


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def seal(c,provenance,parent,admission):
    old=Path(parent['remote_root']);remote=Path(c['remote_root'])
    source=Path(parent['prepared'])/'package';output=Path(c['output'])
    collection=Path(parent['operations_root'])/'collected';run=collection/'run'
    if (output.exists() or output.resolve().is_relative_to(source.resolve())
            or not remote.is_absolute() or '..' in remote.parts or ':' in str(remote)
            or remote.is_relative_to(old) or old.is_relative_to(remote)
            or len(str(remote/'ipc').encode())+8>107):
        raise ValueError('new disjoint replay output and runtime required')
    if (admission['status']!='finite_dp2_training_audited'
            or admission['files_sha256']!=verify(source)
            or admission['fresh_process_replay_verified'] is not False):
        raise ValueError('independently audited initial DP2 prefix required')
    original=json.loads((source/'training.json').read_text())
    if original['start_update']!=0 or original['stop_update']!=2 or original.get('resume_root'):
        raise ValueError('original two-step prefix required')
    transport=json.loads((Path(parent['operations_root'])/'transport.json').read_text())
    commit=json.loads((run/'checkpoint-0001-committed.json').read_text())
    if (commit['updates']!=1 or commit['contract']!=expected_contract(original)
            or set(commit['rank_shards'])!={f'rank-{r}/checkpoint-0001.pt' for r in range(2)}):
        raise ValueError('complete DP2 first-step optimizer contract required')
    required={'checkpoint-0001-committed.json'}|set(commit['rank_shards'])
    for directory in ('policy-0001','adapter-0001'):
        names={n[4:] for n in transport['files'] if n.startswith('run/'+directory+'/')}
        if not names:raise ValueError('complete policy and adapter required')
        required.update(names)
    for name in required:
        if (Path(name).is_absolute() or '..' in Path(name).parts or (run/name).is_symlink()
                or file_hash(run/name)!=transport['files']['run/'+name]):
            raise ValueError('first-step recovery asset changed')
    output.mkdir(parents=True);dest=output/'package';shutil.copytree(source,dest)
    (dest/'FILES.json').unlink()
    for name in sorted(required):
        target=dest/'resume'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(run/name,target)
    config=relocate(original,str(old),str(remote))
    config.update(start_update=1,resume_root=str(remote/'package/resume'),
        initial_adapter=str(remote/'package/resume/adapter-0001'))
    config['macro_client']['attempt_id']='training-replay-v1'
    binding=dict(parent_operations=parent,parent_manifest_sha256=admission['files_sha256'],
        parent_container=admission['containers']['client'],parent_audit=admission,
        initial_behavior_policy_id=commit['joint_policy']['policy_id'])
    write(dest/'REPLAY.json',binding)
    # Keep the exact runtime source manifest. The saved policy must never add
    # Python files to the execution payload or cause a runtime identity change.
    runtime=json.loads((dest/'RUNTIME-SOURCES.json').read_text())
    if {str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*.py')}!=runtime:
        raise ValueError('replay must preserve exact runtime Python sources')
    pins=dict(config['pins'])
    for name in required:pins[str(remote/'package/resume'/name)]=file_hash(dest/'resume'/name)
    config['pins']=pins
    if expected_contract(config)!=commit['contract']:raise ValueError('replay changed objective/runtime')
    write(dest/'training.json',config)
    for i in range(2):
        service=relocate(json.loads((source/f'service-{i}.json').read_text()),str(old),str(remote))
        service['pins']={k:v for k,v in pins.items() if Path(k).is_relative_to(remote/'package')}
        service['policy'].update(snapshot=str(remote/'package/resume/policy-0001'),
            adapter=config['initial_adapter'],policy_id=binding['initial_behavior_policy_id'])
        write(dest/f'service-{i}.json',service)
    write(dest/'FILES.json',{str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()})
    record=json.loads((source.parent/'PREPARED.json').read_text())
    record.update(status='macro_training_replay_prepared_only',files_sha256=verify(dest),
        remote_root=str(remote),ipc_root=str(remote/'ipc'),config_provenance=provenance,
        replay_sha256=file_hash(dest/'REPLAY.json'),pending=['actual fresh-process NPU step2','independent exact replay comparison'])
    write(output/'PREPARED.json',record)
    return record


def prepare(c,provenance):
    parent,_=load_config(c['parent_config'],c['parent_local'])
    admission=audit_collected(parent)
    return seal(c,provenance,parent,admission)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(prepare(c,provenance),indent=2))
