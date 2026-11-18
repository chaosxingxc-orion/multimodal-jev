"""Derive an independently reconstructible inference-only evaluation package.

Model source, source checkpoint, full population and numerical profile stay
byte-identical. The outer runner/client and deployment topology are versioned.
"""
import argparse
import copy
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports import prepare_context_replica_selection as parent_api
from tools.reports.prepare_context_replica_training_candidate import rebase
from tools.reports.prepare_objective_training_candidate import encode,sha
from tools.reports.context_multidevice_evaluation import topology,runner,client_source


def expected(c,provenance):
    parent_api.audit(c['parent'],provenance)
    parent=Path(c['parent']['output']);m=json.loads((parent/'PREPARED.json').read_text())
    old=m['remote_root'];remote=parent_api.path(c['remote_root']);output=Path(c['output'])
    if (output.resolve().is_relative_to(parent.resolve()) or parent.resolve().is_relative_to(output.resolve())
            or Path(remote).is_relative_to(old) or Path(old).is_relative_to(remote)):
        raise ValueError('disjoint evaluation package required')
    t=topology(c['topology']);m=copy.deepcopy(m)
    files={name:(parent/name).read_bytes() for name in m['files']}
    seed=json.loads(files[next(iter(m['layout']['services'].values()))])
    for name in m['layout']['services'].values():files.pop(name)
    m.update(training_devices=[],inference_services=t['inference_services'],evaluation_topology=c['topology'],
        status='multidevice_selection_prepared_only',remote_root=remote,parent_sha256=c['parent']['manifest_sha256'],
        config_provenance=provenance,numerical_admission=False,actor_updates=0,
        pending=['per-device HF parity and full output audit','measured matched throughput'])
    m.pop('batch_plan',None);m.pop('experiment',None)
    m['layout']['services']={s['name']:f"service-{s['name']}.json" for s in t['inference_services']}
    m['layout']['directories']=['hf','hf-process','evaluation','selection-journal','supervision',
        *['service-'+s['name'] for s in t['inference_services']]]
    files['run.py']=runner(m).encode()
    files['selection-client.py']=client_source(files['selection-client.py'].decode()).encode()
    values={name:rebase(json.loads(files[name]),old,remote) for name in ['config.json','parity-hf.json','parity-macro.json']}
    replicas=[dict(socket=remote+f"/service-{s['name']}/socket",exchange=remote+f"/service-{s['name']}/exchange") for s in t['inference_services']]
    for name,value in values.items():
        value['evaluation_topology']=copy.deepcopy(c['topology'])
        value['macro_queries']=t['macro_queries']
        value['macro_client'].update(replicas=replicas,max_queries=t['macro_queries'],rollout_queries=t['service_batch_queries'])
        value.update(training_dispatch_allowed=False,runtime_preparation_only=True,physical_devices=[t['hf_device']],
            process_device_scope=dict(kind='same_container_runtime_visibility_v1',mounted_physical=t['mounted_physical'],visible_ordinal=t['hf_visible_ordinal']))
    for i,s in enumerate(t['inference_services']):
        value=rebase(copy.deepcopy(seed),old,remote)
        value.update(socket_path=replicas[i]['socket'],exchange=replicas[i]['exchange'])
        value['policy'].update(physical_devices=[s['device']],tensor_parallel_size=1,
            max_batch_queries=t['service_batch_queries'],max_batch_tokens=t['service_batch_queries']*value['policy']['max_model_len'],
            process_device_scope=dict(kind='same_container_runtime_visibility_v1',mounted_physical=t['mounted_physical'],
                visible_ordinal=str(t['mounted_physical'].index(s['device']))))
        values[m['layout']['services'][s['name']]]=value
    names=set(values)|{'parity-hf-local.json'}
    external={k:v for k,v in values['config.json']['pins'].items() if not Path(k).is_relative_to(remote)}
    pins=dict(external,**{remote+'/'+n:sha(raw) for n,raw in files.items() if n not in names})
    for value in values.values():value['pins']=pins
    values['parity-hf-local.json']=dict(schema='jev-local-macros-v1',macros={'REPLICA_SELECTION_CLIENT_CONFIG':values['parity-hf.json']})
    for name,value in values.items():files[name]=encode(value)
    # No change to frozen forward implementation, adapter, or source evidence.
    for name,raw in files.items():
        if name.startswith(('source/','initial/','selection-inputs/')) and raw!=(parent/name).read_bytes():
            raise ValueError('evaluation changed model or input evidence')
    m['files']={name:sha(raw) for name,raw in files.items()}
    return files,m


def prepare(c,provenance):
    root=Path(c['output'])
    if root.exists():raise ValueError('fresh evaluation output required')
    files,m=expected(c,provenance);check_paths({'evaluation_package':root});root.mkdir(parents=True,exist_ok=False)
    for name,raw in files.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    commit_json(root/'PREPARED.json',m)
    return dict(status=m['status'],manifest_sha256=file_hash(root/'PREPARED.json'),npu_executed=False)


def audit(c,provenance):
    root=Path(c['output']);manifest=root/'PREPARED.json'
    if root.is_symlink() or file_hash(manifest)!=c['manifest_sha256']:raise ValueError('evaluation manifest differs')
    m=json.loads(manifest.read_text());files,wanted=expected(c,m['config_provenance'])
    actual={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file() and p!=manifest}
    if any(p.is_symlink() for p in root.rglob('*')) or actual!=files or m!=wanted:
        raise ValueError('evaluation differs from independent reconstruction')
    return dict(status='multidevice_selection_package_audited_not_executed',manifest_sha256=c['manifest_sha256'],
        source_step=m['source_step'],policy_id=m['source_policy_id'],replicas=len(m['inference_services']),
        npu_executed=False,numerical_admission=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','audit'],default='prepare');a=p.parse_args()
    c,provenance=load_config(a.config,a.local);print(json.dumps((prepare if a.action=='prepare' else audit)(c,provenance),indent=2))
