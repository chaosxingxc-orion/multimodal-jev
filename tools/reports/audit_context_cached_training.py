"""Audit two cached-reference updates through the shared GRPO result auditor.

Only reference provenance and the checkpoint contract change. Credit, schedule,
actor parity, gradients, exports, activation and exact transport remain required.
No dispatch, full-training release or efficacy claim is made here.
"""
import argparse
import copy
import json
import inspect
from pathlib import Path
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_reference_dataset import ReferenceDataset,digest
from multimodal_jev.context_stage_lora import admit_stage2
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports import audit_context_replica_training as shared
from tools.reports.context_cached_reference_trainer import validate_objective
from tools.reports.context_cached_storage_operations import command,audit_container
from tools.reports.prepare_context_cached_training import audit as audit_package
from multimodal_jev.context_cached_tail_window import contract as tail_contract,select as select_tail


def read(p):return json.loads(Path(p).read_text())


def expected_contract(c,mounted):
    contract=shared.expected_contract(c,mounted)
    contract['stage_lora_transition']=admit_stage2(c['stage_lora_transition'],c['source_contract'])
    if 'objective_config' in c:
        from multimodal_jev.stage2_view_runtime import editable_objective
        contract['stage2_objective']=editable_objective(c)
    else:
        contract['stage2_objective']=validate_objective(c)
    contract['reference_dataset']=dict(manifest_sha256=c['reference_dataset']['manifest_sha256'],
        identity=c['reference_dataset']['identity'],mode='readonly_no_online_fill')
    if c.get('experiment_kind')=='cached_reference_tail_admission':
        contract['tail_admission']=tail_contract(c)
    return contract


def tail_auditor(namespace):
    """Retain shared artifact checks; use authenticated boundary batches only."""
    source=inspect.getsource(shared.audit_outputs)
    replacements=[
        ("if experiment_window(c) != 'engineering_prefix' or start != 0:",
         "if tail_contract(c)['diagnostic_updates'] != 2 or start != 0:"),
        ('        check_schedule(batches, plan)',
         '        check_schedule(batches, plan)\n        batches=select_tail(batches,c,population.keys())'),
        ("if padded and (engineering\n                or index % plan['updates_per_epoch'] != plan['updates_per_epoch'] - 1):",
         "if padded and index != 0:"),
    ]
    for before,after in replacements:
        if source.count(before)!=1:raise ValueError('shared tail audit integration anchor changed')
        source=source.replace(before,after)
    scope=dict(namespace,tail_contract=tail_contract,select_tail=select_tail)
    exec(compile(source,'<cached-tail-output-audit>','exec'),scope)
    return scope['audit_outputs']


class ReferenceConsumption:
    def __init__(self,config,root):
        # The public package audit separately reconstructs producer journals and
        # original full-TRAIN evidence. This table proves exact consumed content.
        reader=ReferenceDataset(root,manifest_sha256=config['manifest_sha256'],
            expected_identity=config['identity'],expected_population=config['population'])
        self.records=read(Path(root)/'records.json')
        self.population=reader.population;self.policy=reader.policy_id
        self.runtime=reader.runtime_identity;self.samples=0

    def check(self,receipt,batch):
        if (receipt['fixed_reference_policy_id']!=self.policy
                or receipt['runtime_identity']!=self.runtime
                or type(receipt['reference_misses']) is not int or receipt['reference_misses']!=0
                or type(receipt['bulk_reference_calls']) is not int or receipt['bulk_reference_calls']!=0
                or type(receipt['bulk_behavior_calls']) is not int or receipt['bulk_behavior_calls']!=1
                or receipt['switches']!=[] or not receipt['queries']
                or len(receipt['queries'])!=len(batch['records'])):
            raise ValueError('cached training must perform only behavior inference without reference switches')
        for query,slot in zip(receipt['queries'],batch['records'],strict=True):
            behavior=query['behavior_record'];reference=query['reference_record']
            if (query['request_id']!=slot['request_id'] or slot['id'] not in self.population
                    or behavior['policy_id']!=receipt['behavior_policy_id']
                    or behavior['request_identity']['policy_id']!=receipt['behavior_policy_id']):
                raise ValueError('scheduled sample and behavior policy differ')
            binding=copy.deepcopy(behavior['request_identity']);binding['policy_id']=self.policy
            key=digest(binding)
            if (self.population[slot['id']]!=key or key not in self.records
                    or reference!=self.records[key] or reference['request_identity']!=binding):
                raise ValueError('consumed reference differs from published sample/input/probability')
        self.samples+=len(receipt['queries'])


def audit_outputs(package,collection,*,reference_root):
    package,collection=Path(package),Path(collection)
    config=read(package/'config.json')
    kind=config['experiment_kind']
    if (kind not in ('engineering_prefix','cached_reference_tail_admission') or config['engineering_prefix'] is not True
            or config['training_dispatch_allowed'] is not False
            or config['start_update']!=0 or config['stop_update']!=2):
        raise ValueError('two-update cached-reference acceptance only')
    if kind=='cached_reference_tail_admission':tail_contract(config)
    verifier=ReferenceConsumption(config['reference_dataset'],reference_root)
    run=collection/'run'
    receipts={}
    for index in range(2):
        receipt=read(run/f'rollout-{index:04d}.json')
        if receipt['update']!=index:raise ValueError('exact next cached update required')
        verifier.check(receipt,read(run/f'batch-{index:04d}.json'))
        receipts[index]=receipt

    def transition(receipt,seen,*,current,reference,replicas):
        if (receipt!=receipts.get(receipt['update']) or reference!=verifier.policy
                or receipt['behavior_policy_id']!=current or replicas!=2):
            raise ValueError('independently verified cached transition required')
        # Empty missing set deliberately yields exactly one behavior macro per
        # update. Shared journal coverage rejects unaccounted reference calls.
        return []

    namespace=dict(shared.audit_outputs.__globals__,
        expected_contract=expected_contract,reference_transition=transition)
    auditor=(tail_auditor(namespace) if kind=='cached_reference_tail_admission' else
        FunctionType(shared.audit_outputs.__code__,namespace,'audit_cached_outputs'))
    result=auditor(package,collection,initial_policy=package/'initial/policy')
    result.update(status='finite_cached_reference_training_outputs_audited',
        reference_manifest_sha256=config['reference_dataset']['manifest_sha256'],
        consumed_reference_queries=verifier.samples,online_reference_calls=0,
        reference_policy_switches=0,training_dispatch_allowed=False)
    if kind=='cached_reference_tail_admission':
        result.update(status='finite_cached_reference_tail_outputs_audited',experiment_kind=kind,
            tail_admission=tail_contract(config))
    return result


def audit(c,provenance):
    return audit_collected(c,provenance,package_check=audit_package,build_command=command,
                          check_outputs=audit_outputs)


def reference_dataset_path(specification, config):
    """Bind the local audit copy to the reference mounted on the training host."""
    root = Path(specification['reference_job']['collection']) / 'reference-build' / 'dataset'
    manifest = root / 'MANIFEST.json'
    if (root.is_symlink() or manifest.is_symlink() or not manifest.is_file()
            or file_hash(manifest) != config['reference_dataset']['manifest_sha256']):
        raise ValueError(f'local reference audit manifest is missing or differs: {manifest}')
    return root


def audit_collected(c,provenance,*,package_check,build_command,check_outputs):
    specification=c['training_package'];package=Path(specification['output'])
    preparation=package_check(specification,provenance)
    manifest=read(package/'PREPARED.json');config=read(package/'config.json')
    argv=build_command(c['deployment'],manifest,config)
    handle=read(c['launch']);observed=read(c['inspect']);transport=read(c['transport'])
    storage=audit_container(argv,config,handle,observed)
    if (handle['manifest_sha256']!=specification['manifest_sha256']
            or handle['remote_root']!=manifest['remote_root'] or handle['layout']!=manifest['layout']
            or handle['config_sha256']!=file_hash(package/'config.json')
            or handle['services']!={n:file_hash(package/p) for n,p in manifest['layout']['services'].items()}
            or handle['image']!=observed['Image']
            or read(c['wait'])!=dict(container_id=handle['container_id'],exit_code=0)
            or transport['status']!='transport_complete_independent_audit_required'
            or transport['container_id']!=handle['container_id'] or transport['image']!=observed['Image']
            or transport['remote_root']!=manifest['remote_root'] or transport['state']!=observed['State']
            or any(transport[k] for k in ('bad_pins','bad_package','launch_binding_mismatches'))):
        raise ValueError('successful exact cached training transport required')
    collection=Path(c['collection'])
    actual={str(p.relative_to(collection)):file_hash(p) for p in collection.rglob('*') if p.is_file()}
    if (collection.is_symlink() or any(p.is_symlink() for p in collection.rglob('*'))
            or actual!=transport['files'] or file_hash(c['archive'])!=transport['archive_sha256']):
        raise ValueError('complete unchanged cached training collection required')
    for name in manifest['layout']['configs']+list(manifest['layout']['services'].values()):
        if actual.get(name)!=file_hash(package/name):raise ValueError('executed training config differs: '+name)
    reference=reference_dataset_path(specification, config)
    result=check_outputs(package,collection,reference_root=reference)
    return dict(result,package_audit=preparation,storage_audit=storage,
        container_id=handle['container_id'],image=observed['Image'],
        manifest_sha256=specification['manifest_sha256'],transport_sha256=file_hash(c['transport']),
        numerical_admission_separate=True,provenance=provenance)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    result=audit(c,provenance);commit_json(c['output'],result)
    print(json.dumps(result,indent=2))
