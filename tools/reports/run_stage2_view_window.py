"""One prepared window, one lease and one independent post-collection audit."""
import json
from pathlib import Path
import re
from types import SimpleNamespace

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.stage2_view_runtime import indices,layout,pinned
from tools.reports import context_composed_pipeline as lifecycle
from tools.reports.audit_context_cached_training import reference_dataset_path
from tools.reports.audit_stage2_view_window import audit
from tools.reports.context_async_pipeline import ReadyQueue
from tools.reports.context_cached_acceptance_sequence import audit_config
from tools.reports.context_cached_formal_successor import monitored_training
from tools.reports.context_release_deployment import audit as release_audit
from tools.reports.context_stage2_release_components import command
from tools.reports.prepare_stage2_view_package import admission,COMPONENTS,ROOT


def request(c):
    if (c.get('schema')!='jev-stage2-view-window-v1' or c.get('dispatch') is not True
            or c.get('monitor_interval_seconds')!=900 or c.get('window') not in ('prefix','recovery','tail','formal')):
        raise ValueError('explicit single-window dispatch and 900-second observer required')


def check_image_cpu(c,manifest,package):
    """Bind the successful fixed-image test to the actual actor and window components."""
    cpu=pinned(c['image_cpu']['path'],c['image_cpu']['sha256'])
    counts=cpu['counts'];collection=Path(cpu['collection'])
    if (cpu['schema']!='configured-fixed-image-cpu-tests-v1' or cpu['image']!=manifest['image']
            or cpu['state']['Status']!='exited' or cpu['state']['ExitCode']!=0 or cpu['state']['OOMKilled']
            or cpu['terminal']['exit_code']!=0 or counts['tests']<=0 or any(counts[k] for k in ('errors','failures','skipped'))):
        raise ValueError('successful image CPU evidence required')
    files=pinned(collection/'package/FILES.json',cpu['source_manifest_sha256'])
    junit=collection/'run/junit.xml'
    if file_hash(junit)!=cpu['files']['run/junit.xml']:raise ValueError('fixed image tests changed')
    import xml.etree.ElementTree as ET
    names={n.get('name') for n in ET.parse(junit).iter('testcase') if len(n)==0}
    if not {'test_view_packaged_once_with_dynamic_budget[prefix]','test_view_packaged_once_with_dynamic_budget[tail]',
            'test_view_packaged_once_with_dynamic_budget[formal]','test_recovery_contract_keeps_prefix_identity_and_tail_uses_epoch_cursor'}<=names:
        raise ValueError('image CPU must exercise this window contract')
    for name,source in COMPONENTS.items():
        if files.get(source)!=file_hash(package/name):raise ValueError('image CPU used another component: '+name)
    config = pinned(package/'config.json', manifest['files']['config.json'])
    setting = config['view_execution'].get('activation_checkpointing')
    if setting and setting['enabled'] and 'test_activation_checkpointing_preserves_outputs_gradients_and_names' not in names:
        raise ValueError('fixed-image CPU checks must cover activation checkpointing')
    if layout(config)['dp_replicas']==6:
        if 'test_dp6_view_launch_tail_and_topology_binding' not in names:
            raise ValueError('fixed-image CPU checks must cover the DP6 view')
    for name in ('tests/test_stage2_view_package.py','tools/reports/prepare_stage2_view_package.py',
                 'tools/reports/audit_stage2_view_window.py','tools/reports/run_stage2_view_window.py',
                 'tools/reports/audit_context_replica_training.py',
                 'tools/reports/context_composed_pipeline.py','tools/reports/audit_objective_recovery.py',
                 'tools/reports/audit_context_cached_recovery.py'):
        if files.get(name)!=file_hash(ROOT/name):raise ValueError('fixed-image control source differs: '+name)


def run(c,p,*,trace_auditor=None):
    request(c)
    ops=dict(c['deployment']);root=Path(c['output'])
    package=Path(ops['package']);spec=c['training_package']
    if not re.fullmatch(r'(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9_.-]*',ops['host']) or not re.fullmatch('[a-z0-9][a-z0-9-]*',ops['tag']):
        raise ValueError('canonical host/tag required')
    for key in ('package','lease','operations_root'):lifecycle.path(ops[key])
    if root.resolve().is_relative_to(package.resolve()) or package.resolve().is_relative_to(root.resolve()):
        raise ValueError('disjoint driver and package required')
    if package!=Path(spec['output']) or ops['manifest_sha256']!=spec['manifest_sha256']:raise ValueError('deployment package identity differs')
    ops.update(training_package=spec,window_dispatch=True)
    check_paths({'controller':root,'operations':ops['operations_root']});root.mkdir(parents=True,exist_ok=False)
    try:
        def validate(_):
            m=pinned(package/'PREPARED.json',spec['manifest_sha256'])
            if (m['image']!=lifecycle.shared.IMAGE or m['status']!='stage2_view_package_prepared_not_admitted'
                    or m['view_window']!=c['window']):raise ValueError('wrong window/image')
            files={str(x.relative_to(package)):file_hash(x) for x in package.rglob('*') if x.is_file() and x!=package/'PREPARED.json'}
            if files!=m['files'] or any(x.is_symlink() for x in package.rglob('*')):raise ValueError('package changed')
            cfg=pinned(package/'config.json',files['config.json']);indices(cfg)
            check_image_cpu(c,m,package)
            reference_dataset_path(spec, cfg)
            if c['window']=='formal' and (cfg['training_dispatch_allowed'] is not True or
                    admission(c['admission'],cfg['view_lineage'],world_size=layout(cfg)['dp_replicas'])!=cfg['view_admission']):
                raise ValueError('formal admission differs from package')
            release_audit(command(ops,m,cfg),cfg);return m,cfg,{}
        queue=ReadyQueue(root/'events')
        effects=SimpleNamespace(**vars(lifecycle.EFFECTS),stage_command=command)
        def execute(op,prov):
            return lifecycle.run_stage(op,prov,validate=validate,queue=queue,priority=1,producer=False,
                authorization='window_dispatch',effects=effects)
        collected=monitored_training(ops,p,root,run_stage=execute)
        if collected['exit_code']!=0:raise ValueError('window not collected')
        ac=audit_config(ops,None);ac.update(objective_loss_atol=c['objective_loss_atol'],prefix=c.get('prefix'))
        result=audit(ac,p,**({'trace_auditor':trace_auditor} if trace_auditor else {}))
        commit_json(root/'audit.json',result)
        commit_json(root/'terminal.json',dict(exit_code=0,status=result['status'],audit_sha256=file_hash(root/'audit.json')))
        return result
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True));raise
