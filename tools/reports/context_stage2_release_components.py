"""Compose Stage2 packaging, admission, lifecycle and existing numerical audits.

No runtime source rewriting. Historical model integration and numerical
auditors remain explicitly pinned dependencies, not new implementations.
"""
import importlib
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.objective_parallel_admission import MODULES,clone_graph
from tools.reports import context_composed_pipeline as lifecycle
from tools.reports import context_stage2_package_components as components
from tools.reports.context_cached_storage_operations import mounts,audit_mounts
from tools.reports.prepare_objective_training_candidate import read_bound


def command(c,m,config,**kwargs):
    argv=lifecycle.command(c,m,config,producer=False)
    root=config['objective_extension']['root']
    if root!=m['remote_root']+'/objective-extension':raise ValueError('bound extension required')
    out=[];index=0
    while index<len(argv):
        if argv[index]=='-v' and Path(argv[index+1].split(':')[1]).is_relative_to(root):
            index+=2;continue
        out.append(argv[index]);index+=1
    index=out.index('--entrypoint');out[index:index]=['-v',root+':'+root+':ro']
    audit_mounts(out,config)
    return out


def cpu_evidence(c):
    report=json.loads(read_bound(c['startup_cpu_report']))
    snapshot=Path(c['startup_cpu_source']);files=json.loads((snapshot/'FILES.json').read_text())
    if (report['counts']!=dict(tests=c['startup_cpu_tests'],failures=0,errors=0,skipped=0)
            or report['image']!=lifecycle.shared.IMAGE or report['state']['ExitCode']!=0
            or report['state']['OOMKilled'] or report['terminal']['exit_code']!=0
            or file_hash(snapshot/'FILES.json')!=report['source_manifest_sha256']):
        raise ValueError('complete fixed-image startup evidence required')
    package=Path(c['training_package']['output']);config=json.loads((package/'config.json').read_text())
    pairs=[(name,'tools/reports/'+source) for name,source in (
        ('run.py','context_stage2_runtime_runner.py'),('runtime-startup.py','context_runtime_startup.py'),
        ('device-visibility.py','context_device_visibility.py'))]
    pairs += [('objective-extension/'+name,'tools/reports/context_objective_bootstrap.py' if name=='bootstrap.py' else name)
              for name in config['objective_extension']['files']]
    for packaged,tested in pairs:
        if files.get(tested)!=file_hash(package/packaged) or file_hash(snapshot/tested)!=files[tested]:
            raise ValueError('startup evidence differs: '+packaged)
    return dict(report_sha256=c['startup_cpu_report']['sha256'],tests=c['startup_cpu_tests'])


def finalization(c,p,*,launch,auditor,recovery=False):
    root=Path(c['operations_root'])/'finalization';root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',p)
    try:
        result=launch(c,p)
        if result['exit_code']!=0:raise ValueError('actor execution failed; exact collection retained')
        stem=Path(c['operations_root'])/c['tag']
        bound=dict(training_package=c['training_package'],deployment=c,objective_loss_atol=c['objective_loss_atol'],
            **{k:str(stem)+suffix for k,suffix in (
                ('launch','-launch.json'),('inspect','-inspect.json'),('wait','-wait.json'),
                ('transport','-collected.transport.json'),('collection','-collected'),('archive','-collected.tar'))})
        commit_json(root/'audit-config.json',bound);proof=auditor(bound,p)
        commit_json(root/'audit.json',proof)
        commit_json(root/'terminal.json',dict(exit_code=0,status='objective_recovery_collected_and_audited'
                    if recovery else 'objective_prefix_collected_and_audited',recovery_admitted=recovery))
        return proof
    except BaseException as error:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(error),restart=False,preserve_handles=True));raise


def graph(engine,queue,trace_auditor=None):
    names=(*MODULES,'audit_context_cached_training','context_cached_efficiency','context_cached_formal',
           'prepare_stage2_objective_run','stage2_objective_run_operations','context_stage2_package_components')
    modules=[importlib.import_module('tools.reports.'+name) for name in dict.fromkeys(names)]
    views=clone_graph(modules,{
        ('tools.reports.context_infonce_selection','reconstruct_plan'):engine.reconstruct,
        ('tools.reports.prepare_context_reference_job','replay_reference_outputs'):engine.replay,
        ('tools.reports.context_objective_training_operations','bootstrap_evidence'):cpu_evidence})
    def view(name):return views['tools.reports.'+name]
    if trace_auditor is not None:
        for name in ('audit_objective_training','audit_objective_recovery'):
            module=view(name)
            module.audit_outputs.__globals__['audit_traces']=trace_auditor
    def replace(module,name,fn):
        old=getattr(module,name)
        for module_name,value in views.items():
            for f in vars(value).values():
                if isinstance(f,FunctionType) and f.__module__==module_name:
                    for key,current in list(f.__globals__.items()):
                        if current is old:f.__globals__[key]=fn
            for key,current in list(vars(value).items()):
                if current is old:setattr(value,key,fn)
        setattr(module,name,fn)
    parts=view('context_stage2_package_components')
    replace(view('prepare_objective_training_candidate'),'expected',parts.expected)
    replace(view('prepare_stage2_objective_run'),'expected',parts.stage_expected)
    # Every validator, collector and independent output audit sees identical argv.
    objective_modules={'context_objective_training_operations','context_objective_recovery_operations',
                       'audit_objective_training','audit_objective_recovery','stage2_objective_run_operations'}
    for module_name,module in views.items():
        if module_name.rsplit('.',1)[1] not in objective_modules:continue
        for f in vars(module).values():
            if isinstance(f,FunctionType) and f.__module__==module_name and 'command' in f.__globals__:f.__globals__['command']=command
        if hasattr(module,'command'):module.command=command
    recovery=view('prepare_objective_recovery');cached=view('prepare_context_cached_recovery')
    def implementation():
        def inputs(c,p):return parts.recovery_inputs(c,p,api=cached,parent_audit=recovery.parent_admission,contract=recovery.expected_contract)
        local=clone_graph([importlib.import_module('tools.reports.prepare_context_cached_recovery')],{
            ('tools.reports.prepare_context_cached_recovery','inputs'):inputs,
            ('tools.reports.prepare_context_cached_recovery','expected_contract'):recovery.expected_contract,
            ('tools.reports.prepare_context_cached_recovery','replay_runner'):lambda m:components.runtime_files()['run.py'].decode()})
        api=local['tools.reports.prepare_context_cached_recovery']
        api.configured.__globals__['expected_contract']=recovery.expected_contract
        return api
    replace(recovery,'implementation',implementation)
    def launcher(module,authorization,status):
        def launch(c,p):
            if c.get(authorization) is not True:raise ValueError('explicit stage dispatch required')
            # Keep full validation inside lease; only source/reference leaf
            # proofs reuse authenticated results. No generic audit-hook cache.
            def locked_check(paths):
                lifecycle.EFFECTS.check_paths(paths);module.validate(c);engine.verify()
            effects=SimpleNamespace(**dict(vars(lifecycle.EFFECTS),check_paths=locked_check))
            fn=FunctionType(lifecycle._launch.__code__,dict(lifecycle._launch.__globals__,command=command))
            result=fn(dict(c,finite_training_dispatch_allowed=True),p,validate=module.validate,
                      queue=queue,priority=1,producer=False,effects=effects)
            if result['exit_code']!=0:raise ValueError('Stage2 execution failed; retained collection')
            return dict(result,status=status)
        return launch
    for name,is_recovery in [('context_objective_training_operations',False),('context_objective_recovery_operations',True)]:
        module=view(name);launch=launcher(module,'finite_training_dispatch_allowed','collected_independent_objective_training_audit_required')
        auditor=view('audit_objective_recovery' if is_recovery else 'audit_objective_training').audit
        def run(c,p,launch=launch,auditor=auditor,recovery=is_recovery):
            return finalization(c,p,launch=launch,auditor=auditor,recovery=recovery)
        replace(module,'run',run);replace(module,'launch',launch)
    stage=view('stage2_objective_run_operations')
    replace(stage,'launch',launcher(stage,'stage2_training_dispatch_allowed','collected_independent_cached_training_audit_required'))
    replace(stage,'admission',stage_admission(views))
    return views


def stage_admission(views):
    def admission(c):
        module=views['tools.reports.stage2_objective_run_operations']
        spec=c['training_package'];kind=spec['kind'];evidence=c['acceptance']
        required={'prefix','recovery'}|({'tail'} if kind=='formal' else set())
        if kind not in ('tail','formal') or set(evidence)!=required:
            raise ValueError('prefix/recovery then tail/formal; no standalone efficiency stage')
        module.preparation.audit(spec,{'source':'composed_stage2_admission'})
        p=module.prefix.audit(evidence['prefix'],{'source':'composed_prefix_admission'})
        r=module.recovery.audit(evidence['recovery'],{'source':'composed_recovery_admission'})
        if (p['status']!='finite_objective_training_outputs_audited' or r['status']!='finite_objective_recovery_audited'
                or r['fresh_process_npu_resume_verified'] is not True or r['replay']['exact_objective_traces'] is not True
                or evidence['prefix']['training_package']!=spec['prefix_package']
                or evidence['recovery']['training_package']['parent_audit']!=evidence['prefix']):
            raise ValueError('same complete prefix and fresh recovery required')
        config=module.preparation.read(Path(spec['output'])/'config.json')
        plugin=module.ObjectivePlugin(module.prefix.packaged_plugin(spec['output'],config))
        if plugin.identity!=p['objective_plugin'] or plugin.identity['options']!={}:raise ValueError('same default editable objective required')
        parent=evidence['prefix']['deployment']
        if c['reference_audit']!=parent['reference_audit'] or any(c[k]!=parent[k] for k in ('host','lease')):
            raise ValueError('same reference and four-card ownership required')
        if kind=='formal':
            prior=evidence['tail'];ops=prior['deployment']
            if (prior['training_package']['prefix_package']!=spec['prefix_package'] or prior['training_package']['kind']!='tail'
                    or ops['reference_audit']!=c['reference_audit'] or any(ops[k]!=c[k] for k in ('host','lease'))
                    or any(ops['acceptance'][k]!=evidence[k] for k in ('prefix','recovery'))):
                raise ValueError('tail lineage differs')
            if module.audit(prior,{'source':'composed_tail_admission'})['status']!='stage2_objective_tail_audited':
                raise ValueError('independent real-tail audit required')
        return dict(status='stage2_objective_'+kind+'_admitted',objective_plugin=plugin.identity,
                    efficiency_statistics='sampled_from_formal_training')
    return admission
