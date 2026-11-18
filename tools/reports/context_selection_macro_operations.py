"""Full selection lifecycle over the shared isolated two-service operations."""
import argparse
import json
from pathlib import Path
import re

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports.context_macro_probe_operations import command as base_command,run as lifecycle
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.prepare_context_batch_probe_candidate import IMAGE


def validate(c,*,source_curve=False,source_kind='grpo'):
    if source_kind not in ('grpo','infonce') or (source_kind=='infonce' and not source_curve):
        raise ValueError('explicit supported source curve kind required')
    package=Path(c['prepared'])/'package';receipt=json.loads((package.parent/'PREPARED.json').read_text())
    config=json.loads((package/'evaluation.json').read_text());root=c['remote_root']
    if (verify(package)!=c['manifest_sha256'] or receipt['files_sha256']!=c['manifest_sha256']
            or receipt['status']!=('macro_source_point_prepared_only' if source_curve else 'macro_selection_prepared_only') or receipt['image']!=IMAGE
            or receipt['remote_root']!=root or receipt['ipc_root']!=root+'/ipc'
            or receipt['selection_recordings']!=9226 or receipt['training_dispatch_allowed'] is not False
            or receipt['policy_role']!=('source_curve_point' if source_curve else 'common_Stage1') or not Path(root).is_absolute() or '..' in Path(root).parts):
        raise ValueError('sealed complete Stage1 selection package required')
    if source_curve:
        plan=json.loads((package/'SOURCE-PLAN.json').read_text())
        if source_kind=='infonce':
            from tools.reports.context_infonce_selection import validate_plan
            validate_plan(plan)
        schema='jev-infonce-source-evaluation-plan-v1' if source_kind=='infonce' else 'jev-macro-source-evaluation-plan-v1'
        steps=[0,*range(32,318,32),318] if source_kind=='infonce' else [0,*range(64,634,64),634]
        algorithm='candidate_infonce' if source_kind=='infonce' else 'dual_credit_grpo_floor_clipped_v1'
        points=[p for p in plan['checkpoints'] if p['step']==receipt['source_step']]
        if (file_hash(package/'SOURCE-PLAN.json')!=receipt['source_plan_sha256']
                or plan['schema']!=schema or len(points)!=1
                or type(receipt['source_step']) is not int
                or [p['step'] for p in plan['checkpoints']]!=steps
                or plan['algorithm']!=algorithm
                or plan['target_tuning_allowed'] is not False or plan['model_selection_performed'] is not False
                or plan['source_operations']['manifest_sha256']!=receipt['parent_manifest_sha256']
                or plan['source_audit']!=json.loads((package/'TRAINING-AUDIT.json').read_text())
                or points[0]['policy_id']!=config['policy_id']
                or points[0]['manifest_sha256']!=file_hash(package/'policy/MANIFEST.json')
                or plan['experiment']['release_sha256']!=config['release_sha256']):
            raise ValueError('sealed source curve step/policy/plan differs')
        from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
        bundle=read_execution_bundle(package/'policy',expected_policy_id=config['policy_id'],
            expected_contract=config['contract'],expected_base_id=config['base_id'])
        verify_native_adapter(package/'adapter',bundle)
    for key in ('host','tag'):
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',c[key]):raise ValueError('safe host/tag required')
    client=config['macro_client']
    if (config['policy_id']!=receipt['policy_id'] or config['output']!=root+'/evaluation'
            or config['snapshot']!=root+'/package/policy' or config['macro_queries']!=128
            or config['encoded_audio_cache']!=c['cache_root'] or config['release']!=c['release']
            or config['release_sha256']!=c['release_sha256']
            or client['runtime_identity']!=receipt['runtime_identity'] or client['max_queries']!=128
            or client['rollout_queries']!=32 or client['journal_root']!=root+'/journal'
            or client['attempt_id']!='selection-v1' or len(client['replicas'])!=2):
        raise ValueError('full population/immutable policy/macro input contract differs')
    for i in range(2):
        service=json.loads((package/f'service-{i}.json').read_text());policy=service['policy'];endpoint=client['replicas'][i]
        if (policy['physical_devices']!=[6+i] or policy['model']!=c['model']
                or policy['policy_id']!=config['policy_id'] or policy['contract']!=config['contract']
                or policy['snapshot']!=config['snapshot'] or policy['adapter']!=root+'/package/adapter'
                or service['source_root']!=root+'/package' or service['source_policy']!=config['snapshot']
                or service['dual_runtime_identity']!=client['runtime_identity']
                or service['socket_path']!=endpoint['socket'] or service['exchange']!=endpoint['exchange']
                or not all(Path(v).is_relative_to(root+'/ipc') for v in endpoint.values())):
            raise ValueError('isolated fixed-policy matching services required')
    files=json.loads((package/'FILES.json').read_text())
    for path,h in config['pins'].items():
        if not Path(path).is_relative_to(root+'/package') or files.get(str(Path(path).relative_to(root+'/package')))!=h:
            raise ValueError('pinned evaluation asset differs')
    for role in ('preflight','client','service-0','service-1'):command(c,role)
    return receipt


def command(c,role):
    argv=base_command(c,role);root=c['remote_root'];package=root+'/package'
    if role in ('client','preflight'):
        extra=[]
        for path in (c['release'],c['cache_root']):
            if (not Path(path).is_absolute() or ':' in path or '..' in Path(path).parts
                    or Path(path).is_relative_to(root) or Path(root).is_relative_to(path)):
                raise ValueError('isolated readonly publication/cache mounts required')
            extra+=['-v',path+':'+path+':ro']
        argv[argv.index('--entrypoint'):argv.index('--entrypoint')]=extra
        if role=='client':
            code="import json; from tools.reports.evaluate_context_selection_macro import run; run(json.load(open("+repr(package+'/evaluation.json')+")), {'source':'sealed_full_selection'})"
        else:
            code="""import json
from pathlib import Path
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.context_dual_runtime_candidate import verify_runtime
from tools.reports.evaluate_context_selection_macro import inputs,run
from tools.reports.context_selection_audited_cache import load_evidence,encoder
from multimodal_jev.cached_context_preparation import processor_cache_identity
from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
r=Path(ROOT);verify(r);c=json.loads((r/'evaluation.json').read_text())
for i in range(2):verify_runtime(json.loads((r/f'service-{i}.json').read_text()))
b=read_execution_bundle(c['snapshot'],expected_policy_id=c['policy_id'],expected_contract=c['contract'],expected_base_id=c['base_id'])
verify_native_adapter(r/'adapter',b)
population,requests,candidates=inputs(c)
p=processor_cache_identity(c['snapshot'])
receipts,proof=load_evidence(c,population,release_manifest=json.loads((Path(c['release'])/'MANIFEST.json').read_text()),processor_sha256=p)
encode=encoder(c['encoded_audio_cache'],requests,candidates,receipts,processor_sha256=p,decision_token_id=b['contract']['token']['id'],audio_token_id=c['audio_token_id'])
for asset in (population[0],population[-1]):encode(asset)
assert len(population)==9226
print('full selection evidence and first/last actual cache/policy/runtime preflight passed')
""".replace('ROOT',repr(package))
        argv=argv[:argv.index(IMAGE)+1]+['python','-B','-c',code]
    return argv


def audit_outputs(package,collection,*,source_curve=False):
    from tools.reports.audit_context_macro_evaluation import audit
    from tools.reports.evaluate_context_selection_macro import inputs
    from tools.jev.evaluate_context_event_records import evaluate
    from tools.jev.context_metric_equivalence import compare
    package=Path(package);collection=Path(collection)
    def read(path):return json.loads(path.read_text())
    config=read(package/'evaluation.json');population,_,candidates=inputs(config)
    independent=read(package/'evaluation-inputs/input-receipts.json')
    for role in ('evaluation','service-0','service-1'):
        if read(collection/role/'terminal.json')['exit_code']!=0:raise ValueError('successful evaluation/services required')
    for i in range(2):
        service=read(package/f'service-{i}.json');ready=read(collection/f'service-{i}/ready.json')
        if (ready['physical_devices']!=[6+i] or ready['policy_id']!=config['policy_id']
                or ready['contract']!=config['contract'] or ready['max_batch_queries']!=service['policy']['max_batch_queries']
                or ready['model_forward_mode']!=config['macro_client']['runtime_identity']['profile']):
            raise ValueError('actual service startup differs')
    inference=collection/'evaluation/inference'
    checked=audit(inference,collection/'journal'/config['macro_client']['attempt_id'],population=population,
        candidate_ids=candidates,input_receipts=independent,policy_id=config['policy_id'],
        runtime_identity=config['macro_client']['runtime_identity'],macro_queries=config['macro_queries'])
    scored=evaluate(config['release'],config['release_sha256'],inference/'predictions.json',
        file_hash(inference/'predictions.json'),config['policy_id'],'selection')
    reported=read(collection/'evaluation/result.json');comparison=compare(scored,reported)
    if (reported['input_receipts_sha256']!=checked['input_receipts_sha256']
            or reported['inference_result_sha256']!=file_hash(inference/'result.json')):
        raise ValueError('scoring output evidence differs')
    return dict(status='complete_source_point_macro_selection_audited' if source_curve else 'complete_stage1_macro_selection_audited',
        inference_audit=checked,result=scored,metric_comparison=comparison,
        scope='Full fixed-policy selection on macro runtime; full curve and target transfer remain separate')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    parser.add_argument('--action',choices=['validate','run','collect'],default='validate');args=parser.parse_args()
    c,p=load_config(args.config,args.local)
    if args.action=='run':result=lifecycle(c,p,validator=validate,command_builder=command)
    elif args.action=='collect':
        from tools.reports.audit_context_macro_probe import collect_and_audit
        result=collect_and_audit(c,validator=validate,command_builder=command,output_auditor=audit_outputs,
            result_dirs=('evaluation','journal','service-0','service-1'))
    else:result=validate(c)
    print(json.dumps(result,indent=2))
