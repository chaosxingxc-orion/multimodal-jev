"""HF-first isolated parity lifecycle over the existing two-service runner.

The preflight role runs HF on card6 and exits before service0 reuses card6;
service1 uses card7. The client has no NPU. No training or target scoring.
"""
import argparse
import json
from pathlib import Path
import re

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports.context_macro_probe_operations import command as base_command,run as lifecycle,IMAGE
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.context_transfer_parity_inputs import load as load_inputs
from tools.reports.context_transfer_macro_parity import compare
from tools.reports.audit_context_macro_evaluation import audit


def read(p):return json.loads(Path(p).read_text())


def command(c,role):
    argv=base_command(c,role);root=c['remote_root'];package=root+'/package'
    if role in ('preflight','client'):
        cache=c['cache_root']
        if (not Path(cache).is_absolute() or ':' in cache or '..' in Path(cache).parts
                or Path(cache).is_relative_to(root) or Path(root).is_relative_to(cache)):
            raise ValueError('disjoint readonly cache mount required')
        extra=['-v',cache+':'+cache+':ro']
        if role=='preflight':
            for name in ('davinci6','davinci_manager','devmm_svm','hisi_hdc'):
                extra+=['--device','/dev/'+name+':/dev/'+name]
        at=argv.index('--entrypoint');argv[at:at]=extra
        code="""import json
from pathlib import Path
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.context_dual_runtime_candidate import verify_runtime
from tools.reports.context_transfer_macro_parity import run
from multimodal_jev.baseline import file_hash
r=Path(ROOT);verify(r)
for i in range(2):verify_runtime(json.loads((r/f'service-{i}.json').read_text()))
c=json.loads((r/'evaluation.json').read_text())
""".replace('ROOT',repr(package))
        if role=='preflight':
            code+="c.update(output="+repr(root+'/hf')+",physical_devices=[6])\nrun(c,{'source':'sealed_hf_parity'},'hf')\n"
        else:
            code+="h=Path("+repr(root+'/hf')+")\n"
            code+="assert json.loads((h/'terminal.json').read_text())=={'exit_code':0}\n"
            code+="c['hf_reference']={'path':str(h/'reference.json'),'sha256':file_hash(h/'reference.json')}\n"
            code+="run(c,{'source':'sealed_macro_parity'},'macro')\n"
        argv=argv[:argv.index(IMAGE)+1]+['python','-B','-c',code]
    return argv


def validate(c):
    root=c['remote_root'];package=Path(c['prepared'])/'package'
    receipt=read(package.parent/'PREPARED.json');e=read(package/'evaluation.json')
    if (not Path(root).is_absolute() or '..' in Path(root).parts or root in ('/','/data','/mnt')
            or receipt['status']!='macro_target_parity_prepared_only' or receipt['image']!=IMAGE
            or receipt['files_sha256']!=c['manifest_sha256'] or verify(package)!=c['manifest_sha256']
            or receipt['remote_root']!=root or receipt['ipc_root']!=root+'/ipc'
            or receipt['training_dispatch_allowed'] is not False):
        raise ValueError('sealed parity-only runtime required')
    for key in ('host','tag'):
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',c[key]):raise ValueError('safe host/tag required')
    spec=e['portable_parity_inputs']
    if (spec['root']!=root+'/package/parity-inputs' or spec['cache_root']!=c['cache_root']
            or e['output']!=root+'/evaluation' or e['model']!=c['model']
            or e['snapshot']!=root+'/package/policy' or e['adapter']!=root+'/package/adapter'
            or e['policy_id']!=receipt['policy_id'] or e['dataset']!=receipt['dataset']):
        raise ValueError('bound input/cache/policy paths required')
    admitted=load_inputs(dict(spec,root=str(package/'parity-inputs')))
    if (admitted['dataset']!=e['dataset'] or len(admitted['population'])!=receipt['recordings']
            or admitted['processor_sha256']!=receipt['runtime_identity']['processor_sha256']):
        raise ValueError('complete admitted target panel required')
    from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
    bundle=read_execution_bundle(package/'policy',expected_policy_id=e['policy_id'],
        expected_contract=e['contract'],expected_base_id=e['base_id'])
    verify_native_adapter(package/'adapter',bundle)
    client=e['macro_client']
    if (client['max_queries']!=128 or client['rollout_queries']!=32 or len(client['replicas'])!=2
            or client['attempt_id']!='target-parity-v1' or client['journal_root']!=root+'/journal'
            or client['runtime_identity']!=receipt['runtime_identity']):
        raise ValueError('same macro128 dual runtime required')
    files=read(package/'FILES.json')
    expected_pins={root+'/package/'+n:h for n,h in files.items()
        if n not in ('evaluation.json','service-0.json','service-1.json')}
    if e['pins']!=expected_pins:raise ValueError('complete client dependency pins required')
    for i in range(2):
        s=read(package/f'service-{i}.json');p=s['policy'];endpoint=client['replicas'][i]
        if (s['pins']!=expected_pins or p['physical_devices']!=[6+i] or p.get('process_device_scope') is not None
                or p['model']!=c['model'] or p['snapshot']!=e['snapshot'] or p['adapter']!=e['adapter']
                or p['policy_id']!=e['policy_id'] or p['contract']!=e['contract'] or p['base_id']!=e['base_id']
                or p['max_decisions']!=len(admitted['candidate_ids']) or p['max_batch_queries']!=32
                or p['max_batch_tokens']!=131072 or s['source_root']!=root+'/package'
                or s['source_policy']!=e['snapshot'] or s['dual_runtime_identity']!=client['runtime_identity']
                or s['socket_path']!=endpoint['socket'] or s['exchange']!=endpoint['exchange']
                or not all(Path(v).is_relative_to(root+'/ipc') for v in endpoint.values())):
            raise ValueError('isolated same-policy target-width services required')
    for role in ('preflight','client','service-0','service-1'):command(c,role)
    return receipt


def audit_outputs(package,collection):
    package=Path(package);collection=Path(collection);e=read(package/'evaluation.json')
    admitted=load_inputs(dict(e['portable_parity_inputs'],root=str(package/'parity-inputs')))
    selected=admitted['selected_panel'];client=e['macro_client']
    for role in ('hf','evaluation','service-0','service-1'):
        if read(collection/role/'terminal.json')['exit_code']!=0:raise ValueError('all parity phases must succeed')
    for role in ('hf','evaluation'):
        if read(collection/role/'panel.json')!=selected:raise ValueError('actual backend panel differs')
    from multimodal_jev.context_runtime_profile import PROFILE
    if read(collection/'hf/hf-runtime.json')['profile']!=PROFILE:
        raise ValueError('actual HF runtime profile differs')
    for i in range(2):
        s=read(package/f'service-{i}.json');ready=read(collection/f'service-{i}/ready.json')
        if (ready['physical_devices']!=[6+i] or ready['policy_id']!=e['policy_id']
                or ready['contract']!=e['contract'] or ready['max_batch_queries']!=s['policy']['max_batch_queries']
                or ready['model_forward_mode']!=client['runtime_identity']['profile']):
            raise ValueError('actual target service startup differs')
    inference=collection/'evaluation/inference'
    proof=audit(inference,collection/'journal'/client['attempt_id'],population=selected['assets'],
        candidate_ids=selected['candidate_ids'],input_receipts=selected['input_receipts'],policy_id=e['policy_id'],
        runtime_identity=client['runtime_identity'],macro_queries=128)
    reference=collection/'hf/reference.json'
    result=compare(read(reference),read(inference/'predictions.json')['records'],selected,policy_id=e['policy_id'])
    reported=read(collection/'evaluation/comparison.json')
    if (not result['exact'] or any(reported.get(k)!=v for k,v in result.items())
            or reported['hf_reference']!={'path':str(Path(e['output']).parent/'hf/reference.json'),'sha256':file_hash(reference)}
            or read(collection/'evaluation/macro-output-audit.json')!=proof
            or reported['macro_output_audit_sha256']!=file_hash(collection/'evaluation/macro-output-audit.json')):
        raise ValueError('independent target-width parity differs')
    return dict(status='target_width_macro_parity_outputs_audited',comparison=result,inference_audit=proof,
        scope='HF and macro outputs match independent target inputs; exact runtime transport checked by collector; finite panel only')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','run','collect'],default='validate')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    if a.action=='run':result=lifecycle(c,provenance,validator=validate,command_builder=command)
    elif a.action=='collect':
        from tools.reports.audit_context_macro_probe import collect_and_audit
        result=collect_and_audit(c,validator=validate,command_builder=command,output_auditor=audit_outputs,
            result_dirs=('hf','evaluation','journal','service-0','service-1'))
    else:result=validate(c)
    print(json.dumps(result,indent=2))
