"""HF panel then full portable target inference, with one shared device lease."""
import argparse
import json
from pathlib import Path
import re

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
from tools.reports.context_macro_probe_operations import command as base_command,run as lifecycle,IMAGE
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.context_transfer_source_admission import admit,verify_bundle
from tools.reports.context_transfer_portable_inputs import load
from tools.reports.context_transfer_parity_inputs import load as load_panel
from tools.reports.audit_context_transfer_input_coverage import audit as coverage
from tools.reports.evaluate_context_portable_transfer import audit_outputs as audit_full
from tools.reports.context_transfer_macro_parity import compare
from tools.reports.audit_context_macro_training import audit_collected as reaudit
from tools.reports.audit_context_macro_probe import collect_and_audit


def read(p):return json.loads(Path(p).read_text())


def command(c,role):
    argv=base_command(c,role);root=c['remote_root'];package=root+'/package'
    if role in ('preflight','client'):
        cache=c['cache_root']
        if (not Path(cache).is_absolute() or '..' in Path(cache).parts or ':' in cache
                or Path(cache).is_relative_to(root) or Path(root).is_relative_to(cache)):
            raise ValueError('disjoint readonly cache required')
        extra=['-v',cache+':'+cache+':ro']
        if role=='preflight':
            for n in ('davinci6','davinci_manager','devmm_svm','hisi_hdc'):
                extra+=['--device','/dev/'+n+':/dev/'+n]
        argv[argv.index('--entrypoint'):argv.index('--entrypoint')]=extra
        code="""import json
from pathlib import Path
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.context_dual_runtime_candidate import verify_runtime
r=Path(ROOT);verify(r)
for i in range(2):verify_runtime(json.loads((r/f'service-{i}.json').read_text()))
""".replace('ROOT',repr(package))
        if role=='preflight':
            code+="from tools.reports.context_transfer_macro_parity import run\nc=json.loads((r/'hf.json').read_text())\nrun(c,{'source':'full_target_hf_panel'},'hf')\n"
        else:
            code+="from tools.reports.evaluate_context_portable_transfer import run\nc=json.loads((r/'evaluation.json').read_text())\n"
            code+="assert json.loads(Path("+repr(root+'/hf/terminal.json')+").read_text())=={'exit_code':0}\n"
            code+="run(c,{'source':'full_portable_target'})\n"
        argv=argv[:argv.index(IMAGE)+1]+['python','-B','-c',code]
    return argv


def local_config(package,e):
    return relocate(e,str(Path(e['snapshot']).parent),str(Path(package).resolve()))


def validate(c):
    root=c['remote_root'];package=Path(c['prepared'])/'package'
    receipt=read(package.parent/'PREPARED.json');e=read(package/'evaluation.json');hf=read(package/'hf.json')
    if (receipt['status']!='full_portable_transfer_prepared_only' or receipt['image']!=IMAGE
            or receipt['files_sha256']!=c['manifest_sha256'] or verify(package)!=c['manifest_sha256']
            or receipt['remote_root']!=root or receipt['ipc_root']!=root+'/ipc'
            or receipt['training_dispatch_allowed'] is not False
            or not Path(root).is_absolute() or '..' in Path(root).parts or root in ('/','/data','/mnt')):
        raise ValueError('sealed full portable transfer package required')
    for k in ('host','tag'):
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',c[k]):raise ValueError('safe host/tag required')
    if (e['snapshot']!=root+'/package/policy' or e['adapter']!=root+'/package/adapter'
            or e['model']!=c['model'] or e['output']!=root+'/evaluation'
            or e['macro_queries']!=128 or e['audio_token_id']!=151676
            or e['portable_full_inputs']['root']!=root+'/package/full-inputs'
            or e['portable_full_inputs']['cache_root']!=c['cache_root']
            or hf['portable_parity_inputs']['root']!=root+'/package/parity-inputs'
            or hf['portable_parity_inputs']['cache_root']!=c['cache_root']
            or hf['output']!=root+'/hf' or hf['physical_devices']!=[6]):
        raise ValueError('full input, policy and HF phase paths differ')
    for k in ('dataset','policy_id','contract','base_id','snapshot','adapter','model','audio_token_id'):
        if hf[k]!=e[k]:raise ValueError('HF and full inference policy/config differ')
    local=local_config(package,e);local_hf=local_config(package,hf)
    inputs=load(local['portable_full_inputs']);coverage(local['portable_full_inputs'],local_hf['portable_parity_inputs'])
    if (inputs['dataset']!=e['dataset'] or len(inputs['population'])!=receipt['recordings']
            or inputs['processor_sha256']!=receipt['runtime_identity']['processor_sha256']
            or receipt['policy_id']!=e['policy_id'] or receipt['policy_role']!=e['policy_role']
            or receipt['dataset']!=e['dataset']):raise ValueError('complete matching target input identity required')
    source=admit(c['source_admission'])
    if (source!=read(package/'SOURCE-ADMISSION.json')
            or source['binding']['policy_id']!=e['policy_id'] or source['binding']['role']!=e['policy_role']
            or source['training']['base_id']!=e['base_id'] or source['training']['source_contract']!=e['contract']
            or file_hash(package/'policy/MANIFEST.json')!=source['point']['manifest_sha256']
            or verify_bundle(package/'source',root+'/package/source',source)!=e['source_evaluation']):
        raise ValueError('actual source admission and deployed policy differ')
    if e['dataset']=='SLURP_clean_TEST60' and (e.get('source_release')!=inputs['source_release']
            or e['source_release']['sha256']!=source['training']['release_sha256']):
        raise ValueError('clean TEST must use the actual trained source ontology')
    bundle=read_execution_bundle(package/'policy',expected_policy_id=e['policy_id'],
        expected_contract=e['contract'],expected_base_id=e['base_id'])
    verify_native_adapter(package/'adapter',bundle)
    client=e['macro_client']
    if (client['max_queries']!=128 or client['rollout_queries']!=32 or len(client['replicas'])!=2
            or client['attempt_id']!='full-transfer-v1' or client['journal_root']!=root+'/journal'
            or client['runtime_identity']!=receipt['runtime_identity']):raise ValueError('macro128 dual runtime required')
    files=read(package/'FILES.json')
    pins={root+'/package/'+n:h for n,h in files.items() if n not in ('evaluation.json','hf.json','service-0.json','service-1.json')}
    if e['pins']!=pins or hf['pins']!=pins:raise ValueError('complete full evaluation dependency pins required')
    sources=read(package/'RUNTIME-SOURCES.json')
    if ({str(p.relative_to(package)):file_hash(p) for p in package.rglob('*.py')}!=sources
            or client['runtime_identity']['source_manifest_sha256']!=file_hash(package/'RUNTIME-SOURCES.json')):
        raise ValueError('complete full evaluation runtime identity required')
    parent=source['source_operations'];parent_package=Path(parent['prepared'])/'package'
    parent_receipt=read(parent_package.parent/'PREPARED.json')
    expected_runtime=dict(parent_receipt['runtime_identity'],source_manifest_sha256=file_hash(package/'RUNTIME-SOURCES.json'))
    if receipt['runtime_identity']!=expected_runtime:
        raise ValueError('full evaluation changed parent precision/processor/image profile')
    for i in range(2):
        s=read(package/f'service-{i}.json');p=s['policy'];endpoint=client['replicas'][i]
        original=relocate(read(parent_package/f'service-{i}.json'),parent['remote_root'],root)
        original.update(pins=pins,dual_runtime_identity=expected_runtime,
                        socket_path=endpoint['socket'],exchange=endpoint['exchange'])
        original['policy'].update(policy_id=e['policy_id'],max_decisions=len(inputs['candidate_ids']),
                                  max_batch_queries=32,max_batch_tokens=131072)
        if s!=original:raise ValueError('full target service must preserve parent execution settings')
        if (s['pins']!=pins or p['physical_devices']!=[6+i] or p.get('process_device_scope') is not None
                or p['model']!=c['model'] or p['snapshot']!=e['snapshot'] or p['adapter']!=e['adapter']
                or p['policy_id']!=e['policy_id'] or p['contract']!=e['contract'] or p['base_id']!=e['base_id']
                or p['max_decisions']!=len(inputs['candidate_ids']) or p['max_batch_queries']!=32
                or p['max_batch_tokens']!=131072 or s['source_root']!=root+'/package'
                or s['source_policy']!=e['snapshot'] or s['dual_runtime_identity']!=client['runtime_identity']
                or s['socket_path']!=endpoint['socket'] or s['exchange']!=endpoint['exchange']
                or not all(Path(v).is_relative_to(root+'/ipc') for v in endpoint.values())):
            raise ValueError('same-policy isolated full transfer services required')
    for role in ('preflight','client','service-0','service-1'):command(c,role)
    return receipt


def audit_outputs(package,collection):
    package=Path(package);collection=Path(collection);e=read(package/'evaluation.json')
    local=local_config(package,e);hf=local_config(package,read(package/'hf.json'))
    panel=load_panel(hf['portable_parity_inputs'])['selected_panel']
    coverage(local['portable_full_inputs'],hf['portable_parity_inputs'])
    for role in ('hf','evaluation','service-0','service-1'):
        if read(collection/role/'terminal.json')['exit_code']!=0:raise ValueError('all full evaluation phases must succeed')
    from multimodal_jev.context_runtime_profile import PROFILE
    if read(collection/'hf/panel.json')!=panel or read(collection/'hf/hf-runtime.json')['profile']!=PROFILE:
        raise ValueError('actual HF panel/profile differs')
    for i in range(2):
        ready=read(collection/f'service-{i}/ready.json')
        if (ready['physical_devices']!=[6+i] or ready['policy_id']!=e['policy_id']
                or ready['contract']!=e['contract'] or ready['max_batch_queries']!=32
                or ready['model_forward_mode']!=e['macro_client']['runtime_identity']['profile']):
            raise ValueError('actual full transfer service startup differs')
    proof=audit_full(local,collection/'evaluation',collection/'journal'/e['macro_client']['attempt_id'])
    records=read(collection/'evaluation/inference/predictions.json')['records']
    selected=[records[i] for i in panel['indices']]
    comparison=compare(read(collection/'hf/reference.json'),selected,panel,policy_id=e['policy_id'])
    if not comparison['exact']:raise ValueError('HF panel differs from actual full inference outputs')
    return dict(status='full_portable_transfer_outputs_audited',inference_audit=proof,
        parity=comparison,dataset=e['dataset'],policy_id=e['policy_id'],policy_role=e['policy_role'],
        target_scored=False,runtime_provenance_verified=False,
        scope='Complete target outputs and same-policy HF subset checked; runtime/transport admitted separately, no task score.')


def audit_collected(c):
    result=reaudit(c,validator=validate,command_builder=command,output_auditor=audit_outputs)
    result['runtime_provenance_verified']=True
    return result


def collect(c):
    collect_and_audit(c,validator=validate,command_builder=command,output_auditor=audit_outputs,
        result_dirs=('hf','evaluation','journal','service-0','service-1'))
    result=audit_collected(c)
    from tools.reports.launch_context_batch_probe_candidate import write_new
    write_new(Path(c['operations_root'])/'full-transfer-audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    parser.add_argument('--action',choices=['validate','run','collect','audit'],default='validate')
    args=parser.parse_args();c,p=load_config(args.config,args.local)
    import torch
    torch.set_num_threads(1)
    if args.action=='run':result=lifecycle(c,p,validator=validate,command_builder=command)
    elif args.action=='collect':result=collect(c)
    elif args.action=='audit':result=audit_collected(c)
    else:result=validate(c)
    print(json.dumps(result,indent=2))
