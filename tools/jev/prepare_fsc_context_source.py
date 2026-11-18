"""Seal the GRPO source-only selected policy pair for frozen FSC transfer.

Re-audits complete source training and selection. Copies evaluated policy IDs
verbatim so existing target inference/scoring contracts require no relabeling.
No target labels, device allocation, probability inference or target tuning.
"""
import argparse
import json
from pathlib import Path
import shutil

from tools.jev.compare_context_rl_curves import load_run
from tools.jev.context_rl_evaluation_plan import bound, sha


def read(path):
    return json.loads(Path(path).read_text())


def authenticate(policy, adapter, identity, contract):
    from multimodal_jev.context_policy_artifacts import read_execution_bundle, verify_native_adapter
    bundle=read_execution_bundle(policy,expected_policy_id=identity,
        expected_contract=contract,expected_base_id=contract['base_id'])
    verify_native_adapter(adapter,bundle)
    return bundle


def verify(package, digest):
    root=Path(package)
    if root.is_symlink() or sha(root/'PACKAGE.json')!=digest:
        raise ValueError('transfer source manifest differs')
    manifest=read(root/'PACKAGE.json');actual={}
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('transfer source symlink forbidden')
        if path.is_file() and path!=root/'PACKAGE.json':actual[str(path.relative_to(root))]=sha(path)
    if actual!=manifest['files']:raise ValueError('transfer source files differ')
    curve=read(root/'source-selection.json')
    if (manifest['status']!='prepared_source_pair_no_target_dispatch'
            or manifest['algorithm']!='grpo_v1' or curve['algorithm']!='grpo_v1'
            or not curve['complete'] or curve['selected_checkpoint'] is None
            or set(manifest['policies'])!={'initial','selected'}
            or manifest['target_tuning_allowed'] is not False or manifest['threshold']!=.5
            or sha(root/'input-audit.json')!=manifest['input_audit_sha256']
            or read(root/'input-audit.json')['manifest']['processor_sha256']!=manifest['processor_sha256']):
        raise ValueError('complete fixed GRPO source selection required')
    from multimodal_jev.cached_context_preparation import processor_cache_identity
    for role,entry in manifest['policies'].items():
        expected=curve['initial_policy_id'] if role=='initial' else curve['selected_checkpoint']['policy_id']
        if entry['policy_id']!=expected or entry['policy']!=role+'/policy' or entry['adapter']!=role+'/adapter':
            raise ValueError('source-only pair binding differs')
        authenticate(root/entry['policy'],root/entry['adapter'],expected,entry['contract'])
        if processor_cache_identity(root/entry['policy'])!=manifest['processor_sha256']:
            raise ValueError('transfer processor differs from audited cache')
    if manifest['policies']['initial']['contract']!=manifest['policies']['selected']['contract']:
        raise ValueError('same policy contract required')
    return manifest


def prepare(experiment_directory, input_audit, input_audit_sha256, output):
    experiment_directory=Path(experiment_directory);output=Path(output)
    if sha(input_audit)!=input_audit_sha256:raise ValueError('input audit identity differs')
    proof=read(input_audit);state=proof['state']
    if (state['Status']!='exited' or state['ExitCode']!=0 or state['OOMKilled']
            or proof['manifest']['recordings']!=3793 or proof['manifest']['decisions']!=31):
        raise ValueError('complete audited FSC inputs required')
    # Fixed primary algorithm: never choose GRPO versus RLOO using target scores.
    admitted=load_run(experiment_directory,'grpo_v1');curve=admitted['curve']
    points={entry['step']:entry for entry in curve['independently_audited_points']}
    plan=read(experiment_directory/'curve-spec.json')
    transports={entry['step']:read(entry['transport']) for entry in plan['evaluations']}
    choices=dict(initial=0,selected=curve['selected_checkpoint']['step'])
    policies={};sources={}
    from multimodal_jev.cached_context_preparation import processor_cache_identity
    for role,step in choices.items():
        entry=points[step];root=Path(entry['collection']);transport=transports[step]
        if output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve()):
            raise ValueError('independent transfer package required')
        policy=root/'evaluation/run/policy';adapter=root/'evaluation/run/adapter'
        manifest=read(bound(root,transport,'evaluation/run/policy/MANIFEST.json'))
        identity=entry['inference_policy_id'];contract=manifest['contract']
        expected=curve['initial_policy_id'] if role=='initial' else curve['selected_checkpoint']['policy_id']
        if (identity!=expected or manifest['policy_id']!=identity
                or contract['request_version']!='jev-context-exclusive_event-v1'
                or contract['token']['registration_status']!='admitted'):
            raise ValueError('source-selected admitted evaluated policy required')
        for folder in (policy,adapter):
            for path in folder.rglob('*'):
                if path.is_symlink():raise ValueError('source symlink forbidden')
                if path.is_file():bound(root,transport,str(path.relative_to(root)))
        authenticate(policy,adapter,identity,contract)
        if processor_cache_identity(policy)!=proof['manifest']['processor_sha256']:
            raise ValueError('source processor differs from existing FSC cache')
        policies[role]=dict(step=step,policy_id=identity,source_policy_id=entry['source_policy_id'],
            contract=contract,policy=role+'/policy',adapter=role+'/adapter')
        sources[role]=(policy,adapter)
    if output.resolve().is_relative_to(experiment_directory.resolve()) or experiment_directory.resolve().is_relative_to(output.resolve()):
        raise ValueError('independent transfer package required')
    from multimodal_jev.local_runtime_storage import check_paths
    check_paths({'fsc_source_package':output});output.mkdir(exist_ok=False)
    for role,(policy,adapter) in sources.items():
        shutil.copytree(policy,output/role/'policy');shutil.copytree(adapter,output/role/'adapter')
    (output/'source-selection.json').write_text(json.dumps(curve,indent=2)+'\n')
    shutil.copyfile(input_audit,output/'input-audit.json')
    if sha(output/'input-audit.json')!=input_audit_sha256:raise ValueError('input audit changed during copy')
    manifest=dict(status='prepared_source_pair_no_target_dispatch',algorithm='grpo_v1',
        policies=policies,processor_sha256=proof['manifest']['processor_sha256'],
        source_provenance=admitted['provenance'],input_audit_sha256=input_audit_sha256,
        target_tuning_allowed=False,threshold=.5,
        scope='Source-only chosen immutable weights and compatible processor; actual 31-span parity, '
              'target inference and independent scoring are not performed by preparation.',
        files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    (output/'PACKAGE.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return verify(output,sha(output/'PACKAGE.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('experiment-directory','input-audit','input-audit-sha256','output'):
        parser.add_argument('--'+name,required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())),indent=2))
