"""Reconstruct source learning/selection evidence before deploying target roles.

No target inputs/scores opened. Validate the actual selected policy tensors,
then export a self-contained source score bundle for the isolated client.
"""
import copy
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
from tools.reports.collect_context_macro_source_curve import reconstruct
from tools.reports.evaluate_context_transfer_macro import read_bound,source_binding
from tools.reports.prepare_context_macro_replay import write


def admit(c):
    source=read_bound(c['source_evaluation'])
    checked=reconstruct(c['source_curve_config'])
    plan=checked['plan'];curve=checked['curve']
    if (source['algorithm']!=plan['algorithm'] or source['experiment']!=plan['experiment']
            or source['initial_policy_id']!=plan['initial_policy_id']
            or source['selected_checkpoint']!=curve['selected_checkpoint']
            or read_bound(source['curve'])!=curve):
        raise ValueError('published source curve differs from actual independent reconstruction')
    published=[dict(step=item['step'],result=read_bound(item['result'])) for item in source['points']]
    if published!=checked['points']:
        raise ValueError('published source metrics differ from actual audited inference')
    audits=read_bound(source['source_audits'])
    expected=dict(source_plan=c['source_curve_config']['source_plan'],
        source_plan_sha256=c['source_curve_config']['source_plan_sha256'],
        training=checked['training'],evaluations=checked['proofs'],matched_controls=checked['controls'])
    if {k:v for k,v in audits.items() if k!='config_provenance'}!=expected:
        raise ValueError('published source provenance differs from original runtime evidence')
    binding=source_binding(c)
    points=[p for p in plan['checkpoints'] if p['step']==binding['step']]
    if len(points)!=1 or points[0]['policy_id']!=binding['policy_id']:
        raise ValueError('source role does not identify an audited policy')
    point=points[0]
    source_ops=plan['source_operations']
    training=json.loads((Path(source_ops['prepared'])/'package/training.json').read_text())
    bundle=read_execution_bundle(point['snapshot'],expected_policy_id=binding['policy_id'],
        expected_contract=training['source_contract'],expected_base_id=training['base_id'])
    verify_native_adapter(point['adapter'],bundle)
    if file_hash(Path(point['snapshot'])/'MANIFEST.json')!=point['manifest_sha256']:
        raise ValueError('source role policy changed after reconstruction')
    return dict(status='source_role_runtime_and_weights_admitted',binding=binding,point=point,
        source_operations=source_ops,training=training,controls=checked['controls'],
        source_evaluation=c['source_evaluation'],source_curve_config=c['source_curve_config'],
        source=source,curve=curve,points=published,audits=audits,target_inputs_opened=False)


def seal(admission,output,remote_root):
    """Only source scores/metadata; no training archives or target labels copied."""
    if admission['status']!='source_role_runtime_and_weights_admitted':
        raise ValueError('actual source role admission required')
    output=Path(output);remote=Path(remote_root)
    if not remote.is_absolute() or '..' in remote.parts:
        raise ValueError('absolute source bundle runtime path required')
    output.mkdir(parents=True,exist_ok=False)
    source=copy.deepcopy(admission['source'])
    def entry(name,value):
        write(output/name,value)
        return dict(path=str(remote/name),sha256=file_hash(output/name))
    source['points']=[dict(step=p['step'],result=entry(f'score-{p["step"]:04d}.json',p['result']))
                      for p in admission['points']]
    source['curve']=entry('curve.json',admission['curve'])
    source['source_audits']=entry('audits.json',admission['audits'])
    result=entry('source-evaluation.json',source)
    write(output/'ORIGIN.json',dict(source_evaluation=admission['source_evaluation'],
        source_curve_config=admission['source_curve_config'],binding=admission['binding'],
        point=admission['point'],target_inputs_opened=False))
    return result


def verify_bundle(root,remote_root,admission):
    """Rebind relocated JSON to re-audited source; accept only path substitutions."""
    root=Path(root);remote=Path(remote_root)
    def read(name):return json.loads((root/name).read_text())
    source=read('source-evaluation.json');expected=copy.deepcopy(admission['source'])
    names={'ORIGIN.json','source-evaluation.json','curve.json','audits.json'}
    def entry(name,value):
        names.add(name)
        if read(name)!=value:raise ValueError('source bundle content differs: '+name)
        return dict(path=str(remote/name),sha256=file_hash(root/name))
    expected['points']=[dict(step=p['step'],result=entry(f'score-{p["step"]:04d}.json',p['result']))
                        for p in admission['points']]
    expected['curve']=entry('curve.json',admission['curve'])
    expected['source_audits']=entry('audits.json',admission['audits'])
    if source!=expected or read('ORIGIN.json')!=dict(source_evaluation=admission['source_evaluation'],
        source_curve_config=admission['source_curve_config'],binding=admission['binding'],
        point=admission['point'],target_inputs_opened=False):
        raise ValueError('portable source role binding differs')
    if ({str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}!=names
            or any(p.is_symlink() for p in root.rglob('*'))):
        raise ValueError('exact portable source evidence inventory required')
    return dict(path=str(remote/'source-evaluation.json'),sha256=file_hash(root/'source-evaluation.json'))
