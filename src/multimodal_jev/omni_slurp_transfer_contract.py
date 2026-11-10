"""Full-task shard admission under a frozen FSC-selected transfer protocol."""
import json
from pathlib import Path

from .baseline import file_hash
from .omni_slurp_contract import validate_runtime
from .slurp_jev_full import expand_bundle
from .slurp_jev_shards import load_shard

RUNTIME_KEYS=('audio_model','dtype','seed','max_context_tokens','max_new_tokens',
              'sampling','route','model_family','parameters_frozen','task_format','audio_condition')

TRAIN_REQUEST_SHA='8fff6d3097fb9e9ada62110ae0cdf362e39104fe73456f8e964f035f5afc763a'
PARENT_PROMPT_SHA='ab9c559716d480775dc93bbfc3f62ac5c7e1a2aa7c95a8c6b10d72b61ad982a2'


def select_transfer_train(cfg,rows):
    """Exercise the exact new engine on the original complete TRAIN probe."""
    validate_runtime(cfg)
    if (cfg['split']!='train' or cfg['evaluation_phase']!='slurp-jev-transfer-train'
            or cfg['task_format']!='jev' or cfg['audio_condition']!='actual'
            or cfg['max_new_tokens']!=128 or cfg['parameters_frozen'] is not True
            or cfg['source_groups']!=['slurp:12507']):
        raise ValueError('fixed original TRAIN transfer validation required')
    path=Path(cfg['requests'])
    if (file_hash(path)!=TRAIN_REQUEST_SHA
            or file_hash(cfg['instruction'])!=PARENT_PROMPT_SHA):
        raise ValueError('original TRAIN requests and FSC parent prompt required')
    expected=[json.loads(line) for line in path.open()]
    if (rows!=expected or len(rows)!=336 or len({r['id'] for r in rows})!=336
            or len({r['id'].rsplit('::',1)[0] for r in rows})!=2
            or any(r['split']!='train' or r['source_group']!='slurp:12507' for r in rows)):
        raise ValueError('all168 tasks on both original TRAIN recordings required')
    return rows


def select_transfer_shard(cfg,rows):
    validate_runtime(cfg)
    if (cfg['split']!='test' or cfg['evaluation_phase']!='slurp-jev-transfer'
            or cfg['task_format']!='jev' or cfg['audio_condition']!='actual'
            or cfg['max_new_tokens']!=128 or cfg['parameters_frozen'] is not True):
        raise ValueError('frozen full-task TEST transfer shard required')
    path=Path(cfg['transfer_protocol'])
    if file_hash(path)!=cfg['transfer_protocol_sha256']:raise ValueError('transfer protocol changed')
    protocol=json.loads(path.read_text())
    if (protocol['schema']!='jev-omni-slurp-transfer-v1'
            or {k:cfg[k] for k in RUNTIME_KEYS}!=protocol['runtime']
            or protocol['slurp_feedback_used_for_prompt_optimization'] is not False):
        raise ValueError('registered transfer runtime required')
    for name in ('selection_result','selection_protocol','shard_plan'):
        if file_hash(protocol[name])!=protocol[name+'_sha256']:
            raise ValueError('transfer source changed: '+name)
    selected=json.loads(Path(protocol['selection_result']).read_text())
    prior=json.loads(Path(protocol['selection_protocol']).read_text())
    winner=selected['selected_id'];prompts={'parent':prior['prompts']['parent']}
    if winner!='parent':prompts[winner]=prior['prompts'][winner]
    if (selected['protocol_sha256']!=protocol['selection_protocol_sha256']
            or selected['test_feedback_used'] is not False or selected['weight_updates']!=0
            or selected['selected_instruction_sha256']!=prompts[winner]
            or protocol['prompts']!=prompts or cfg['prompt_id'] not in prompts
            or file_hash(cfg['instruction'])!=prompts[cfg['prompt_id']]
            or file_hash(cfg['asset_audit'])!=protocol['asset_audit_sha256']):
        raise ValueError('parent or final selected unchanged prompt required')
    plan=json.loads(Path(protocol['shard_plan']).read_text())
    if (plan['split']!='test' or plan['recordings']!=13078 or plan['source_groups']!=2974
            or plan['tasks_per_recording']!=168 or plan['decisions']!=2197104):
        raise ValueError('complete SLURP-Jev TEST plan required')
    bundles,bank=load_shard(plan,cfg['shard_id'])
    expected=[r for b in bundles for r in expand_bundle(b,bank)]
    requests=Path(cfg['requests']);manifest=json.loads((requests.parent/'MANIFEST.json').read_text())
    if (file_hash(requests)!=manifest['requests_sha256']
            or manifest['shard_plan_sha256']!=protocol['shard_plan_sha256']
            or manifest['shard_id']!=cfg['shard_id'] or manifest['split']!='test'
            or manifest['source_release']!=plan['release']
            or manifest['source_manifest_sha256']!=plan['source_manifest_sha256']
            or manifest['recordings']!=len(bundles) or manifest['requests']!=len(expected)
            or manifest['source_groups']!=sorted({b['source_group'] for b in bundles})
            or cfg['source_groups']!=manifest['source_groups'] or rows!=expected):
        raise ValueError('complete registered requests required; no task filtering')
    return rows
