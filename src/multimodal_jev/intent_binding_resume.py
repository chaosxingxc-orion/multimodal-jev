"""Fixed source/view budgets and artifact lineage for binding RL continuation.

Exact Docker completion is the dispatch controller's responsibility. Restoring
actual FP32 tensors remains the rank worker's responsibility; neither is proven
by matching JSON receipts here.
"""
import json
from pathlib import Path
from .baseline import file_hash
from .intent_binding_inputs import load,select
from .intent_optimizer import optimizer_kwargs
from .span_candidate_intervention import VARIANTS

SCHEMA='intent-binding-finite-dp4-v1'


def validate_config(config):
    if config.get('schema')!=SCHEMA:raise ValueError('explicit binding training schema required')
    for key,value in (('source_batch',8),('views_per_source',4),('global_requests',32),('world_size',4)):
        if type(config.get(key)) is not int or config[key]!=value:
            raise ValueError('fixed binding source/view layout required: '+key)
    if (config.get('variants')!=list(VARIANTS) or config.get('arm') not in ('D','Q','P')
            or config.get('lora_execution_dtype')!='bfloat16'
            or config.get('runtime_profile')!='qwen3-asr-bf16-hf-order-short-audio-v2'
            or type(config.get('seed')) is not int
            or type(config.get('step')) is not int or config['step']<0):
        raise ValueError('fixed variants, arm, BF16 profile and update cursor required')
    for key in ('binding_release','binding_manifest_sha256','model_path','initial_adapter','acceptance_root','image'):
        if not isinstance(config.get(key),str) or not config[key]:raise ValueError('explicit binding identity required: '+key)
    for key in ('lora_rank','lora_alpha'):
        if type(config.get(key)) is not int or config[key]<1:raise ValueError('positive fixed LoRA layout required')
    optimizer_kwargs(config)
    if config.get('initial_optimizer_checkpoint'):raise ValueError('no implicit optimizer reset or foreign checkpoint')
    if config['step']==0:
        if config.get('previous_root') or config.get('previous_suite'):raise ValueError('first binding step requires fresh optimizer')
    elif not config.get('previous_root') or not config.get('previous_suite'):
        raise ValueError('binding continuation requires accepted predecessor')


def validate_lineage(previous,current):
    validate_config(previous);validate_config(current)
    for key in ('schema','binding_release','binding_manifest_sha256','model_path','arm','source_batch',
                'views_per_source','global_requests','world_size','variants','seed','lora_rank','lora_alpha',
                'lora_execution_dtype','runtime_profile','image'):
        if previous[key]!=current[key]:raise ValueError('binding continuation changed: '+key)
    if optimizer_kwargs(previous)!=optimizer_kwargs(current):raise ValueError('binding optimizer changed')
    if current['step']!=previous['step']+1:raise ValueError('exactly one update advancement required')
    if (Path(current['initial_adapter'])!=Path(current['previous_root'])/'run/adapter'
            or Path(current['acceptance_root'])!=Path(current['previous_suite'])):
        raise ValueError('accepted binding predecessor adapter and policy required')


def expected_selection(config):
    validate_config(config)
    rows,_=load(config['binding_release'],config['binding_manifest_sha256'],config['arm'])
    chosen,selection=select(rows,config['seed'],config['step'])
    return chosen,dict(selection,step=config['step'],seed=config['seed'],arm=config['arm'],
        binding_manifest_sha256=config['binding_manifest_sha256'])


def authenticate(config):
    """Require binding-specific acceptance and exact complete shard file custody."""
    validate_config(config)
    if config['step']==0:raise ValueError('no predecessor exists at binding step zero')
    root=Path(config['previous_root']);suite=Path(config['previous_suite'])
    def pinned(path):
        if config['pins'].get(str(path))!=file_hash(path):raise ValueError('binding predecessor unpinned or changed: '+str(path))
        return json.loads(path.read_text())
    previous=pinned(root/'config.json');validate_lineage(previous,config)
    accepted=pinned(suite/'ACCEPTED.json');verified=pinned(suite/'verified.json')
    if pinned(suite/'controller-terminal.json')['exit_code']!=0:raise ValueError('binding acceptance failed')
    gate=pinned(suite/'config.json')
    if (gate.get('training_root')!=str(root) or gate['adapter']!=config['initial_adapter']
            or gate['model']!=config['model_path'] or gate['runtime_profile']!=config['runtime_profile']
            or gate['population']!=dict(recordings=89,requests=712)):
        raise ValueError('binding suite lineage differs')
    if (accepted.get('passed') is not True or verified.get('passed') is not True
            or verified.get('schema')!='intent-binding-update-verified-v1'
            or accepted['verified_sha256']!=file_hash(suite/'verified.json')
            or verified['root']!=str(root) or verified['suite']!=str(suite)
            or verified['updates']!=config['step'] or verified['arm']!=config['arm']
            or verified['binding_manifest_sha256']!=config['binding_manifest_sha256']
            or verified['full_gate_requests']!=712 or verified['source_uses']!=8 or verified['view_uses']!=32
            or verified.get('exact_export_reload') is not True
            or accepted['adapter_id']!=verified['adapter_id'] or gate['adapter_id']!=verified['adapter_id']):
        raise ValueError('binding accepted update differs')
    _,expected=expected_selection(previous)
    selection=pinned(root/'run/selection.json')
    if any(selection.get(k)!=v for k,v in expected.items()):raise ValueError('binding predecessor source/view schedule differs')
    required={str(root/'run/selection.json')}
    for rank in range(4):
        required.update(str(root/'run'/f'rank-{rank}'/name) for name in
            ('optimizer-local.pt','result.json','terminal.json','selection.json','before.json','after.json'))
    if not required<=verified['files'].keys():raise ValueError('all four rank states and source/view evidence required')
    for path,digest in verified['files'].items():
        if config['pins'].get(path)!=digest or file_hash(path)!=digest:
            raise ValueError('binding verified state unpinned or changed')
    # Recompute the complete transformed gate, never trust only a passed receipt.
    from .intent_binding_gate import authenticate_artifacts
    policy_id=gate['base_id']+':'+verified['adapter_id']
    _,_,audit=authenticate_artifacts(config,policy_id)
    return dict(previous_root=str(root),previous_suite=str(suite),previous_updates=config['step'],
        adapter_id=verified['adapter_id'],verified_sha256=file_hash(suite/'verified.json'),
        source_uses=8,view_uses=32,gate=audit)
