"""Authenticate a context policy and derive its native PEFT LoRA payload.

No weight merging or inference-runtime mutation. A serving owner must switch
the adapter, frozen marker row, head and processor as one quiescent policy.
"""
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import torch

from .baseline import file_hash
from .context_decision_head import ContextDecisionHead
from .context_policy_snapshot import _hash, _digest, _check_marker


def read_execution_bundle(path, *, expected_policy_id, expected_contract, expected_base_id):
    """Validate every asset/tensor before returning CPU execution components."""
    path = Path(path)
    if (not _digest(expected_policy_id) or not _digest(expected_base_id) or
            expected_contract.get('base_id') != expected_base_id or path.is_symlink()):
        raise ValueError('authenticated policy, contract and base required')
    manifest = json.loads((path/'MANIFEST.json').read_text())
    if set(manifest) != {'schema', 'contract', 'files', 'policy_id'}:
        raise ValueError('invalid joint manifest')
    payload = {k:v for k,v in manifest.items() if k != 'policy_id'}
    if (manifest['schema'] != 'context-policy-execution-v1' or
            manifest['policy_id'] != expected_policy_id or _hash(payload) != expected_policy_id or
            manifest['contract'] != expected_contract):
        raise ValueError('joint identity or execution contract differs')
    actual = {}
    for p in path.rglob('*'):
        if p.is_symlink():
            raise ValueError('policy symlinks forbidden')
        if p.is_file() and p != path/'MANIFEST.json':
            actual[str(p.relative_to(path))] = file_hash(p)
    if actual != manifest['files']:
        raise ValueError('joint asset integrity differs')
    contract = manifest['contract']
    if set(contract) != {'base_id','runtime_profile','request_version','branch','token','output','lora'}:
        raise ValueError('complete joint contract required')
    if any(type(contract[k]) is not str or not contract[k] for k in ('runtime_profile','request_version')):
        raise ValueError('explicit runtime and request identity required')
    token = contract['token']
    if (set(token) != {'id','text','row_trainable','registration_status'} or
            type(token['id']) is not int or token['id'] <= 0 or
            token['text'] != '<|jev_decision|>' or token['row_trainable'] is not False or
            token['registration_status'] not in ('test_only','admitted')):
        raise ValueError('registered frozen decision marker required')
    _check_marker(path/'processor', contract)
    if contract['output'] != dict(field='prob',activation='sigmoid',normalize_across_candidates=False):
        raise ValueError('applicability probability output required')
    branch = contract['branch']
    dimensions = {'hidden_size','num_heads','ffn_size','residual_scale'}
    semantics = dict(layers=1,position_encoding='inherited_backbone_states_no_extra_branch_encoding',
        dropout=0.,readout='decision_marker_itself',
        attention_scope='same_request_valid_causal_prefix_including_marker')
    if set(branch) != dimensions | semantics.keys() or any(branch[k] != v for k,v in semantics.items()):
        raise ValueError('supported shared history branch required')
    # Shape validation must not consume the caller's RNG or allocate a fresh
    # real-sized head before authenticating all of its tensors.
    with torch.device('meta'):
        head = ContextDecisionHead(**{k:branch[k] for k in dimensions})
    shapes = {'decision_head.'+k:tuple(v.shape) for k,v in head.state_dict().items()}
    shapes['decision_token_row'] = (branch['hidden_size'],)
    lora = contract['lora']
    if type(lora) is not dict or not lora:
        raise ValueError('explicit language LoRA required')
    for name, spec in lora.items():
        if (not re.fullmatch(r'language_model\.layers\.\d+\.self_attn\.[qv]_proj',name) or
                set(spec) != {'rank','alpha','scaling','fan_in_fan_out','active_adapter',
                              'a_shape','b_shape','dropout','use_dora'}):
            raise ValueError('explicit language q/v LoRA only')
        rank, alpha = spec['rank'], spec['alpha']
        if (type(rank) is not int or rank <= 0 or type(alpha) not in (int,float) or
                not 0 < alpha < float('inf') or spec['scaling'] != alpha/rank or
                spec['fan_in_fan_out'] is not False or spec['active_adapter'] != 'default' or
                spec['dropout'] != 0 or spec['use_dora'] is not False):
            raise ValueError('ordinary unmerged rank/alpha LoRA semantics required')
        a, b = spec['a_shape'], spec['b_shape']
        if (type(a) is not list or type(b) is not list or len(a) != 2 or len(b) != 2 or
                any(type(x) is not int or x <= 0 for x in a+b) or
                a != [rank,branch['hidden_size']] or b[1] != rank):
            raise ValueError('LoRA dimensions differ')
        shapes['backbone.'+name+'.lora_A.default.weight'] = tuple(a)
        shapes['backbone.'+name+'.lora_B.default.weight'] = tuple(b)
    tensors = torch.load(path/'execution.pt',map_location='cpu',weights_only=True)
    if type(tensors) is not dict or tensors.keys() != shapes.keys():
        raise ValueError('complete head, marker and LoRA tensor set required')
    for name, shape in shapes.items():
        t = tensors[name]
        if (not isinstance(t,torch.Tensor) or t.dtype != torch.bfloat16 or
                tuple(t.shape) != shape or not torch.isfinite(t).all()):
            raise ValueError('invalid execution tensor: '+name)
    return dict(policy_id=expected_policy_id,contract=contract,tensors=tensors,
                processor=str(path/'processor'))


def native_peft_payload(bundle):
    """Derive standard PEFT names preserving unmerged BF16 matrix bytes.

    Input must be read_execution_bundle's authenticated output. The existing
    native vLLM Qwen3-ASR mapper consumes model.language_model.* targets.
    Heterogeneous rank/alpha is rejected until a runtime path is tested.
    """
    lora = bundle['contract']['lora']
    settings = {(s['rank'],s['alpha']) for s in lora.values()}
    if len(settings) != 1:
        raise ValueError('uniform rank and alpha required by candidate runtime')
    rank, alpha = next(iter(settings))
    config = dict(peft_type='LORA',task_type='CAUSAL_LM',inference_mode=True,
        r=rank,lora_alpha=alpha,lora_dropout=0.,bias='none',fan_in_fan_out=False,
        use_rslora=False,use_dora=False,target_modules=sorted('model.'+n for n in lora))
    weights = {}
    for name in sorted(lora):
        for matrix in ('A','B'):
            source = f'backbone.{name}.lora_{matrix}.default.weight'
            dest = f'base_model.model.model.{name}.lora_{matrix}.weight'
            weights[dest] = bundle['tensors'][source].contiguous().clone()
    return config, weights


def export_native_adapter(destination, bundle):
    """Publish a derived immutable adapter with source-policy provenance.

    The joint snapshot remains authoritative for marker/head/processor. This
    directory alone is never a complete context policy.
    """
    from safetensors.torch import save_file
    from .local_runtime_storage import check_paths
    from .local_checkpoint import sync_directory
    destination = Path(destination)
    check_paths({'native_adapter':destination})
    if destination.exists():
        raise FileExistsError(destination)
    config, tensors = native_peft_payload(bundle)
    stage = Path(tempfile.mkdtemp(prefix='.'+destination.name+'-',dir=destination.parent))
    try:
        (stage/'adapter_config.json').write_text(json.dumps(config,sort_keys=True)+'\n')
        save_file(tensors,str(stage/'adapter_model.safetensors'))
        files = {p.name:file_hash(p) for p in stage.iterdir()}
        provenance = dict(schema='context-native-adapter-v1',policy_id=bundle['policy_id'],
                          files=files,complete_context_policy=False)
        (stage/'CONTEXT_POLICY.json').write_text(json.dumps(provenance,sort_keys=True)+'\n')
        for p in stage.iterdir():
            with p.open('rb') as stream:
                os.fchmod(stream.fileno(),0o644)
                os.fsync(stream.fileno())
        stage.chmod(0o755)
        sync_directory(stage)
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(stage,destination)
        sync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    return provenance


def verify_native_adapter(path, bundle):
    """Authenticate source binding and independently rederive every matrix."""
    from safetensors.torch import load_file
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.rglob('*')):
        raise ValueError('self-contained adapter required')
    names = {p.name for p in path.iterdir()}
    if names != {'adapter_config.json','adapter_model.safetensors','CONTEXT_POLICY.json'}:
        raise ValueError('complete derived adapter file set required')
    receipt = json.loads((path/'CONTEXT_POLICY.json').read_text())
    expected = dict(schema='context-native-adapter-v1',policy_id=bundle['policy_id'],
        files={name:file_hash(path/name) for name in ('adapter_config.json','adapter_model.safetensors')},
        complete_context_policy=False)
    if receipt != expected:
        raise ValueError('derived adapter provenance differs')
    config, tensors = native_peft_payload(bundle)
    actual = load_file(str(path/'adapter_model.safetensors'),device='cpu')
    if (json.loads((path/'adapter_config.json').read_text()) != config or actual.keys() != tensors.keys() or
            any(actual[k].dtype != tensors[k].dtype or actual[k].shape != tensors[k].shape or
                not torch.equal(actual[k],tensors[k]) for k in tensors)):
        raise ValueError('derived adapter tensors/config differ from joint policy')
    return receipt
