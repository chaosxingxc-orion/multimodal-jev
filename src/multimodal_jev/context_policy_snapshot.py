"""Version the entire context-head inference policy, not just its LoRA adapter.

Stores BF16 execution tensors, marker row, processor assets and model contract.
FP32 optimizer/master recovery is a separate training-checkpoint obligation.
No vLLM loader or task efficacy is implied by a successful local roundtrip.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch

from .baseline import file_hash
from .context_decision_contract import DECISION_TOKEN
from .local_runtime_storage import check_paths
from .local_checkpoint import sync_directory


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _digest(value):
    return (type(value) is str and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value))


def lora_contract(backbone):
    """Bind scaling and active adapter semantics in addition to matrix bytes."""
    result = {}
    for name, module in backbone.named_modules():
        if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
            continue
        if (set(module.lora_A) != {'default'} or set(module.lora_B) != {'default'} or
                list(module.active_adapters) != ['default'] or module.disable_adapters or
                module.merged or module.use_dora.get('default', False) or
                module.lora_dropout['default'].training and
                getattr(module.lora_dropout['default'], 'p', 0.) != 0.):
            raise ValueError('single active unmerged default LoRA without DoRA/dropout required')
        if getattr(module.lora_dropout['default'], 'p', 0.) != 0.:
            raise ValueError('zero LoRA dropout required')
        a, b = module.lora_A['default'], module.lora_B['default']
        rank = module.r['default']
        alpha, scale = float(module.lora_alpha['default']), float(module.scaling['default'])
        if (type(rank) is not int or rank <= 0 or a.weight.shape[0] != rank or
                b.weight.shape[1] != rank or a.bias is not None or b.bias is not None or
                not 0 < alpha < float('inf') or not 0 < scale < float('inf')):
            raise ValueError('finite positive LoRA rank/scaling and bias-free matrices required')
        result[name] = dict(rank=rank, alpha=alpha, scaling=scale,
            fan_in_fan_out=bool(module.fan_in_fan_out), active_adapter='default',
            a_shape=list(a.weight.shape), b_shape=list(b.weight.shape), dropout=0., use_dora=False)
    if not result:
        raise ValueError('explicit LoRA execution modules required')
    return result


def _contract(model, contract):
    required = {'base_id', 'runtime_profile', 'request_version', 'branch', 'token', 'output', 'lora'}
    if set(contract) != required or not _digest(contract['base_id']):
        raise ValueError('complete versioned policy contract required')
    if any(type(contract[k]) is not str or not contract[k] for k in ('runtime_profile', 'request_version')):
        raise ValueError('explicit runtime and input contract versions required')
    h = model.decision_head
    branch = dict(hidden_size=h.hidden_size, num_heads=h.num_heads,
                  ffn_size=h.ffn[0].out_features, residual_scale=h.residual_scale,
                  layers=1, position_encoding='inherited_backbone_states_no_extra_branch_encoding',
                  dropout=0.0, readout='decision_marker_itself',
                  attention_scope='same_request_valid_causal_prefix_including_marker')
    if contract['branch'] != branch:
        raise ValueError('head architecture differs from policy contract')
    if contract['lora'] != lora_contract(model.backbone):
        raise ValueError('LoRA execution configuration differs from policy contract')
    token = contract['token']
    if (set(token) != {'id', 'text', 'row_trainable', 'registration_status'} or
            type(token['id']) is not int or token['id'] != model.decision_token_id or
            token['text'] != DECISION_TOKEN or token['row_trainable'] is not False or
            token['registration_status'] not in ('test_only', 'admitted')):
        raise ValueError('explicit frozen marker-row contract required')
    if contract['output'] != dict(field='prob', activation='sigmoid', normalize_across_candidates=False):
        raise ValueError('independent applicability probability contract required')
    embedding = model.backbone.get_input_embeddings().weight
    if (embedding.requires_grad or embedding.dtype != torch.bfloat16 or
            not 0 <= token['id'] < embedding.shape[0] or embedding.shape[1] != h.hidden_size):
        raise ValueError('BF16 frozen embedding with valid decision row required')
    if not torch.isfinite(embedding[token['id']]).all():
        raise ValueError('finite decision embedding row required')


def _parameters(model):
    selected = {}
    lora = 0
    for name, p in model.named_parameters():
        allowed = name.startswith('decision_head.') or (name.startswith('backbone.') and
                  ('.lora_A.' in name or '.lora_B.' in name))
        if allowed:
            if p.dtype != torch.bfloat16 or not torch.isfinite(p).all():
                raise ValueError('finite BF16 execution head/LoRA required')
            selected[name] = p
            lora += name.startswith('backbone.')
        elif p.requires_grad:
            raise ValueError('unexported trainable parameter: '+name)
    if not lora:
        raise ValueError('backbone LoRA required')
    return selected


def _processor_files(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('processor symlinks are not self-contained')
        if path.is_file():
            files[str(path.relative_to(root))] = file_hash(path)
    if not {'tokenizer.json', 'tokenizer_config.json'} <= files.keys():
        raise ValueError('complete tokenizer assets required')
    return files


def _check_marker(processor, contract):
    data = json.loads((Path(processor)/'tokenizer.json').read_text())
    entries = [v for v in data['added_tokens'] if v['content'] == DECISION_TOKEN]
    if (len(entries) != 1 or entries[0]['id'] != contract['token']['id'] or
            entries[0]['special'] is not True):
        raise ValueError('processor marker differs from policy')


def export_policy(destination, model, processor, contract):
    """Immutable local execution snapshot. Caller verifies underlying base ID."""
    destination = Path(destination)
    _contract(model, contract)
    params = _parameters(model)
    processor = Path(processor)
    _processor_files(processor)
    _check_marker(processor, contract)
    check_paths({'context_policy':destination})
    if destination.exists():
        raise FileExistsError(destination)
    stage = Path(tempfile.mkdtemp(prefix='.'+destination.name+'-', dir=destination.parent))
    try:
        shutil.copytree(processor, stage/'processor')
        tensors = {k:p.detach().cpu().clone() for k,p in params.items()}
        tensors['decision_token_row'] = model.backbone.get_input_embeddings().weight[
            model.decision_token_id].detach().cpu().clone()
        torch.save(tensors, stage/'execution.pt')
        files = {'execution.pt':file_hash(stage/'execution.pt'),
                 **{'processor/'+k:v for k,v in _processor_files(stage/'processor').items()}}
        payload = dict(schema='context-policy-execution-v1', contract=contract, files=files)
        identity = _hash(payload)
        (stage/'MANIFEST.json').write_text(json.dumps(dict(payload,policy_id=identity),indent=2)+'\n')
        for path in stage.rglob('*'):
            if path.is_file():
                with path.open('rb') as stream:
                    os.fchmod(stream.fileno(),0o644)
                    os.fsync(stream.fileno())
        for path in sorted((p for p in stage.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
            path.chmod(0o755)
            sync_directory(path)
        stage.chmod(0o755)
        sync_directory(stage)
        # Destination is unique per policy; do not reuse published directories.
        os.rename(stage,destination)
        sync_directory(destination.parent)
        return identity
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise


def load_policy(snapshot, model, *, expected_policy_id, expected_contract, expected_base_id):
    """Validate all assets before copying any execution parameters into model.

Caller must authenticate the actually loaded base against expected_base_id.
Returns the processor directory, which the backend must load for this policy.
This does not restore optimizer state or prove actor/vLLM numeric parity.
"""
    if not _digest(expected_policy_id) or expected_contract['base_id'] != expected_base_id:
        raise ValueError('authenticated expected policy and base required')
    _contract(model, expected_contract)
    snapshot = Path(snapshot)
    manifest = json.loads((snapshot/'MANIFEST.json').read_text())
    if set(manifest) != {'schema','contract','files','policy_id'}:
        raise ValueError('invalid policy manifest')
    payload = {k:v for k,v in manifest.items() if k != 'policy_id'}
    if (manifest['schema'] != 'context-policy-execution-v1' or
            manifest['policy_id'] != expected_policy_id or _hash(payload) != expected_policy_id or
            manifest['contract'] != expected_contract):
        raise ValueError('policy identity or contract differs')
    actual = {}
    for path in snapshot.rglob('*'):
        if path.is_symlink():
            raise ValueError('snapshot symlink forbidden')
        if path.is_file() and path != snapshot/'MANIFEST.json':
            actual[str(path.relative_to(snapshot))] = file_hash(path)
    if actual != manifest['files']:
        raise ValueError('policy artifact integrity differs')
    _check_marker(snapshot/'processor',expected_contract)
    saved = torch.load(snapshot/'execution.pt',map_location='cpu',weights_only=True)
    params = _parameters(model)
    params['decision_token_row'] = model.backbone.get_input_embeddings().weight[model.decision_token_id]
    if saved.keys() != params.keys():
        raise ValueError('complete head, LoRA and marker tensors required')
    for name,p in params.items():
        t = saved[name]
        if not isinstance(t,torch.Tensor) or t.dtype != p.dtype or t.shape != p.shape or not torch.isfinite(t).all():
            raise ValueError('execution tensor mismatch: '+name)
    with torch.no_grad():
        for name,p in params.items():
            p.copy_(saved[name])
    return dict(policy_id=expected_policy_id,processor=str(snapshot/'processor'),
                restored_tensors=len(params),optimizer_restored=False,
                training_dispatch_allowed=False)
