"""CPU acceptance for the 1.7B path on the same configs and algorithms as 0.6B.

Derives the large model's geometry from its published HF config, checks the new
experiment design against it, builds the shared decision head at hidden 2048 and
drives the real policy snapshot / native-adapter code with a stand-in backbone of
2048 width. Capacity numbers come from both checkpoints' safetensors headers; no
weight file is loaded, no HF model is constructed, no device is used and no
model is called.

The stand-in backbone exists because peft, transformers and safetensors are not
installed on this host. It exposes exactly the attributes ``lora_contract`` reads
and the q/v output widths the real projections have, so the snapshot roundtrip
and payload derivation run at 1.7B width; it does not prove that real PEFT
produces those attributes, and every step blocked by a missing package is listed
in ``not_exercised`` with the import that would be needed.

The verdict is deterministic and computed here. A non-empty ``failures`` list is
a rejected report, not a warning.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import platform
import shutil
import struct
import sys
import torch
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_decision_contract import DECISION_TOKEN
from multimodal_jev.context_model_profile import (
    build_head, from_hf, record, validate_against_config)
from multimodal_jev.context_policy_artifacts import native_peft_payload, read_execution_bundle
from multimodal_jev.context_policy_snapshot import (
    _parameters as execution_parameters, export_policy, load_policy, lora_contract)
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config

SCHEMA = 'jev-context-model-profile-1.7b-audit-v1'
LARGE_HIDDEN = 2048
# Only the model identity may differ between the two designs.
ALLOWED_DESIGN_DIFFERENCES = {'model', 'model_profile'}
MISSING = ('peft', 'transformers', 'safetensors', 'vllm')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def differences(left, right, prefix=''):
    """Dotted paths whose values differ between two JSON documents."""
    if type(left) is not dict or type(right) is not dict:
        return [] if left == right else [prefix.rstrip('.')]
    found = []
    for key in sorted(set(left) | set(right)):
        name = f'{prefix}{key}'
        if key not in left or key not in right:
            found.append(name)
        else:
            found.extend(differences(left[key], right[key], name + '.'))
    return found


def header(path):
    """Tensor name -> (dtype, shape) from a single-file safetensors header."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('ordinary safetensors file required')
    with path.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        if not 0 < size < 100 * 1024**2:
            raise ValueError('implausible safetensors header length')
        table = json.loads(stream.read(size))
    if type(table) is not dict:
        raise ValueError('object safetensors header required')
    return {name: (spec['dtype'], tuple(spec['shape']))
            for name, spec in table.items() if name != '__metadata__'}


def parameters(path):
    """Declared tensor and parameter counts, parsed without loading weights."""
    tensors = header(path)
    total = sum(math.prod(shape) for _, shape in tensors.values())
    return dict(tensors=len(tensors), parameters=total, bf16_bytes=total * 2,
                file_bytes=Path(path).stat().st_size,
                dtypes=sorted({dtype for dtype, _ in tensors.values()}))


def target_widths(profile):
    """Checkpoint q/v matrix shape per LoRA target, from the attention geometry."""
    q = profile.num_attention_heads * profile.head_dim
    v = profile.num_key_value_heads * profile.head_dim
    return {name: [q if '.q_proj' in name else v, profile.hidden_size]
            for name in profile.lora_targets()}


def checkpoint_targets(path, profile):
    """LoRA targets whose checkpoint matrix is present at the derived width."""
    tensors = header(path)
    expected = target_widths(profile)
    found, wrong = {}, {}
    for name, shape in expected.items():
        key = 'model.' + name + '.weight'
        if key not in tensors:
            continue
        actual = list(tensors[key][1])
        (found if actual == shape else wrong)[name] = actual
    return dict(present=found, unexpected_width=wrong, expected_counts=len(expected),
                q_width=profile.num_attention_heads * profile.head_dim,
                v_width=profile.num_key_value_heads * profile.head_dim)


class _LoRA(torch.nn.Module):
    """Stand-in exposing the attributes ``lora_contract`` reads from PEFT."""

    def __init__(self, hidden, rank, alpha, output):
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({'default': torch.nn.Linear(hidden, rank, bias=False)})
        self.lora_B = torch.nn.ModuleDict({'default': torch.nn.Linear(rank, output, bias=False)})
        self.lora_dropout = torch.nn.ModuleDict({'default': torch.nn.Dropout(0.0)})
        self.r = {'default': rank}
        self.lora_alpha = {'default': alpha}
        self.scaling = {'default': alpha / rank}
        self.active_adapters = ['default']
        self.disable_adapters = False
        self.merged = False
        self.use_dora = {'default': False}
        self.fan_in_fan_out = False


class _Backbone(torch.nn.Module):
    def __init__(self, profile, rank, alpha, rows):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(rows, profile.hidden_size,
                                               dtype=torch.bfloat16)
        self.embed_tokens.weight.requires_grad_(False)
        widths = target_widths(profile)
        self.language_model = torch.nn.Module()
        self.language_model.layers = torch.nn.ModuleList()
        for _ in range(profile.num_hidden_layers):
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            for name in ('q_proj', 'v_proj'):
                full = next(t for t in widths if t.endswith('self_attn.' + name))
                setattr(layer.self_attn, name,
                        _LoRA(profile.hidden_size, rank, alpha, widths[full][0]))
            self.language_model.layers.append(layer)

    def get_input_embeddings(self):
        return self.embed_tokens


class _Policy(torch.nn.Module):
    def __init__(self, profile, rank, alpha, token_id):
        super().__init__()
        self.decision_head = build_head(profile)
        self.backbone = _Backbone(profile, rank, alpha, token_id + 1)
        self.decision_token_id = token_id


def synthetic_contract(model, profile, rank, alpha, token_id):
    head = model.decision_head
    return dict(base_id=canonical(dict(profile=profile.geometry(), rank=rank, alpha=alpha)),
                runtime_profile='qwen3_asr_jev_context_v1',
                request_version='jev-context-request-v1',
                branch=dict(hidden_size=head.hidden_size, num_heads=head.num_heads,
                            ffn_size=head.ffn[0].out_features,
                            residual_scale=head.residual_scale, layers=1,
                            position_encoding='inherited_backbone_states_no_extra_branch_encoding',
                            dropout=0.0, readout='decision_marker_itself',
                            attention_scope='same_request_valid_causal_prefix_including_marker'),
                token=dict(id=token_id, text=DECISION_TOKEN, row_trainable=False,
                           registration_status='test_only'),
                output=dict(field='prob', activation='sigmoid',
                            normalize_across_candidates=False),
                lora=lora_contract(model.backbone))


def processor(directory, token_id):
    directory.mkdir(parents=True)
    (directory/'tokenizer.json').write_text(json.dumps(dict(added_tokens=[
        dict(id=token_id, content=DECISION_TOKEN, special=True)])))
    (directory/'tokenizer_config.json').write_text(json.dumps(dict(model_max_length=1)))
    return directory


def roundtrip(root, profile, rank, alpha, token_id, seed):
    """Export a 2048-wide policy, reload it into a differently initialised copy."""
    torch.manual_seed(seed)
    source = _Policy(profile, rank, alpha, token_id).to(torch.bfloat16)
    contract = synthetic_contract(source, profile, rank, alpha, token_id)
    snapshot = root/'snapshot'
    identity = export_policy(snapshot, source, processor(root/'processor', token_id), contract)
    saved = {name: value.detach().clone()
             for name, value in execution_parameters(source).items()}
    torch.manual_seed(seed + 1)
    target = _Policy(profile, rank, alpha, token_id).to(torch.bfloat16)
    loaded = load_policy(snapshot, target, expected_policy_id=identity,
                         expected_contract=contract, expected_base_id=contract['base_id'])
    restored = execution_parameters(target)
    different = sorted(name for name in saved
                       if not torch.equal(saved[name], restored[name].detach()))
    bundle = read_execution_bundle(snapshot, expected_policy_id=identity,
                                   expected_contract=contract,
                                   expected_base_id=contract['base_id'])
    config, weights = native_peft_payload(bundle)
    by_projection = {}
    for name in config['target_modules']:
        source = name.replace('model.', 'base_model.model.model.', 1)
        for matrix in ('A', 'B'):
            by_projection.setdefault(name.rsplit('.', 1)[-1], {})[matrix] = list(
                weights[source + f'.lora_{matrix}.weight'].shape)
    execution_sha256 = file_hash(snapshot/'execution.pt')
    return dict(policy_id=identity,
                exported_tensors=len(saved), restored_tensors=loaded['restored_tensors'],
                restored_optimizer=loaded['optimizer_restored'],
                byte_identical=not different, differing_tensors=different,
                head_width=contract['branch']['hidden_size'],
                lora_targets=len(contract['lora']),
                matrix_shapes=sorted({(tuple(spec['a_shape']), tuple(spec['b_shape']))
                                      for spec in contract['lora'].values()}),
                payload_targets=len(config['target_modules']),
                payload_tensors=len(weights),
                payload_by_projection=by_projection,
                payload_rank=config['r'], payload_alpha=config['lora_alpha'],
                snapshot_bytes=sum(path.stat().st_size for path in snapshot.rglob('*')
                                   if path.is_file()),
                execution_sha256=execution_sha256)


def environment():
    result = dict(python=sys.version.split()[0], host=platform.node(),
                  machine=platform.machine(), torch=torch.__version__,
                  torch_threads=torch.get_num_threads())
    for name in MISSING:
        result[name] = importlib.util.find_spec(name) is not None
    return result


def not_exercised(host):
    """Steps this host cannot run, each with the import that would be needed."""
    steps = []
    if not host['peft']:
        steps.append(dict(step='sft.initialize_adapter adapter construction',
                          blocked_by='peft is not installed',
                          covered_instead='LoRA target names and q/v matrix widths derived '
                                          'from geometry and read from the checkpoint header'))
    if not host['transformers']:
        steps.append(dict(step='real HF Qwen3-ASR 1.7B weight load and forward pass',
                          blocked_by='transformers is not installed',
                          covered_instead='contract, marker and LoRA shape validation at '
                                          'hidden 2048 with a stand-in backbone'))
    if not host['safetensors']:
        steps.append(dict(step='export_native_adapter safetensors write',
                          blocked_by='safetensors is not installed',
                          covered_instead='native_peft_payload names, shapes and rank'))
    if not host['vllm']:
        steps.append(dict(step='vLLM engine load, policy switch and numeric parity',
                          blocked_by='vllm is not installed',
                          covered_instead='read_execution_bundle authentication of a 2048 contract'))
    steps.append(dict(step='four-device capacity, microbatch and packed-row kernel parity',
                      blocked_by='NPU dispatch is out of scope this round',
                      covered_instead='declared parameter and BF16 byte counts'))
    return steps


def run(c, provenance):
    if c['schema'] != SCHEMA:
        raise ValueError('explicit 1.7B profile audit schema required')
    models = {entry['role']: entry for entry in c['models']}
    if set(models) != {'small', 'large'}:
        raise ValueError('exactly one small and one large model required')
    out = Path(c['output'])
    out.mkdir(parents=True, exist_ok=False)
    commit_json(out/'config.json', c)
    commit_json(out/'provenance.json', provenance)
    failures = []
    host = environment()

    profiles, rows, designs = {}, {}, {}
    for role, entry in models.items():
        profile = from_hf(entry['path'])
        profiles[role] = profile
        designs[role] = json.loads(Path(entry['design']).read_text())
        rows[role] = dict(record(profile), identity=profile.identity(),
                          path=entry['path'], design=entry['design'],
                          design_sha256=file_hash(entry['design']))
        if profile.hidden_size != entry['expected_hidden_size']:
            failures.append(role + ' hidden size differs from the declared expectation')

    small, large = profiles['small'], profiles['large']
    if large.hidden_size != LARGE_HIDDEN:
        failures.append('large model is not hidden 2048')
    if small.lora_targets() != large.lora_targets():
        failures.append('LoRA target sets differ between model sizes')
    targets = large.lora_targets()
    if len(targets) != 56:
        failures.append(f'shared LoRA target count is {len(targets)}, not 56')
    target_sha256 = canonical(targets)

    # Same schema, same algorithms: only the model identity may differ.
    design_diff = differences(designs['small'], designs['large'])
    if not set(design_diff) <= ALLOWED_DESIGN_DIFFERENCES:
        failures.append('designs differ outside model identity: ' + ', '.join(design_diff))
    if designs['large'].get('training_dispatch_allowed') is not False:
        failures.append('large design does not refuse training dispatch')
    if designs['large'].get('model_profile') != large.geometry():
        failures.append('large design does not declare the derived 1.7B geometry')

    consistency = {}
    for role, profile in profiles.items():
        consistency[role] = validate_against_config(profile, designs[role])
    try:
        validate_against_config(small, designs['large'])
        failures.append('small profile passes the 1.7B design: geometry is not pinned')
    except ValueError as error:
        consistency['small_against_large_design'] = 'refused: ' + str(error)
    # The 0.6B design declares no model_profile block, so the 1.7B profile passes
    # its target check; that is the shared-LoRA-block fact, recorded, not a failure.
    consistency['large_against_small_design'] = validate_against_config(
        large, designs['small'])['status']

    # Head construction and a BF16 forward pass at 1.7B width, no backbone weights.
    head = build_head(large).to(torch.bfloat16)
    generator = torch.Generator().manual_seed(c['seed'])
    hidden = torch.randn(2, 6, large.hidden_size, dtype=torch.bfloat16, generator=generator)
    positions = torch.tensor([[3], [5]])
    valid = torch.ones(2, 6, dtype=torch.bool)
    with torch.no_grad():
        probability = head(hidden, positions, valid)
    head_report = dict(dimensions=large.decision_head(),
                       parameters=sum(p.numel() for p in head.parameters()),
                       output_shape=list(probability.shape),
                       output_dtype=str(probability.dtype),
                       finite=bool(torch.isfinite(probability).all()),
                       in_unit_interval=bool(((probability >= 0) & (probability <= 1)).all()))
    if head_report['dimensions'] != dict(hidden_size=LARGE_HIDDEN, num_heads=8,
                                         ffn_size=LARGE_HIDDEN, residual_scale=0.1):
        failures.append('large decision head dimensions are not (2048, 8, 2048, 0.1)')
    if list(probability.shape) != [2, 1] or not head_report['finite']:
        failures.append('large decision head forward pass is not finite [2, 1]')
    if not head_report['in_unit_interval']:
        failures.append('large decision head probability is outside [0, 1]')

    # Capacity: parsed from both checkpoint headers, never loaded.
    capacity = {}
    for role, profile in profiles.items():
        weights = Path(models[role]['path'])/'model.safetensors'
        found = checkpoint_targets(weights, profile)
        capacity[role] = dict(parameters(weights), partial_file_hash='not computed',
                              checkpoint_targets=found)
        if len(found['present']) != len(targets):
            failures.append(role + ' checkpoint lacks a declared q/v target')
        if found['unexpected_width']:
            failures.append(role + ' checkpoint q/v matrices differ from derived width')
    lora_parameters = 2 * c['lora']['rank'] * LARGE_HIDDEN * len(targets)
    head_parameters = head_report['parameters']
    capacity['trainable'] = dict(lora_parameters=lora_parameters,
                                 head_parameters=head_parameters,
                                 total=lora_parameters + head_parameters,
                                 bf16_bytes=(lora_parameters + head_parameters) * 2)
    capacity['parameter_ratio'] = round(capacity['large']['parameters']
                                        / capacity['small']['parameters'], 4)

    workspace = out/'workspace'
    workspace.mkdir(parents=True, exist_ok=False)
    snapshot = None
    try:
        snapshot = roundtrip(workspace, large, c['lora']['rank'], c['lora']['alpha'],
                             c['lora']['token_id'], c['lora']['seed'])
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        removed = not workspace.exists()
    snapshot['snapshot_removed'] = removed
    if not snapshot['byte_identical']:
        failures.append('snapshot reload did not restore every tensor byte for byte')
    if snapshot['restored_tensors'] != snapshot['exported_tensors'] + 1:
        failures.append('snapshot reload restored a different tensor count')
    if snapshot['payload_targets'] != 56 or snapshot['payload_tensors'] != 112:
        failures.append('native PEFT payload is not 56 targets x 2 matrices')
    widths = target_widths(large)
    expected_shapes = {((c['lora']['rank'], LARGE_HIDDEN), (widths[targets[0]][0], c['lora']['rank'])),
                       ((c['lora']['rank'], LARGE_HIDDEN), (widths[targets[1]][0], c['lora']['rank']))}
    if {tuple(shape) for shape in snapshot['matrix_shapes']} != expected_shapes:
        failures.append('LoRA matrix shapes are not rank x 2048 with q/v output rows')
    if snapshot['payload_by_projection'] != dict(
            q_proj=dict(A=[c['lora']['rank'], LARGE_HIDDEN],
                        B=[widths[targets[0]][0], c['lora']['rank']]),
            v_proj=dict(A=[c['lora']['rank'], LARGE_HIDDEN],
                        B=[widths[targets[1]][0], c['lora']['rank']])):
        failures.append('native PEFT payload matrix shapes differ from the derived q/v widths')

    report = dict(
        schema=SCHEMA, status='rejected' if failures else 'accepted',
        model_calls=0, npu_executed=False, failures=failures,
        environment=host, implementation_sha256=file_hash(__file__),
        provenance=provenance,
        designs=dict(differences=design_diff, allowed=sorted(ALLOWED_DESIGN_DIFFERENCES),
                     small_sha256=rows['small']['design_sha256'],
                     large_sha256=rows['large']['design_sha256']),
        profiles=rows, design_consistency=consistency,
        shared_lora_targets=dict(count=len(targets), sha256=target_sha256),
        decision_head=head_report, policy_snapshot=snapshot, capacity=capacity,
        not_exercised=not_exercised(host),
        pending=['1.7B four-device capacity and activation-memory acceptance',
                 '1.7B NPU numerical parity for the shared head',
                 'real PEFT adapter construction and vLLM engine parity'])
    commit_json(out/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    args = parser.parse_args()
    c, provenance = load_config(args.config, args.local)
    report = run(c, provenance)
    print(json.dumps(dict(status=report['status'], failures=report['failures'],
                          parameter_ratio=report['capacity']['parameter_ratio'],
                          policy_id=report['policy_snapshot']['policy_id']), indent=2))
    if report['status'] != 'accepted':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
