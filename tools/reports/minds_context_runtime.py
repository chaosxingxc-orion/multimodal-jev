"""Label-free MINDS native-descriptor transfer inference on the frozen 14-candidate interface.

Reuse the accepted processor cache and the existing BF16 actor/vLLM runtime.
No scoring, source checkpoint selection, decoding, training or device launcher.
Source packaging and independent runtime collection remain separate obligations.
"""
import argparse
import json
import math
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.slurp_training import content_hash
from tools.jev.infer_fsc_context_transfer import read_bound, validate_source
from tools.reports.preflight_minds_context_inputs import authenticate
from tools.reports.prepare_minds_context_inputs import reconstruct, definitions_for


def load_inputs(config):
    """Bind full preparation, terminal collection and independent cache audit."""
    rows, definitions = authenticate(config)
    protocol = read_bound(config['protocol'])
    input_manifest = json.loads((Path(config['inputs'])/'MANIFEST.json').read_text())
    if (definitions != definitions_for(protocol)
            or config['protocol']['sha256'] != input_manifest['protocol']['sha256']):
        raise ValueError('Frozen native-label descriptors required')
    manifest = read_bound(config['processor_manifest'])
    audit = read_bound(config['processor_audit'])
    preparation = read_bound(config['preparation_transport'])
    verification = read_bound(config['verification_transport'])
    image = 'sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
    for transport in (preparation, verification):
        state = transport['state']
        if (transport['image'] != image or state['Status'] != 'exited'
                or state['Running'] or state['ExitCode'] != 0 or state['OOMKilled']):
            raise ValueError('Fixed-image successful independent CPU terminals required')
    if preparation['container_id'] == verification['container_id']:
        raise ValueError('Distinct preparation and verification processes required')
    if (manifest['schema'] != 'minds14-context-processor-v1'
            or manifest['recordings'] != 8168 or manifest['decisions'] != 14
            or manifest['context_limit'] != 4096 or manifest['max_tokens'] > 4096
            or manifest['cache_root'] != config['cache_root']
            or manifest['input_manifest_sha256'] != config['input_manifest_sha256']
            or audit['schema'] != 'minds14-processor-cache-verification-v1'
            or audit['recordings'] != 8168 or audit['decisions'] != 14
            or audit['processor_sha256'] != manifest['processor_sha256']
            or audit['max_tokens'] != manifest['max_tokens']
            or audit['input_manifest_sha256'] != config['input_manifest_sha256']
            or audit['output_manifest_sha256'] != config['processor_manifest']['sha256']
            or not audit['all_cache_tensors_verified'] or not audit['all_whole_audio_hops_present']
            or audit['transport_receipt_sha256'] != config['verification_transport']['sha256']
            or audit['preparation_transport_receipt_sha256'] != config['preparation_transport']['sha256']
            or preparation['files']['run/MANIFEST.json'] != config['processor_manifest']['sha256']
            or preparation['files']['run/inputs.jsonl'] != manifest['receipts_sha256']
            or file_hash(config['processor_receipts']) != manifest['receipts_sha256']):
        raise ValueError('Complete fixed processor/cache evidence required')
    # The published summary adds only transport bindings to the actual verifier result.
    actual = read_bound(config['actual_verification'])
    if (verification['files']['run/verification.json'] != config['actual_verification']['sha256']
            or verification['files']['package/tools/reports/verify_minds_processor_cache.py']
                != audit['verifier_sha256']
            or {k:v for k,v in audit.items() if k not in
                ('transport_receipt_sha256','preparation_transport_receipt_sha256')} != actual):
        raise ValueError('Actual executed verifier and published proof differ')
    with Path(config['processor_receipts']).open() as stream:
        receipts = [json.loads(line) for line in stream]
    if len(rows) != 8168 or len(receipts) != len(rows):
        raise ValueError('Complete target population required')
    for row, receipt in zip(rows, receipts, strict=True):
        if any(row[k] != receipt[k] for k in
               ('id', 'audio_sha256', 'request_sha256', 'rendered_sha256')):
            raise ValueError('Input/cache receipt order differs')
        if (set(receipt['tensors']) != {'input_ids', 'attention_mask', 'input_features',
                                      'input_features_mask', 'positions'}
                or len(receipt['positions']) != 1 or len(receipt['positions'][0]) != 14
                or not 0 < receipt['tokens'] <= 4096):
            raise ValueError('Full 14-marker input receipt required')
    return rows, receipts, definitions, manifest


def panel_indices(rows, receipts):
    """256 fixed inputs: length extremes, then deterministic ID-hash order."""
    if (len(rows) != len(receipts) or len(rows) < 256
            or len({r['id'] for r in rows}) != len(rows)
            or any(r['id'] != p['id'] for r, p in zip(rows, receipts, strict=True))):
        raise ValueError('aligned unique full input roster required')
    by_length = sorted(range(len(rows)), key=lambda i: (receipts[i]['tokens'], rows[i]['id']))
    chosen = set(by_length[:16] + by_length[-16:])
    for i in sorted(range(len(rows)), key=lambda i: (content_hash(rows[i]['id']), rows[i]['id'])):
        chosen.add(i)
        if len(chosen) == 256:
            break
    return sorted(chosen)


def checked_prediction(row, receipt, prediction, policy_id):
    identity = dict(policy_id=policy_id, positions=receipt['positions'][0],
                    inputs={k: v for k, v in receipt['tensors'].items() if k != 'positions'})
    if (prediction['policy_id'] != policy_id or prediction['request_identity'] != identity
            or prediction['request_id'] != content_hash(identity)
            or len(prediction['prob']) != 14
            or any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1
                   for p in prediction['prob'])):
        raise ValueError('engine policy, inputs or full probability vector differs')
    return dict(prediction, id=row['id'])


def audited_encoder(config, definitions, manifest):
    from multimodal_jev.cached_context_preparation import CachedContextPreparer, processor_cache_identity
    from multimodal_jev.encoded_audio_cache import EncodedAudioCache, key_for
    from multimodal_jev.span_request_identity import tensor_identity
    processor_id = processor_cache_identity(config['snapshot'])
    if processor_id != manifest['processor_sha256']:
        raise ValueError('policy processor differs from audited transfer inputs')
    cache_root = Path(config['cache_root'])
    if not cache_root.is_dir():
        raise ValueError('existing audited cache required')
    def no_rebuild():
        raise ValueError('Transfer cache miss; no implicit preprocessing during inference')
    prepare = CachedContextPreparer(no_rebuild, EncodedAudioCache(cache_root),
        processor_sha256=processor_id, decision_token_id=151706, audio_token_id=151676, max_tokens=4096)
    def encode(pair):
        row, receipt = pair
        request = reconstruct(row, definitions)
        key = key_for(prepare.identity(request, row['audio_sha256']))
        if key != receipt['cache_key'] or file_hash(cache_root / (key+'.pt')) != receipt['cache_sha256']:
            raise ValueError('audited cache bytes differ')
        value = prepare((request, row['audio_sha256']))
        tensors = {k: tensor_identity(v) for k, v in dict(value.inputs, positions=value.positions).items()}
        if (not value.cache_hit or tensors != receipt['tensors']
                or value.positions.tolist() != receipt['positions']):
            raise ValueError('actual execution inputs differ from independent audit')
        return row, receipt, value
    return encode


def compare(hf, vl):
    if (hf['policy_id'] != vl['policy_id'] or hf['indices'] != vl['indices']
            or hf['input_manifest_sha256'] != vl['input_manifest_sha256']
            or len(hf['indices']) != 256 or len(set(hf['indices'])) != 256
            or len(hf['records']) != 256 or len(vl['records']) != 256
            or len({r['id'] for r in hf['records']}) != 256):
        raise ValueError('same policy and full predeclared parity panel required')
    maximum = 0.
    flips = 0
    for a, b in zip(hf['records'], vl['records'], strict=True):
        if (a['id'] != b['id'] or a['request_identity'] != b['request_identity']
                or a['policy_id'] != hf['policy_id'] or b['policy_id'] != hf['policy_id']
                or a['request_id'] != content_hash(a['request_identity'])
                or b['request_id'] != a['request_id'] or len(a['prob']) != 14 or len(b['prob']) != 14):
            raise ValueError('same complete policy/input records required')
        for x, y in zip(a['prob'], b['prob'], strict=True):
            if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in (x, y)):
                raise ValueError('finite probabilities required')
            maximum = max(maximum, abs(x-y))
            flips += int((x >= .5) != (y >= .5))
    return dict(queries=256, decisions=14, probability_max_abs=maximum,
                threshold_flips=flips, exact=maximum == 0 and flips == 0)


def require_parity(config, out, rows, receipts):
    hf = json.loads((out/'hf.json').read_text())
    vl = json.loads((out/'vllm.json').read_text())
    indices = panel_indices(rows, receipts)
    if (hf['policy_id'] != config['policy_id'] or hf['indices'] != indices
            or hf['input_manifest_sha256'] != config['input_manifest_sha256']):
        raise ValueError('parity must cover this policy and predeclared input panel')
    for report in (hf, vl):
        for i, prediction in zip(indices, report['records'], strict=True):
            if prediction['id'] != rows[i]['id']:
                raise ValueError('parity input order differs')
            checked_prediction(rows[i], receipts[i], prediction, config['policy_id'])
    result = compare(hf, vl)
    if not result['exact']:
        raise ValueError('HF/vLLM probability parity failed')
    return result


def infer(config, out, rows, receipts, encode, engine):
    """Shared inference loop; engine and encoder are supplied by admitted runtime."""
    from multimodal_jev.local_checkpoint import commit_json
    from multimodal_jev.ordered_prefetch import OrderedPrefetch
    records = []
    started = time.monotonic()
    with OrderedPrefetch(encode, list(zip(rows, receipts, strict=True)), workers=1, ahead=2) as prepared:
        for index, (row, receipt, value) in enumerate(prepared):
            prediction = checked_prediction(row, receipt,
                engine.score(value.inputs, value.positions, audio_token_id=151676), config['policy_id'])
            commit_json(out/f'record-{index:05d}.json', dict(prediction=prediction, input_receipt=receipt))
            records.append(prediction)
            if (index+1) % 64 == 0:
                atomic_json(out/'progress.json', dict(completed=index+1, total=len(rows),
                    seconds=time.monotonic()-started, policy_id=config['policy_id']))
    if len(records) != len(rows) or len({r['id'] for r in records}) != len(rows):
        raise ValueError('incomplete unique transfer inference')
    commit_json(out/'predictions.json', dict(records=records))
    result = dict(recordings=len(records), decisions=14, policy_id=config['policy_id'],
        source_role=config['source_role'], input_manifest_sha256=config['input_manifest_sha256'],
        predictions_sha256=file_hash(out/'predictions.json'), seconds=time.monotonic()-started,
        scope='Label-free inference only; independent collection and target scoring required.')
    commit_json(out/'result.json', result)
    return result


def run(config, out, phase):
    import torch
    import torch_npu
    from multimodal_jev.context_policy_artifacts import read_execution_bundle, verify_native_adapter
    from multimodal_jev.context_runtime_profile import PROFILE
    from multimodal_jev.context_resident_rollout import FixedContextRollout
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.local_checkpoint import commit_json
    from multimodal_jev.policy_identity import checkpoint_identity
    if phase not in ('hf', 'vllm', 'inference'):
        raise ValueError('explicit runtime phase required')
    validate_source(config)
    rows, receipts, definitions, manifest = load_inputs(config)
    require_training_storage(out)
    if (len(config['physical_devices']) != 1 or sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))
            != ['davinci'+str(config['physical_devices'][0])]):
        raise ValueError('one isolated NPU required')
    if checkpoint_identity(config['model'])[0] != config['base_id']:
        raise ValueError('actual base identity differs')
    bundle = read_execution_bundle(config['snapshot'], expected_policy_id=config['policy_id'],
        expected_contract=config['contract'], expected_base_id=config['base_id'])
    if (bundle['contract']['runtime_profile'] != PROFILE
            or bundle['contract']['request_version'] != 'jev-context-exclusive_event-v1'
            or bundle['contract']['token']['registration_status'] != 'admitted'):
        raise ValueError('admitted shared context runtime required')
    verify_native_adapter(config['adapter'], bundle)
    encode = audited_encoder(config, definitions, manifest)
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch.manual_seed(317)
    torch.npu.manual_seed_all(317)
    indices = panel_indices(rows, receipts)
    if phase == 'inference':
        require_parity(config, out, rows, receipts)
    if phase == 'hf':
        from multimodal_jev.asr import load_runtime
        from multimodal_jev.sft import initialize_adapter
        from multimodal_jev.context_decision_head import ContextDecisionHead
        from multimodal_jev.context_decision_model import ContextDecisionModel
        from multimodal_jev.context_policy_snapshot import load_policy
        from multimodal_jev.context_runtime_profile import install_actor
        device = torch.device('npu', 0)
        _, base = load_runtime(config['model'], device, 'bfloat16')
        adapted = initialize_adapter(base, dict(lora_rank=4, lora_alpha=8, lora_execution_dtype='bfloat16')).eval()
        head = ContextDecisionHead(1024, 8, 1024, .1).to(device=device, dtype=torch.bfloat16).eval()
        model = ContextDecisionModel(adapted.get_base_model().model, head, 151706).eval()
        load_policy(config['snapshot'], model, expected_policy_id=config['policy_id'],
            expected_contract=config['contract'], expected_base_id=config['base_id'])
        handle, runtime = install_actor(adapted, model, PROFILE)
        commit_json(out/'hf-runtime.json', runtime)
        records = []
        for i in indices:
            row, receipt, value = encode((rows[i], receipts[i]))
            with torch.no_grad():
                prob = model({k: v.to(device) for k, v in value.inputs.items()}, value.positions.to(device)).cpu()[0].tolist()
            identity = dict(policy_id=config['policy_id'], positions=receipt['positions'][0],
                inputs={k: v for k, v in receipt['tensors'].items() if k != 'positions'})
            records.append(checked_prediction(row, receipt, dict(policy_id=config['policy_id'],
                request_identity=identity, request_id=content_hash(identity), prob=prob), config['policy_id']))
    else:
        ec = {k: config[k] for k in ('model', 'base_id', 'snapshot', 'policy_id', 'contract', 'adapter', 'physical_devices')}
        ec.update(tensor_parallel_size=1, max_model_len=4096, max_decisions=14)
        engine = FixedContextRollout(ec)
        try:
            commit_json(out/(phase+'-runtime.json'), engine.runtime)
            if phase == 'inference':
                return infer(config, out, rows, receipts, encode, engine)
            records = []
            for i in indices:
                row, receipt, value = encode((rows[i], receipts[i]))
                records.append(checked_prediction(row, receipt,
                    engine.score(value.inputs, value.positions, audio_token_id=151676), config['policy_id']))
        finally:
            engine.close()
    result = dict(policy_id=config['policy_id'], indices=indices, records=records,
                  input_manifest_sha256=config['input_manifest_sha256'])
    commit_json(out/(phase+'.json'), result)
    if phase == 'vllm':
        commit_json(out/'parity.json', require_parity(config, out, rows, receipts))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--phase', required=True, choices=['hf', 'vllm', 'inference'])
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(exist_ok=True)
    run(json.loads(Path(a.config).read_text()), out, a.phase)
