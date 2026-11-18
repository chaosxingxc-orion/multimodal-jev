"""Label-free full native evaluation using the frozen two-replica runtime.

Source selection precedes target access. Each policy gets independent HF
verification on its authenticated target-width panel; the full macro phase
never opens gold. Container, policy tensor and source provenance admission
remain independent launcher/collector obligations.
"""
import argparse
import json
from pathlib import Path
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_policy_artifacts import read_execution_bundle, verify_native_adapter
from multimodal_jev.context_reference_dataset import digest
from multimodal_jev.context_runtime_profile import PROFILE
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache
from tools.reports import audit_context_transfer_input_coverage as coverage
from tools.reports import audit_context_transfer_macro as shared_audit
from tools.reports import context_transfer_macro_inputs as transfer_inputs
from tools.reports.context_dual_runtime_candidate import verify_runtime
from tools.reports.context_macro_evaluation import evaluate
from tools.reports.context_replica_macro_client import connect_pool
from tools.reports.context_train_macro_parity import hf_reference
from tools.reports.context_transfer_macro_parity import compare
from tools.reports.context_transfer_parity_inputs import load as load_panel
from tools.reports.context_transfer_portable_inputs import load as load_full, SIZES
from tools.reports.evaluate_context_transfer_macro import source_binding, read_bound


def read(path):
    return json.loads(Path(path).read_text())


def validate_config(c):
    client = c['macro_client']
    if (c['dataset'] not in SIZES
            or client['schema'] != 'jev-replica-macro-client-v1'
            or client['max_queries'] != 128 or client['rollout_queries'] != 32
            or len(client['replicas']) != 2 or c['macro_queries'] != 128
            or client['runtime_identity'] != c['dual_runtime_identity']
            or c['source_policy'] != c['snapshot']
            or c['source_policy_id'] != c['policy_id']
            or c['source_contract'] != c['contract']):
        raise ValueError('frozen source policy, native simple task and replica macro128/rollout32 required')
    for name, expected in c['pins'].items():
        if file_hash(name) != expected:
            raise ValueError('transfer dependency changed: ' + name)
    verify_runtime(c)
    bundle = read_execution_bundle(c['snapshot'], expected_policy_id=c['policy_id'],
        expected_contract=c['contract'], expected_base_id=c['base_id'])
    contract = bundle['contract']
    if (contract['request_version'] != 'jev-context-exclusive_event-v1'
            or contract['runtime_profile'] != PROFILE
            or contract['token']['registration_status'] != 'admitted'):
        raise ValueError('admitted direct-audio exclusive-event policy required')
    verify_native_adapter(c['initial_adapter'], bundle)
    return bundle


def inputs(c):
    full = load_full(c['portable_full_inputs'])
    panel = load_panel(c['portable_parity_inputs'])
    if full['dataset'] != c['dataset']:
        raise ValueError('portable full dataset differs')
    if c['dataset'] == 'SLURP_clean_TEST60' and full['source_release'] != c.get('source_release'):
        raise ValueError('held-out source ontology differs')
    if Path(full['cache_root']).resolve() != Path(panel['cache_root']).resolve():
        raise ValueError('full and panel must consume the same readonly cache')
    # Reuse the independent coverage verifier without re-parsing the full
    # portable inventory a second time. Both loaders above authenticate bytes.
    bound = FunctionType(coverage.audit.__code__, dict(coverage.audit.__globals__,
        load_full=lambda _: full, load_panel=lambda _: panel))
    proof = bound(c['portable_full_inputs'], c['portable_parity_inputs'])
    return full, panel['selected_panel'], proof


def encoder(c, admitted, bundle):
    original = transfer_inputs.encoder
    readonly = FunctionType(original.__code__, dict(original.__globals__,
        EncodedAudioCache=ReadOnlyEncodedAudioCache))
    checked = FunctionType(transfer_inputs.audited_encoder.__code__,
        dict(transfer_inputs.audited_encoder.__globals__, encoder=readonly))
    return checked(c, admitted, snapshot=c['snapshot'],
        decision_token_id=bundle['contract']['token']['id'], audio_token_id=c['audio_token_id'])


def input_evidence(c, admitted, proof):
    spec = c['portable_full_inputs']
    return dict(dataset=c['dataset'], processor_sha256=admitted['processor_sha256'],
        inputs={k: spec[k] for k in ('manifest_sha256', 'population_evidence_sha256')},
        recordings=len(admitted['population']), decisions=len(admitted['candidate_ids']),
        coverage=proof)


def parity(reference, records, selected, policy_id):
    lookup = {r['id']: r for r in records}
    if len(lookup) != len(records) or any(a['id'] not in lookup for a in selected['assets']):
        raise ValueError('unique complete full prediction population required')
    proof = compare(reference, [lookup[a['id']] for a in selected['assets']], selected,
        policy_id=policy_id)
    if proof['exact'] is not True:
        raise ValueError('exact per-policy target-width HF/replica parity required')
    return proof


def run(c, provenance, phase):
    if phase not in ('hf', 'macro'):
        raise ValueError('explicit HF or full macro phase required')
    binding = source_binding(c)  # Freeze source roles before touching targets.
    bundle = validate_config(c)
    admitted, selected, proof = inputs(c)
    encode = encoder(c, admitted, bundle)
    root = Path(c['output'])
    check_paths({'transfer_output': root})
    root.mkdir(parents=True, exist_ok=False)
    pool = None
    try:
        commit_json(root/'provenance.json', provenance)
        commit_json(root/'source-binding.json', binding)
        commit_json(root/'input-evidence.json', input_evidence(c, admitted, proof))
        commit_json(root/'panel.json', selected)
        if phase == 'hf':
            result = hf_reference(c, selected, bundle, encode, root)
            commit_json(root/'reference.json', result)
        else:
            reference = read_bound(c['hf_reference'])
            if (reference['policy_id'] != c['policy_id']
                    or reference['panel_sha256'] != digest(selected)):
                raise ValueError('same source-frozen policy and input-only HF panel required')
            pool = connect_pool(c['macro_client'], bundle=bundle)
            inference = evaluate(pool, admitted['population'], encode,
                candidate_ids=admitted['candidate_ids'], policy_id=c['policy_id'],
                runtime_identity=c['dual_runtime_identity'], audio_token_id=c['audio_token_id'],
                macro_queries=128, output=root/'inference')
            pool.close()
            pool = None
            records = read(root/'inference/predictions.json')['records']
            if [r['id'] for r in records] != [a['id'] for a in admitted['population']]:
                raise ValueError('complete ordered held-out predictions required')
            comparison = parity(reference, records, selected, c['policy_id'])
            commit_json(root/'parity.json', comparison)
            result = dict(dataset=c['dataset'], policy_role=c['policy_role'], policy_id=c['policy_id'],
                recordings=inference['recordings'], decisions=inference['decisions'],
                predictions_sha256=inference['predictions_sha256'],
                input_receipts_sha256=inference['input_receipts_sha256'],
                inference_result_sha256=file_hash(root/'inference/result.json'),
                source_binding_sha256=file_hash(root/'source-binding.json'),
                parity_sha256=file_hash(root/'parity.json'), target_scored=False,
                scope='Full label-free replica inference; independent runtime/source audit and target scoring required')
            commit_json(root/'result.json', result)
        commit_json(root/'terminal.json', dict(exit_code=0))
        return result
    except BaseException as exc:
        commit_json(root/'terminal.json', dict(exit_code=1, error=repr(exc)))
        raise
    finally:
        if pool is not None:
            pool.close()


def audit_outputs(c, collection, journal, hf_collection):
    """Reconstruct full results and parity; no target gold or model execution."""
    source_binding(c)
    admitted, selected, proof = inputs(c)
    root, hf = Path(collection), Path(hf_collection)
    expected_evidence = input_evidence(c, admitted, proof)
    for directory in (root, hf):
        if (read(directory/'terminal.json') != {'exit_code': 0}
                or read(directory/'panel.json') != selected
                or read(directory/'input-evidence.json') != expected_evidence):
            raise ValueError('successful same-population HF and macro phases required')
    if read(root/'source-binding.json') != read(hf/'source-binding.json'):
        raise ValueError('HF and macro source roles differ')
    audit = FunctionType(shared_audit.audit_outputs.__code__,
        dict(shared_audit.audit_outputs.__globals__, load=lambda _: admitted))
    result = audit(c, root, journal)
    reference = read(hf/'reference.json')
    comparison = parity(reference, read(root/'inference/predictions.json')['records'],
        selected, c['policy_id'])
    if (read(root/'parity.json') != comparison
            or read(root/'result.json')['parity_sha256'] != file_hash(root/'parity.json')):
        raise ValueError('reported parity differs from independent reconstruction')
    return dict(result, status='complete_replica_transfer_outputs_audited', parity=comparison,
        coverage=proof, runtime_provenance_verified=False, target_scored=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    p.add_argument('--phase', choices=['hf', 'macro'], required=True)
    a = p.parse_args()
    config, provenance = load_config(a.config, a.local)
    print(json.dumps(run(config, provenance, a.phase), indent=2))
