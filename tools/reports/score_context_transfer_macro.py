"""Offline native task scoring of independently reconstructed macro outputs.

This reusable collector component never dispatches inference or chooses a policy.
It does not certify container provenance or backend parity; the production
collector must admit those before invoking it. No standalone dispatch entry.
"""
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.fsc_transfer_population import scoring_targets
from multimodal_jev.slurp_training import content_hash
from tools.jev.prepare_fsc_context_transfer import compile_native
from tools.jev.fsc_context_metrics import score as score_fsc, compare as compare_fsc
from tools.reports.score_minds_context_pair import score_records as score_minds, compare as compare_minds
from tools.reports.audit_context_transfer_macro import audit_outputs
from tools.reports.context_transfer_macro_inputs import load
from tools.reports.evaluate_context_transfer_macro import read_bound, source_binding


ROLES = ('common_Stage1', 'source_selected_new_GRPO', 'fixed_endpoint_new_GRPO')


def fsc_truth(config, scoring, admitted):
    population = read_bound(scoring['population'])
    task = read_bound(scoring['task'])
    manifest = read_bound(config['inputs']['input_manifest'])
    if (manifest['task_sha256'] != scoring['task']['sha256']
            or manifest['population_sha256'] != scoring['population']['sha256']
            or population['split'] != 'test' or population['recordings'] != 3793
            or len(population['records']) != 3793
            or [c['id'] for c in task['commands']] != admitted['candidate_ids']):
        raise ValueError('scoring must use the exact full FSC test population and definitions')
    for row, asset, original in zip(population['records'], admitted['population'],
                                    admitted['source_rows'], strict=True):
        request = compile_native(row['audio'], task)
        if (any(row[k] != asset[k] for k in ('id', 'audio_sha256'))
                or row['speaker_id'] != original['speaker_id']
                or content_hash(request.model_input()) != asset['request_sha256']):
            raise ValueError('FSC scoring ontology, speaker or audio differs from inference')
    native = scoring_targets(population, task)
    fields = ('action', 'object', 'location')
    lookup = {tuple(c['values'][k] for k in fields): c['id'] for c in task['commands']}
    return population['records'], {r['id']: lookup[tuple(r['gold'][k] for k in fields)] for r in native}


def minds_truth(config, scoring, admitted):
    contract = read_bound(scoring['contract'])
    if (contract['schema'] != 'minds-native-descriptor-scoring-v1'
            or contract['threshold'] != .5 or contract['target_tuning'] is not False):
        raise ValueError('fixed native MINDS scoring contract required')
    artifacts = contract['artifacts']
    root = Path(config['inputs']['inputs'])
    for name in ('MANIFEST.json', 'inputs.jsonl', 'definitions.json', 'conversion.jsonl'):
        entry = artifacts[name.replace('.', '_')]
        if file_hash(root/name) != entry['sha256'] or file_hash(entry['path']) != entry['sha256']:
            raise ValueError('scoring inputs differ from authenticated inference inputs')
    if artifacts['protocol']['sha256'] != config['inputs']['protocol']['sha256']:
        raise ValueError('scoring protocol differs from inference')
    protocol = read_bound(artifacts['protocol'])
    if protocol['population']['inventory']['sha256'] != artifacts['inventory']['sha256']:
        raise ValueError('scoring inventory differs from frozen protocol')
    inventory = read_bound(artifacts['inventory'])
    ledger = [json.loads(line) for line in (root/'conversion.jsonl').read_text().splitlines()]
    inputs = admitted['source_rows']
    original = {r['id']: r for r in inventory['records']}
    if (len(original) != len(inventory['records']) or len(original) != 8168
            or len(inputs) != 8168 or len(ledger) != 8168
            or set(original) != {r['id'] for r in inputs}
            or inventory['intent_ids'] != admitted['candidate_ids']
            or [c['id'] for c in protocol['candidates']] != admitted['candidate_ids']):
        raise ValueError('complete frozen MINDS population and ontology required')
    truth = {}
    for row, conversion in zip(inputs, ledger, strict=True):
        old = original[row['id']]
        if (conversion['id'] != row['id']
                or conversion['source_audio_sha256'] != old['encoded_audio_sha256']
                or conversion['target_audio_sha256'] != row['audio_sha256']
                or conversion['source_samples'] != old['samples']
                or conversion['target_samples'] != 2*old['samples'] or old['upstream_split'] != 'train'):
            raise ValueError('MINDS gold does not match original and resampled audio')
        truth[row['id']] = dict(intent=old['intent'], language=old['language'])
    return inputs, truth


def score_matrix(entries, scoring):
    """Require all three roles before labels; identical policies reuse one output.

    Each role has evaluation config, collection and journal. An alias role uses
    its requested source binding but may reference another role's execution;
    the actual saved role is audited unchanged and policy IDs must be identical.
    """
    if set(entries) != set(ROLES):
        raise ValueError('all three predeclared source roles required')
    reports = {}; audited = {}; common = None; admitted = None
    for role in ROLES:
        entry = entries[role]; config = entry['evaluation']
        if config['policy_role'] != role:
            raise ValueError('matrix role differs from source binding')
        binding = source_binding(config)
        identity = dict(dataset=config['dataset'], source=config['source_evaluation']['sha256'],
                        inputs=config['inputs'], runtime=config['macro_client']['runtime_identity'])
        if config['dataset']=='SLURP_clean_TEST60':identity['source_release']=config['source_release']
        if common is None: common = identity
        if identity != common:
            raise ValueError('same source curve, inputs and runtime required for the matrix')
        policy = binding['policy_id']
        location = (str(Path(entry['collection']).resolve()), str(Path(entry['journal']).resolve()))
        if policy in audited:
            if location != audited[policy]['location']:
                raise ValueError('identical policy must reuse authenticated predictions')
            reports[role] = audited[policy]
            continue
        actual = json.loads((Path(entry['collection'])/'source-binding.json').read_text())
        actual_config = dict(config, policy_role=actual['role'])
        proof = audit_outputs(actual_config, entry['collection'], entry['journal'])
        if proof['policy_id'] != policy:
            raise ValueError('collected predictions belong to a different policy')
        predictions = Path(entry['collection'])/'inference/predictions.json'
        if file_hash(predictions) != proof['predictions_sha256']:
            raise ValueError('predictions changed after independent output audit')
        reports[role] = audited[policy] = dict(location=location, proof=proof,
            records=json.loads(predictions.read_text())['records'])
    # No gold or label inventory is opened until every unique policy passes.
    config = entries[ROLES[0]]['evaluation']
    admitted = load(config)
    return score_admitted(reports, admitted, config, scoring)


def score_admitted(reports, admitted, config, scoring):
    """Native metrics after the caller authenticates every role and input."""
    if set(reports) != set(ROLES):
        raise ValueError('all three admitted roles required before target gold')
    dataset = config['dataset']
    if dataset == 'FSC_native31':
        population, truth = fsc_truth(config, scoring, admitted)
        compare = compare_fsc
    elif dataset == 'MINDS_native14':
        population, truth = minds_truth(config, scoring, admitted)
        compare = compare_minds
    elif dataset == 'SLURP_clean_TEST60':
        from tools.reports.context_slurp_test_macro_truth import load as slurp_truth
        from tools.reports.score_slurp_context_test_pair import compare, score_records as score_slurp
        population,truth=slurp_truth(config,scoring,admitted)
    else:
        raise ValueError('registered native transfer task required')
    scored = {}; cached = {}
    for role in ROLES:
        report = reports[role]; policy = report['proof']['policy_id']
        if policy not in cached:
            args = (report['records'], population, truth, admitted['candidate_ids'])
            if dataset == 'FSC_native31':
                cached[policy] = score_fsc(*args, policy_id=policy, input_receipts=admitted['cache_receipts'])
            elif dataset=='MINDS_native14':
                cached[policy] = score_minds(report['records'], population, admitted['cache_receipts'],
                    truth, admitted['candidate_ids'], policy, request_identity_protocol='macro_compact_json_v1')
                if len(cached[policy]['variants']) != 14 or len(cached[policy]['languages']) != 12:
                    raise ValueError('full MINDS 14-variety and 12-language scores required')
            else:
                cached[policy]=score_slurp(report['records'],population,admitted['cache_receipts'],
                    truth,admitted['candidate_ids'],policy,request_identity_protocol='macro_compact_json_v1')
        scored[role] = cached[policy]
    return dict(dataset=dataset, policies=scored,
        comparisons={role: compare(scored[ROLES[0]], scored[role]) for role in ROLES[1:]},
        inference_proofs={role: reports[role]['proof'] for role in ROLES},
        unique_policies=len(cached), target_tuning=False, runtime_provenance_verified=False,
        evaluation_kind='source_domain_held_out' if dataset=='SLURP_clean_TEST60' else 'cross_dataset_transfer',
        scope='Full native scores from independently reconstructed outputs; production collector must '
              'authenticate source weights, actual runtime transport and target-width parity. No semantic-presence claim.')
