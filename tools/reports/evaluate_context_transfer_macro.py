"""Label-free held-out macro inference for source-frozen policy roles.

Source point tensor/runtime audits and target-width backend parity must be
authenticated by the launcher/collector. A bound curve is an input to this
client, not proof that these independent audits actually happened.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from multimodal_jev.context_runtime_profile import PROFILE
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.jev.context_selection_curve import summarize
from tools.reports.context_macro_client import connect_pool
from tools.reports.context_macro_evaluation import evaluate
from tools.reports.context_transfer_macro_inputs import load, audited_encoder


def read_bound(entry):
    if set(entry) != {'path', 'sha256'} or file_hash(entry['path']) != entry['sha256']:
        raise ValueError('bound source evidence changed')
    return json.loads(Path(entry['path']).read_text())


def source_binding(config):
    """Recompute selection from complete source metrics, never from target scores."""
    source = read_bound(config['source_evaluation'])
    experiment = source['experiment']
    if (config.get('dataset')=='SLURP_clean_TEST60'
            and config.get('source_release',{}).get('sha256')!=experiment['release_sha256']):
        raise ValueError('clean TEST ontology must bind the actual selected source release')
    evaluation = experiment['evaluation']
    if (source['schema'] != 'jev-macro-transfer-source-evaluation-v1'
            or source['algorithm'] != 'dual_credit_grpo_floor_clipped_v1'
            or experiment['task_stage']['updates'] != 634
            or experiment['selection_recordings'] != 9226
            or evaluation['checkpoints'] != [0, *range(64, 634, 64), 634]
            or evaluation['primary'] != 'query_macro_set_f1'
            or 'candidate_binary_nll' not in evaluation['secondary']
            or evaluation['set_threshold'] != .5
            or source['target_tuning_allowed'] is not False):
        raise ValueError('predeclared full source-only dual-credit GRPO curve required')
    points = []
    for item in source['points']:
        if set(item) != {'step', 'result'} or type(item['step']) is not int:
            raise ValueError('source points may only bind step and audited score artifact')
        points.append(dict(step=item['step'], result=read_bound(item['result'])))
    curve = summarize(points, experiment, initial_policy_id=source['initial_policy_id'])
    if not curve['complete'] or curve['selected_checkpoint'] != source['selected_checkpoint']:
        raise ValueError('complete source curve and recomputed checkpoint choice required')
    roles = dict(common_Stage1=dict(step=0, policy_id=curve['initial_policy_id']),
        source_selected_new_GRPO=curve['selected_checkpoint'],
        fixed_endpoint_new_GRPO=dict(step=634, policy_id=curve['curve'][-1]['policy_id']))
    role = config['policy_role']
    if role not in roles or config['policy_id'] != roles[role]['policy_id']:
        raise ValueError('policy does not match the frozen source role')
    return dict(source_evaluation=config['source_evaluation'], role=role, **roles[role],
        source_selected_checkpoint=curve['selected_checkpoint'], target_tuning_allowed=False)


def run(config, provenance):
    # Source roles are frozen before even opening label-free target requests.
    binding = source_binding(config)
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('pinned dependency changed: ' + path)
    bundle = read_execution_bundle(config['snapshot'], expected_policy_id=config['policy_id'],
        expected_contract=config['contract'], expected_base_id=config['base_id'])
    contract = bundle['contract']
    if (contract['request_version'] != 'jev-context-exclusive_event-v1'
            or contract['runtime_profile'] != PROFILE
            or contract['token']['registration_status'] != 'admitted'):
        raise ValueError('admitted exclusive-event policy required')
    admitted = load(config)
    encode = audited_encoder(config, admitted, snapshot=config['snapshot'],
        decision_token_id=contract['token']['id'], audio_token_id=config['audio_token_id'])
    out = Path(config['output'])
    check_paths({'evaluation': out, 'cache': Path(admitted['cache_root'])})
    out.mkdir(parents=True, exist_ok=False)
    pool = None
    try:
        commit_json(out/'provenance.json', provenance)
        commit_json(out/'source-binding.json', binding)
        commit_json(out/'input-evidence.json', dict(dataset=config['dataset'],
            processor_sha256=admitted['processor_sha256'],
            inputs=config['inputs'], recordings=len(admitted['population']),
            decisions=len(admitted['candidate_ids'])))
        pool = connect_pool(config['macro_client'], bundle=bundle)
        inference = evaluate(pool, admitted['population'], encode,
            candidate_ids=admitted['candidate_ids'], policy_id=config['policy_id'],
            runtime_identity=config['macro_client']['runtime_identity'],
            audio_token_id=config['audio_token_id'], macro_queries=config['macro_queries'],
            output=out/'inference')
        pool.close()
        pool = None
        result = dict(dataset=config['dataset'], policy_role=config['policy_role'],
            policy_id=config['policy_id'], recordings=inference['recordings'],
            decisions=inference['decisions'], predictions_sha256=inference['predictions_sha256'],
            input_receipts_sha256=inference['input_receipts_sha256'],
            inference_result_sha256=file_hash(out/'inference/result.json'),
            source_binding_sha256=file_hash(out/'source-binding.json'),
            target_scored=False,
            scope='Label-free full macro inference; independent runtime/source/parity audit and target scoring required')
        commit_json(out/'result.json', result)
        commit_json(out/'terminal.json', dict(exit_code=0))
        return result
    except BaseException as exc:
        commit_json(out/'terminal.json', dict(exit_code=1, error=repr(exc)))
        raise
    finally:
        if pool is not None:
            pool.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    args = parser.parse_args()
    config, provenance = load_config(args.config, args.local)
    run(config, provenance)
