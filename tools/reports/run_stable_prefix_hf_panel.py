"""Run the HF side of endpoint parity on an authenticated stable-prefix TRAIN panel."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_preparation import processor_cache_identity
from multimodal_jev.context_decision_contract import PREFIX_LAYOUT, ApplicabilityQuestion, ContextDecisionRequest
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import require_training_storage
from multimodal_jev.policy_identity import checkpoint_identity
from tools.reports.context_selection_audited_cache import encoder
from tools.reports.context_train_parity_inputs import load as load_panel
from tools.reports.context_transfer_macro_parity import hf_reference


def run(c, provenance):
    root = Path(c['output'])
    require_training_storage(root)
    root.mkdir(parents=True, exist_ok=False)
    try:
        point = c['point']
        snapshot = Path(c['snapshot'])
        if file_hash(snapshot/'MANIFEST.json') != point['manifest_sha256']:
            raise ValueError('audited checkpoint manifest changed')
        bundle = read_execution_bundle(snapshot, expected_policy_id=point['policy_id'],
                                       expected_contract=c['contract'],
                                       expected_base_id=c['base_id'])
        if checkpoint_identity(c['model'])[0] != c['base_id']:
            raise ValueError('actual base differs')
        values = load_panel(c['panel'])
        source = values['source']
        processor = processor_cache_identity(snapshot)
        if (source.get('prompt_layout') != PREFIX_LAYOUT or source['processor_sha256'] != processor
                or source['release_sha256'] != c['release_sha256']):
            raise ValueError('same stable-prefix TRAIN release and processor required')
        requests, assets, priors = {}, [], []
        for row in values['inputs']:
            r, prior = row['request'], row['receipt']
            requests[row['id']] = ContextDecisionRequest(
                r['audio'], tuple(ApplicabilityQuestion(**q) for q in r['questions']),
                tuple(r['requirements']), tuple(r['history']), r['version'])
            assets.append({k: prior[k] for k in ('id', 'audio_sha256', 'request_sha256')})
            priors.append(prior)
        encode = encoder(c['panel']['cache_root'], requests, values['candidate_ids'], priors,
                         processor_sha256=processor, decision_token_id=bundle['contract']['token']['id'],
                         audio_token_id=c['audio_token_id'], prompt_layout=PREFIX_LAYOUT)

        def prepare(asset):
            row, receipt = encode(asset)
            return row, {k: v for k, v in receipt.items() if k != 'audio_receipt'}

        from multimodal_jev.context_reference_dataset import digest
        selected = dict(schema='jev-source-train-runtime-panel-v1', dataset='SLURP_clean_TRAIN60',
                        candidate_ids=values['candidate_ids'], assets=assets,
                        input_receipts=[prepare(asset)[1] for asset in assets],
                        selection_sha256=digest(values['panel']),
                        input_manifest_sha256=c['panel']['manifest_sha256'], labels_used=False)
        commit_json(root/'panel.json', selected)
        commit_json(root/'provenance.json', provenance)
        hf_config = dict(c, policy_id=bundle['policy_id'], contract=bundle['contract'])
        proof = hf_reference(hf_config, {}, selected, bundle, prepare, root)
        commit_json(root/'reference.json', proof)
        commit_json(root/'terminal.json', dict(exit_code=0, phase='hf_parity_side_only'))
    except BaseException as exc:
        commit_json(root/'terminal.json', dict(exit_code=1, error=repr(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    args = parser.parse_args()
    config, meta = load_config(args.config, args.local)
    run(config, meta)
