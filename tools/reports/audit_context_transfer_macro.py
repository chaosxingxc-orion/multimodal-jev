"""Independently reconstruct transfer client outputs before opening target gold.

The caller supplies authenticated source evidence and portable target inputs.
Actual container transport, adapter tensors and target-width HF/vLLM parity
must be independently admitted; this module does not certify those properties.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.audit_context_macro_evaluation import audit
from tools.reports.context_transfer_macro_inputs import load
from tools.reports.evaluate_context_transfer_macro import source_binding


def audit_outputs(config, collection, journal):
    """config may relocate input files; hashes and all logical identities persist."""
    binding = source_binding(config)
    admitted = load(config)
    root = Path(collection)
    def read(name):
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('contained transfer output required')
        return json.loads(path.read_text())
    if read('terminal.json') != {'exit_code': 0}:
        raise ValueError('successful complete transfer client required')
    saved = read('source-binding.json')
    # Allow relocation of the source evidence path only, never its content hash.
    expected = dict(binding, source_evaluation=saved['source_evaluation'])
    if (saved != expected or set(saved['source_evaluation']) != {'path', 'sha256'}
            or saved['source_evaluation']['sha256'] != binding['source_evaluation']['sha256']):
        raise ValueError('source role or selected source evidence differs')
    evidence = read('input-evidence.json')
    if (evidence['dataset'] != config['dataset']
            or evidence['processor_sha256'] != admitted['processor_sha256']
            or evidence['recordings'] != len(admitted['population'])
            or evidence['decisions'] != len(admitted['candidate_ids'])):
        raise ValueError('target input evidence identity differs')
    checked = audit(root/'inference', journal, population=admitted['population'],
        candidate_ids=admitted['candidate_ids'], input_receipts=admitted['input_receipts'],
        policy_id=config['policy_id'], runtime_identity=config['macro_client']['runtime_identity'],
        macro_queries=config['macro_queries'])
    result = read('result.json')
    expected_result = dict(dataset=config['dataset'], policy_role=config['policy_role'],
        policy_id=config['policy_id'], recordings=checked['recordings'], decisions=checked['decisions'],
        predictions_sha256=checked['predictions_sha256'],
        input_receipts_sha256=checked['input_receipts_sha256'],
        inference_result_sha256=file_hash(root/'inference/result.json'),
        source_binding_sha256=file_hash(root/'source-binding.json'), target_scored=False)
    if any(result.get(k) != v for k, v in expected_result.items()):
        raise ValueError('transfer result differs from independent reconstruction')
    return dict(status='complete_transfer_macro_outputs_audited', **expected_result,
        inference_audit=checked, source_binding=binding,
        scope='Complete independently bound source role, target inputs and macro outputs; '
              'runtime transport, tensors, target-width backend parity and target scoring are separate obligations')


def run(config, provenance):
    report = audit_outputs(config['evaluation'], config['collection'], config['journal'])
    target = Path(config['output'])
    check_paths({'audit': target})
    target.parent.mkdir(parents=True, exist_ok=True)
    report['provenance'] = provenance
    commit_json(target, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    args = parser.parse_args()
    config, provenance = load_config(args.config, args.local)
    run(config, provenance)
