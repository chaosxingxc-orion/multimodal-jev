"""Native three-role scoring over independently audited replica collections."""
import argparse
import inspect
import json
from types import FunctionType

from multimodal_jev.local_config import load_config
from tools.reports import collect_context_full_transfer_scores as shared
from tools.reports import context_replica_transfer_operations as operations


def score(c):
    source = inspect.getsource(shared.score)
    replacements = [
        ("Path(ops['prepared'])/'package'", "Path(ops['transfer_package']['output'])"),
        ("read(package/'evaluation.json')", "read(package/'parity-macro.json')"),
        ("origin=ops['source_admission']", "origin=ops['transfer_package']['source_admission']"),
        ("audit_collected(ops)", "audit_collected(ops, {'source':'replica_transfer_scores'})"),
        ("'full_portable_transfer_outputs_audited'", "'complete_replica_transfer_audited'"),
        ("proof['files_sha256']!=ops['manifest_sha256']", "proof['manifest_sha256']!=ops['transfer_package']['manifest_sha256']"),
        ("local_config(package,e)", "local_config(package)"),
        ("Path(ops['operations_root'])/'collected/evaluation/inference/predictions.json'",
         "Path(ops['collection'])/'evaluation/inference/predictions.json'"),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('shared scoring layout anchor changed: '+before)
        source = source.replace(before, after)
    namespace = dict(shared.score.__globals__, audit_collected=operations.audit, local_config=operations.local_config)
    exec(compile(source, '<replica-transfer-native-scores>', 'exec'), namespace)
    return namespace['score'](c)


def collect(c, provenance):
    fn = FunctionType(shared.collect.__code__, dict(shared.collect.__globals__, score=score))
    return fn(c, provenance)


if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    a = p.parse_args()
    c, provenance = load_config(a.config, a.local)
    result = collect(c, provenance)
    print(json.dumps(dict(dataset=result['dataset'], unique_policies=result['unique_policies'],
        comparisons=result['comparisons']), indent=2))
