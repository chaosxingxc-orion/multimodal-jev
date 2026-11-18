"""Generate an isolated full-run entry from the exact checkpoint candidate.

No launch or release authorization. The new entry requires a separately
published formal release and a full Stage1 source package at runtime.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


CANDIDATE_SHA = '941b9c89f8b61d07c11f452a29923002c40ca49280c1b00a27a9961b8b147699'
GUARD = """        if c['engineering_prefix'] is not True or c['training_dispatch_allowed'] is not False:
            raise ValueError('continuous candidate is engineering-only, not released')
        if c['engineering_prefix'] is True and stop_update-start_update>2:
            raise ValueError('engineering prefix is at most two updates')
"""
REPLACEMENT = """        from context_formal_runtime import validate
        validate(c)
"""


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(source):
    if hashlib.sha256(source.encode()).hexdigest() != CANDIDATE_SHA:
        raise ValueError('exact checkpoint candidate required')
    if source.count(GUARD) != 1:raise ValueError('bounded candidate guard differs')
    changed = source.replace(GUARD, REPLACEMENT)
    # Runtime admission changes only. The numerical update/publication body
    # remains byte-identical to the candidate queued for NPU acceptance.
    ast.parse(changed)
    return changed


def prepare(*, source, output):
    source, output = Path(source), Path(output)
    value = transform(source.read_text())
    output.mkdir(exist_ok=False)
    target = output / 'train_context_formal.py'
    target.write_text(value)
    validator = Path('tools/jev/context_formal_runtime.py')
    (output / validator.name).write_bytes(validator.read_bytes())
    manifest = dict(status='formal_entry_candidate_not_released', training_dispatch_allowed=False,
        source_sha256=sha(source), changes=['replace_two_update_guard_with_formal_release_validation'],
        unchanged=['native_verl_update', 'sampling', 'loss', 'precision', 'every_update_policy_activation',
                   'interval_optimizer_checkpoint', 'fixed_reference'],
        pending=['actual_continuous_replay_checkpoint_acceptance', 'full_stage1_source',
                 'formal_release_package_and_launcher'],
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output / 'PREPARED.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    print(json.dumps(prepare(**vars(p.parse_args())), indent=2))
