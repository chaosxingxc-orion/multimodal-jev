"""Prepare a configured runtime after complete acoustic/KV efficiency evidence.

Reuse the formal package builder without modifying frozen live dependencies.
Selection is explicit, not automatic promotion; tail/parity/recovery and launch
admission remain separate. Machine bindings come from local macros.
"""
import argparse
import json

from multimodal_jev.local_config import load_config
from tools.reports import prepare_context_macro_formal as shared
from tools.reports import context_macro_formal_operations as baseline
from tools.reports import context_acoustic_formal_operations as acoustic
from tools.reports import context_training_efficiency_recommendation as efficiency
from tools.reports.watch_context_acoustic_training import bind


def choice(c):
    selected = c['selected_arm']
    reason = c.get('selection_reason')
    if selected not in efficiency.ARMS or not isinstance(reason, str) or not reason.strip():
        raise ValueError('explicit supported runtime and selection reason required')
    config, provenance = load_config(c['recommendation_config'], c['recommendation_local'])
    result = efficiency.report(config)
    if result.get('status') != 'complete_efficiency_evidence_reconstructed':
        raise ValueError('complete independently reconstructed training efficiency required')
    # The shared builder reloads every arm for overlap checks; raw operations JSON
    # is accepted by load_config with the same local macro file.
    arms = {name:dict(config=path, local=c['recommendation_local'])
            for name, path in config['operations'].items()}
    parents = {}
    for name, spec in arms.items():
        if efficiency.file_hash(spec['config']) != config['pins'][spec['config']]:
            raise ValueError('operations changed after efficiency reconstruction')
        parents[name], _ = load_config(spec['config'], spec['local'])
        if parents[name]['manifest_sha256'] != result['manifests'][name]:
            raise ValueError('selected operations differ from reconstructed evidence')
    result = dict(result, configured_selection=selected, selection_reason=reason.strip())
    return (dict(arms=arms), provenance, result, selected, parents[selected],
            result['manifests'][selected], 'explicit_configured_runtime_after_complete100_reconstruction')


def prepare(c, provenance):
    selected = choice(c)  # Audit before creating any output directory.
    validator = baseline.validate if selected[3] == 'service32' else acoustic.validate
    builder = bind(shared.prepare, efficiency_choice=lambda _:selected, validate=validator)
    return builder(c, provenance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    args = parser.parse_args()
    config, provenance = load_config(args.config, args.local)
    import torch
    torch.set_num_threads(1)
    print(json.dumps(prepare(config, provenance), indent=2))
