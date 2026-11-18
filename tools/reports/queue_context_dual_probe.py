"""One-shot actual-audio probe after the existing research chain completes.

Wait on an authenticated process, then reuse priority/lease/device admission.
No polling, retry, training interruption, or full-training admission.
"""
import argparse
import json
import os
from pathlib import Path
import time

from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.committed_signal_report import write
from tools.reports.continue_recovered_research import verify, wait_exact
from tools.reports import context_dual_probe_operations as probe


def validate(c):
    if c['schema'] != 'context-dual-probe-queue-v1':
        raise ValueError('Explicit probe queue schema required')
    verify(c['pins'])
    for name in (c['probe_config'], c['probe_local']):
        if name not in c['pins']:
            raise ValueError('Probe configuration must be pinned')
    p, provenance = load_config(c['probe_config'], c['probe_local'])
    if (p.get('priority_mode', 'completed_research_chain') != 'completed_research_chain'
            or p['finite_inference_dispatch_allowed'] is not True
            or p['priority_root'] != c['predecessor_root']):
        raise ValueError('Only a finite probe after the completed research chain is allowed')
    probe.validate(p)
    return p, provenance


def run(c, provenance):
    validate(c)
    root = Path(c['output_directory'])
    check_paths({'queue': root})
    # Atomic directory creation refuses duplicate queue drivers and old failures.
    root.mkdir(parents=True, exist_ok=False)
    write(root/'input.json', c)
    write(root/'provenance.json', provenance)
    def status(phase, **extra):
        write(root/'status.json', dict(phase=phase, pid=os.getpid(), unix=time.time(), **extra))
    try:
        status('waiting_exact_research_process')
        wait_exact(c['predecessor_process'])
        p, pp = validate(c)
        # Process exit alone does not establish successful completion.
        probe.check_priority(p)
        status('executing_finite_audio_probe')
        result = probe.launch(p, pp)
        if (result.get('status') != 'finite_fixed_policy_inference_audited'
                or result.get('exact_actual_logits') is not True
                or result.get('exact_probabilities_and_G16') is not True):
            raise ValueError('Actual audio probe did not pass independent audit')
        write(root/'result.json', result)
        status('complete_finite_audio_probe', training_dispatch_allowed=False,
               switch_old_training_allowed=False)
    except BaseException as exc:
        status('stopped_preserve_handles', error=repr(exc), restart=False)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    parser.add_argument('--action', choices=['validate', 'run'], default='validate')
    args = parser.parse_args()
    config, provenance = load_config(args.config, args.local)
    if args.action == 'run':
        run(config, provenance)
    else:
        validate(config)
        print(json.dumps(dict(status='validated_only_no_dispatch')))
