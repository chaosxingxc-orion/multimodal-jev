"""Launch admission for the acoustic/KV16 formal run.

The generic formal launch selects its runtime with the two-arm microbatch or
service comparison contract. The acoustic arm was prepared from the three-arm
efficiency recommendation, so this adapter binds that selection plus the acoustic
validation and tail audit into the generic admission. It does not modify the
baseline entry, the frozen v9 package or any run: it only wires the acoustic
selection the package was already prepared under.
"""
import argparse
import json

from multimodal_jev.local_config import load_config
from tools.reports.watch_context_acoustic_training import bind
from tools.reports import launch_context_macro_formal as launch
from tools.reports import context_acoustic_formal_operations as acoustic
from tools.reports import context_acoustic_tail_operations as acoustic_tail
from tools.reports.prepare_context_acoustic_formal import choice as acoustic_choice


# The acoustic tail audit is the one recorded for the same underlying runtime;
# the base auditor does not verify the acoustic service sources.
_audit_tail = acoustic_tail.audit_collected
# Override the two globals the generic admission resolves at call time: the
# three-arm selection and the acoustic validate/command entry.
admit = bind(launch.admit, efficiency_choice=acoustic_choice,
             formal=acoustic, audit_tail=_audit_tail)
run = bind(launch.run, admit=admit, efficiency_choice=acoustic_choice,
           formal=acoustic, audit_tail=_audit_tail)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    p.add_argument('--action', choices=['admit', 'run'], default='admit')
    args = p.parse_args()
    c, provenance = load_config(args.config, args.local)
    import torch
    torch.set_num_threads(1)
    print(json.dumps(run(c, provenance) if args.action == 'run' else admit(c), indent=2))
