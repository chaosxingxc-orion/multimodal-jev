"""Audit model geometry for the shared 0.6B/1.7B execution path (CPU, read-only).

Derives geometry, LoRA targets and decision-head dimensions from each declared
HF config, checks the frozen experiment design against the derived targets, and
inventories remaining hardcoded head dimensions as measured debt. No weights are
loaded, no device is used and no model is called.
"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import sys

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_model_profile import (
    SCHEMA as PROFILE_SCHEMA, from_hf, record, validate_against_config)
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config

SCHEMA = 'jev-context-model-geometry-audit-v1'
HEAD_PATTERN = re.compile(r'ContextDecisionHead\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)')


def canonical(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def audit_model(entry):
    profile = from_hf(entry['path'])
    if profile.hidden_size != entry['expected_hidden_size']:
        raise ValueError('declared hidden size differs from the model config')
    result = record(profile)
    result['identity'] = profile.identity()
    result['role'] = entry['role']
    result['path'] = entry['path']
    return result, profile


def hardcoded_sites(roots):
    """Report literal head dimensions still present; measured debt, not a gate."""
    sites = []
    for root in roots:
        root = Path(root)
        if root.is_symlink() or not root.is_dir():
            raise ValueError('ordinary source root required')
        for path in sorted(root.rglob('*.py')):
            if path.is_symlink() or not path.is_file():
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                for match in HEAD_PATTERN.finditer(line):
                    dims = [int(x) for x in match.groups()]
                    sites.append(dict(file=str(path), line=number, dimensions=dims,
                                      derived=dims[0] == dims[2]))
    return sites


def environment():
    result = dict(python=sys.version.split()[0], host=platform.node(),
                  machine=platform.machine(), cpu_count=os.cpu_count())
    try:
        import torch
        result['torch'] = torch.__version__
    except ImportError:
        result['torch'] = None
    return result


def run(c, provenance):
    if c['schema'] != SCHEMA:
        raise ValueError('explicit geometry audit schema required')
    out = Path(c['output'])
    out.mkdir(parents=True, exist_ok=False)
    commit_json(out/'config.json', c)
    commit_json(out/'provenance.json', provenance)
    models, profiles = {}, {}
    for entry in c['models']:
        models[entry['role']], profiles[entry['role']] = audit_model(entry)
    if set(profiles) != {'small', 'large'}:
        raise ValueError('exactly one small and one large model required')
    small, large = profiles['small'], profiles['large']
    if small.lora_targets() != large.lora_targets():
        raise ValueError('LoRA target sets differ between model sizes')
    design = json.loads(Path(c['design']).read_text())
    consistency = {role: validate_against_config(profile, design)
                   for role, profile in profiles.items()}
    sites = hardcoded_sites(c['source_roots'])
    report = dict(schema=SCHEMA, status='audited', model_calls=0, npu_executed=False,
                  profile_schema=PROFILE_SCHEMA, environment=environment(),
                  implementation_sha256=file_hash(__file__),
                  design=dict(path=c['design'], sha256=file_hash(c['design'])),
                  models=models, design_consistency=consistency,
                  shared_lora_targets=dict(count=len(small.lora_targets()),
                                           sha256=canonical(small.lora_targets())),
                  decision_heads={role: profile.decision_head()
                                  for role, profile in profiles.items()},
                  hardcoded_sites=dict(count=len(sites), sites=sites),
                  pending=['1.7B four-device capacity and microbatch acceptance',
                           '1.7B NPU numerical parity for the shared head'])
    commit_json(out/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    args = parser.parse_args()
    c, provenance = load_config(args.config, args.local)
    print(json.dumps(run(c, provenance), indent=2))


if __name__ == '__main__':
    main()