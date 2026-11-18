"""Additional numerical-release binding on the unchanged formal runtime checks."""
import json
from pathlib import Path

try:
    from context_formal_runtime import validate as validate_original
    from context_probability_policy_v2 import PROFILE
except ModuleNotFoundError:
    from tools.jev.context_formal_runtime import validate as validate_original
    from tools.reports.context_probability_policy_v2 import PROFILE


def validate(c):
    result = validate_original(c)
    if c.get('binary_policy_arithmetic') != PROFILE:
        raise ValueError('explicit repaired arithmetic required')
    path = Path(c['formal_release']); release = json.loads(path.read_text())
    identity = release.get('numerical_repair', {})
    name = identity.get('acceptance_file')
    if identity.get('profile') != PROFILE or name not in release['files']:
        raise ValueError('bound numerical acceptance required')
    evidence = json.loads((path.parent / name).read_text())
    if (evidence.get('status') != 'arithmetic_affected_paths_accepted_no_dispatch'
            or evidence.get('binary_policy_arithmetic') != PROFILE
            or evidence.get('source_policy_id') != c['source_policy_id']):
        raise ValueError('numerical acceptance/source mismatch')
    for role in ('continuous', 'fresh_replay'):
        report = json.loads((path.parent / release['acceptance'][role]).read_text())
        if (report.get('passed') is not True or report.get('binary_policy_arithmetic') != PROFILE
                or report.get('source_policy_id') != c['source_policy_id']):
            raise ValueError('repaired NPU evidence required')
    return dict(**result, binary_policy_arithmetic=PROFILE)
