"""Native FSC scoring for SpeechBrain's released semantic serialization."""
import re


FIELDS = ('action', 'object', 'location')
_SEMANTICS = re.compile(r'\{"action:" "([^"|]*)"\| "object": "([^"|]*)"\| "location": "([^"|]*)"\}')


def render_reference(row):
    """Match pinned upstream recipes/fluent-speech-commands/prepare.py exactly."""
    values = [row[key] for key in FIELDS]
    if any(not isinstance(v, str) or any(c in v for c in '"|\n\r') for v in values):
        raise ValueError('unsupported reference field')
    return '{"action:" "' + values[0] + '"| "object": "' + values[1] + '"| "location": "' + values[2] + '"}'


def parse_prediction(raw):
    if not isinstance(raw, str):
        return None
    match = _SEMANTICS.fullmatch(raw)
    return dict(zip(FIELDS, match.groups())) if match else None


def score_predictions(references, predictions):
    """Missing outputs count as wrong; duplicate and unknown IDs are errors."""
    gold = {row['path']: row for row in references}
    if not gold or len(gold) != len(references):
        raise ValueError('nonempty unique references required')
    found = {}
    for pred in predictions:
        key = pred['example_id']
        if key not in gold or key in found:
            raise ValueError('unknown or duplicate prediction ID')
        found[key] = pred['raw_prediction']
    parsed = joint = exact = 0
    component = {key: 0 for key in FIELDS}
    for key, row in gold.items():
        raw = found.get(key)
        exact += raw == render_reference(row)
        value = parse_prediction(raw)
        if value is None:
            continue
        parsed += 1
        for field in FIELDS:
            component[field] += value[field] == row[field]
        joint += all(value[field] == row[field] for field in FIELDS)
    n = len(gold)
    return {'examples': n, 'predictions': len(found), 'missing': n-len(found),
            'parsed': parsed, 'joint_correct': joint, 'joint_accuracy': joint/n,
            'serialization_exact_correct': exact, 'serialization_exact_accuracy': exact/n,
            'component_correct': component,
            'component_accuracy': {key: value/n for key, value in component.items()},
            'scope': 'native joint semantics and original serialization exact; not WER, JSON validity or Jev schema accuracy'}
