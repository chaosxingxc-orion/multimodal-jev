"""Publisher-preserving MMAR inputs and full-denominator scoring.

Empty padding candidates and non-single-answer references are deliberate source
data. Never repair candidates from the answer or pass answers into inference.
"""
import math


def model_input(row):
    question, choices = row['question'], row['choices']
    if not isinstance(question, str) or not question.strip():
        raise ValueError('missing question')
    if not isinstance(choices, list) or not 2 <= len(choices) <= 36:
        raise ValueError('invalid candidate count')
    if any(not isinstance(text, str) for text in choices):
        raise ValueError('candidate descriptions must be strings')
    return {'question': question, 'choices': list(choices), 'audio': row['audio_path']}


def native_prompt(inputs, render_prompt):
    # Bypass the public API's sorting and whitespace normalization. Position is
    # the identity, so duplicate/padding descriptions can be retained verbatim.
    return render_prompt(inputs['question'], list(enumerate(inputs['choices'])))


def selected_index(probabilities, count):
    if not isinstance(probabilities, list) or len(probabilities) != count:
        raise ValueError('candidate probability count mismatch')
    if any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1
           for p in probabilities):
        raise ValueError('invalid candidate probabilities')
    if not math.isclose(sum(probabilities), 1, abs_tol=1e-5):
        raise ValueError('candidate probabilities do not sum to one')
    return max(range(count), key=probabilities.__getitem__)


def score(rows, predictions, publisher_match):
    expected = {r['id']: r for r in rows}
    if not rows or len(expected) != len(rows):
        raise ValueError('nonempty unique source IDs required')
    found = {}
    for pred in predictions:
        key = pred['id']
        if key not in expected or key in found:
            raise ValueError('unknown or duplicate prediction ID')
        row = expected[key]
        index = selected_index(pred['probabilities'], len(row['choices']))
        if pred.get('selected_index') != index:
            raise ValueError('selected index disagrees with probabilities')
        found[key] = row['choices'][index]
    totals = {}; correct = exact = missing = 0
    for row in rows:
        present = row['id'] in found
        text = found.get(row['id'], '')
        matched = present and publisher_match(row['answer'], text, row['choices'])
        exact_match = present and text == row['answer']
        correct += int(matched); exact += int(exact_match); missing += int(not present)
        for field in ('modality', 'category', 'sub-category'):
            key = str(row.get(field, 'unknown'))
            bucket = totals.setdefault(field, {}).setdefault(key, {'correct': 0, 'total': 0})
            bucket['correct'] += int(matched); bucket['total'] += 1
    for groups in totals.values():
        for bucket in groups.values():
            bucket['accuracy'] = bucket['correct'] / bucket['total']
    return {'total': len(rows), 'predictions': len(found), 'missing': missing,
            'publisher_correct': correct, 'publisher_accuracy': correct / len(rows),
            'exact_description_correct': exact, 'exact_description_accuracy': exact / len(rows),
            'by_group': totals, 'complete': not missing,
            'denominator_policy': 'all source questions; missing predictions are incorrect'}
