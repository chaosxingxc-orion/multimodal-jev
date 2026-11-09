"""Evaluate saved predictions without changing the frozen data manifest."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from .task_schema import JevExample
from .reward import score_rollout
from .answers import parse_answer, AnswerError


def parsed_prediction(answer):
    if answer is None or answer.get('format_valid', True) is not True:
        return None
    try:
        text=answer.get('raw_text',json.dumps({k:answer[k] for k in ('value','confidence') if k in answer}))
        return parse_answer(text)
    except AnswerError:
        return None


def evaluate(manifest: Path, predictions: Path):
    answers = {}
    for line in predictions.read_text().splitlines():
        prediction = json.loads(line)
        key = prediction['example_id']
        if key in answers:
            raise ValueError(f'duplicate prediction: {key}')
        answers[key] = prediction
    stats = defaultdict(lambda: {'count': 0, 'correct': 0, 'valid': 0, 'answered': 0,
                                'answered_correct': 0, 'unknown_targets': 0,
                                'refusals': 0, 'correct_refusals': 0, 'missing': 0,
                                'format_valid': 0, 'malformed': 0, 'absent': 0,
                                'brier_sum': 0.0, 'confidence_count': 0})
    groups = defaultdict(dict)
    strata = defaultdict(lambda: {'count': 0, 'correct': 0})
    domains = defaultdict(lambda: {'count':0,'correct':0,'format_valid':0,'null_targets':0})
    seen = set()
    with manifest.open() as handle:
        for line in handle:
            ex = JevExample.from_record(json.loads(line))
            if ex.example_id in seen:
                raise ValueError(f'duplicate manifest example: {ex.example_id}')
            seen.add(ex.example_id)
            answer = answers.pop(ex.example_id, None)
            parsed = parsed_prediction(answer)
            missing = parsed is None
            value = parsed['value'] if parsed is not None else None
            confidence = parsed.get('confidence') if parsed is not None else None
            score = score_rollout(ex, value, confidence=confidence)
            correct = not missing and bool(score['semantic'])
            variant = ex.metadata.get('variant', ex.schema.decision_type)
            for key in ('all', variant):
                cell = stats[key]
                cell['count'] += 1
                cell['correct'] += correct
                cell['valid'] += not missing and ex.schema.accepts(value)
                cell['missing'] += missing
                cell['format_valid'] += not missing
                cell['malformed'] += answer is not None and missing
                cell['absent'] += answer is None
                cell['answered'] += not missing and value is not None
                cell['answered_correct'] += correct and value is not None
                cell['unknown_targets'] += ex.target is None
                cell['refusals'] += not missing and value is None
                cell['correct_refusals'] += correct and value is None
                if confidence is not None:
                    cell['brier_sum'] += (confidence-float(correct)) ** 2
                    cell['confidence_count'] += 1
            stratum = strata[variant + ('.target_null' if ex.target is None else '.target_present')]
            stratum['count'] += 1
            stratum['correct'] += correct
            if 'scenario' in ex.metadata:
                key=(ex.metadata.get('partition',ex.split),ex.metadata['scenario'],variant,
                     'null' if ex.target is None else 'present')
                domain=domains[key]
                domain['count']+=1;domain['correct']+=correct
                domain['format_valid']+=not missing;domain['null_targets']+=ex.target is None
            group = ex.metadata.get('source_group', ex.example_id)
            if variant in groups[group]:
                raise ValueError(f'duplicate group/variant: {group}/{variant}')
            groups[group][variant] = {'correct': correct, 'value': value, 'missing': missing,
                'semantic': next((c.description for c in ex.schema.candidates if c.id == value), None)}
    if answers:
        raise ValueError(f'{len(answers)} predictions not in manifest')
    for cell in stats.values():
        ratio = lambda n,d: n/d if d else None
        cell['accuracy'] = ratio(cell['correct'],cell['count'])
        cell['format_valid_rate'] = ratio(cell['format_valid'],cell['count'])
        cell['schema_valid_rate'] = ratio(cell['valid'],cell['count'])
        cell['coverage'] = ratio(cell['answered'],cell['count'])
        cell['risk'] = ratio(cell['answered']-cell['answered_correct'],cell['answered'])
        cell['unknown_precision'] = ratio(cell['correct_refusals'],cell['refusals'])
        cell['unknown_recall'] = ratio(cell['correct_refusals'],cell['unknown_targets'])
        cell['brier'] = ratio(cell['brier_sum'],cell['confidence_count'])
    pairs = {}
    for variant in ('permutation','rebind','missing_candidate'):
        eligible = [g for g in groups.values() if 'native' in g and variant in g]
        both_correct = sum(g['native']['correct'] and g[variant]['correct'] for g in eligible)
        semantic_equal = sum(not g['native']['missing'] and not g[variant]['missing'] and
                    g['native']['semantic'] is not None and g['native']['semantic']==g[variant]['semantic'] for g in eligible)
        pairs[variant] = {'groups':len(eligible),'both_correct':both_correct,
            'both_correct_rate':both_correct/len(eligible) if eligible else None,
            'semantic_equal': semantic_equal if variant != 'missing_candidate' else None}
    eligible = [g for g in groups.values() if 'noul.True' in g and 'noul.False' in g]
    both = sum(g['noul.True']['correct'] and g['noul.False']['correct'] for g in eligible)
    pairs['noul'] = {'groups': len(eligible), 'both_correct': both,
                    'both_correct_rate': both / len(eligible) if eligible else None}
    for cell in strata.values():
        cell['accuracy'] = cell['correct'] / cell['count']
    domain_rows=[]
    for (partition,scenario,variant,presence),cell in sorted(domains.items()):
        domain_rows.append({'partition':partition,'scenario':scenario,'variant':variant,'target_presence':presence,
                            **cell,'accuracy':cell['correct']/cell['count']})
    return {'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'predictions_sha256':hashlib.sha256(predictions.read_bytes()).hexdigest(),
            'by_variant':dict(stats),'pairs':pairs,'by_target_presence':dict(strata),'by_domain':domain_rows,
            'note':'Derived views are correlated; all.accuracy is descriptive, not independent evidence.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=evaluate(args.manifest,args.predictions)
    with args.output.open('x') as out: json.dump(result,out,indent=2)


if __name__=='__main__': main()
