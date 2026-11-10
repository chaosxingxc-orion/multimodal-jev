"""Strict six-view scoring for the separated FSC-Jev consumption release."""
from collections import defaultdict
import hashlib
from .answers import parse_answer, AnswerError
from .consumption import canonical, validate_request
from .task_schema import DecisionSchema

VIEWS = ('native', 'permutation', 'rebind', 'missing_candidate', 'noul.True', 'noul.False')


def score_six_views(requests, targets, predictions):
    gold = {r['id']: r for r in targets}
    if len(gold) != len(targets): raise ValueError('duplicate targets')
    groups = defaultdict(dict); identities = set()
    for request in requests:
        validate_request(request)
        uid = request['id']
        if uid in identities or uid not in gold: raise ValueError('duplicate request or missing gold')
        identities.add(uid); target = gold[uid]; meta = target['metadata']
        if target['request_sha256'] != hashlib.sha256(canonical(request)).hexdigest():
            raise ValueError('target request binding changed')
        variant = meta['variant']; group = request['source_group']
        if (variant not in VIEWS or variant in groups[group]
                or meta['source_group'] != group or meta['audio_sha256'] != request['audio_sha256']):
            raise ValueError('invalid group/view provenance')
        schema = DecisionSchema.from_dict(request['input']['schema'])
        if not schema.accepts(target['target']): raise ValueError('invalid target')
        groups[group][variant] = (request, target)
    if not groups: raise ValueError('nonempty six-view population required')
    if len({r['split'] for r in requests}) != 1: raise ValueError('single split required')
    for views in groups.values():
        if set(views) != set(VIEWS): raise ValueError('incomplete six-view group')
        if len({(r['split'], r['audio_sha256']) for r, t in views.values()}) != 1:
            raise ValueError('group audio or split differs')
        native = views['native'][1]['target']
        if (type(native) is not str or views['permutation'][1]['target'] != native
                or views['rebind'][1]['target'] == native
                or views['missing_candidate'][1]['target'] is not None
                or views['noul.True'][1]['target'] is not True
                or views['noul.False'][1]['target'] is not False):
            raise ValueError('six-view target contrast changed')
        definitions = {v: r['input']['schema'] for v, (r, t) in views.items()}
        mappings = {v: {c['id']: c['description'] for c in d['candidates']}
                    for v, d in definitions.items()}
        reference = mappings['native']
        if (any(definitions[v]['decision_type'] != 'choice' for v in VIEWS[:4])
                or mappings['permutation'] != reference
                or set(mappings['rebind']) != set(reference)
                or sorted(mappings['rebind'].values()) != sorted(reference.values())
                or mappings['rebind'].get(views['rebind'][1]['target']) != reference[native]
                or mappings['missing_candidate'] != {k: v for k, v in reference.items() if k != native}
                or definitions['missing_candidate']['allow_unknown'] is not True
                or any(definitions[v]['question'] != definitions['native']['question'] for v in VIEWS[1:4])):
            raise ValueError('six-view semantic definitions changed')
        expected_noul = lambda desc: 'Does the utterance request exactly this command: ' + desc + '?'
        if (any(definitions[v]['decision_type'] != 'noul' for v in VIEWS[4:])
                or definitions['noul.True']['question'] != expected_noul(reference[native])
                or definitions['noul.False']['question'] not in
                   {expected_noul(desc) for key, desc in reference.items() if key != native}):
            raise ValueError('Noul question contrast changed')
    found = {}
    for row in predictions:
        key = (row['example_id'], row['replicate'])
        if (key in found or key[0] not in identities or type(key[1]) is not int or key[1] < 0):
            raise ValueError('invalid prediction identity')
        if row['request_sha256'] != gold[key[0]]['request_sha256']:
            raise ValueError('prediction request binding changed')
        found[key] = row
    replicas = {k[1] for k in found}
    if (not replicas or replicas != set(range(max(replicas) + 1))
            or set(found) != {(uid, rep) for uid in identities for rep in replicas}):
        raise ValueError('incomplete prediction coverage')
    reports = []
    for rep in sorted(replicas):
        metrics = {v: dict(examples=0, correct=0, format_valid=0, schema_valid=0) for v in VIEWS}
        pairs = {v: dict(groups=len(groups), both_correct=0) for v in ('permutation', 'rebind', 'missing_candidate', 'noul')}
        all_correct = 0
        for views in groups.values():
            outcomes = {}
            for variant, (request, target) in views.items():
                row = found[request['id'], rep]
                try: value = parse_answer(row['raw_text'])['value']; valid = True
                except AnswerError: value = None; valid = False
                correct = valid and type(value) is type(target['target']) and value == target['target']
                outcomes[variant] = correct
                cell = metrics[variant]; cell['examples'] += 1; cell['correct'] += correct
                cell['format_valid'] += valid
                cell['schema_valid'] += valid and DecisionSchema.from_dict(request['input']['schema']).accepts(value)
            all_correct += all(outcomes.values())
            for v in ('permutation', 'rebind', 'missing_candidate'):
                pairs[v]['both_correct'] += outcomes['native'] and outcomes[v]
            pairs['noul']['both_correct'] += outcomes['noul.True'] and outcomes['noul.False']
        for cell in metrics.values(): cell['accuracy'] = cell['correct'] / cell['examples']
        for cell in pairs.values(): cell['both_correct_rate'] = cell['both_correct'] / cell['groups']
        reports.append(dict(replicate=rep, by_variant=metrics, pairs=pairs,
                            all_six_correct=all_correct, all_six_correct_rate=all_correct / len(groups)))
    return dict(replicates=reports, source_groups=len(groups), requests=len(identities),
                predictions=len(found), coverage=1.,
                scope='correlated derived views; paired correctness is required; not six independent recordings')
