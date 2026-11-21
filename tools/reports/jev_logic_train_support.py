"""Count logical truth cells from aggregate labels in an admitted clean SLURP TRAIN view."""
import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

from multimodal_jev.local_config import load_config
from tools.reports.jev_train_confusions import pinned


def truth(expr, env):
    if isinstance(expr, str):
        return env[expr]
    match expr:
        case ['IS', str(name)]:
            return env[name]
        case ['NOT', e]:
            return not truth(e, env)
        case ['OR', lhs, rhs]:
            a, b = truth(lhs, env), truth(rhs, env)
            return a or b
        case _:
            raise ValueError(f'unsupported expression: {expr!r}')


def count(cfg):
    if cfg['schema'] == 'jev-logic-support-clean-train-v1':
        root = Path(cfg['clean']['root'])
        source = cfg['clean']['sha256']
        m = json.loads(pinned(root / 'MANIFEST.json', source))
        if m['label_basis'] != 'official_clean_v7' or m['allowed_optimization_role'] != 'learning':
            raise ValueError('admitted native training labels required')
        raw = pinned(root / 'learning.jsonl', m['files']['learning.jsonl'])
        population = m['counts']['learning']['recordings']
        seen, classes = set(), Counter()
        for line in raw.splitlines():
            row = json.loads(line)
            if (row['role'] != 'learning' or row['official_split'] != 'train'
                    or row['known_pending_source'] or row['id'] in seen):
                raise ValueError('unique clean TRAIN labels required')
            seen.add(row['id'])
            classes[row['scenario'], row['action']] += 1
    elif cfg['schema'] == 'jev-logic-support-v1':
        source = cfg['train_view_receipt_sha256']
        release = json.loads(pinned(Path(cfg['train_view_receipt']), source))
        if (release['status'] != 'stage2_dataset_release_audited'
                or release['consumer']['reference_mode'] != 'full_train_readonly'):
            raise ValueError('audited full-TRAIN source required')
        summary = release['summary']
        population = summary['population']
        classes = {tuple(label.split('_', 1)): row['population'] for label, row in summary['by_class'].items()}
    else:
        raise ValueError('unknown logic support source')
    predicates = cfg['predicates']
    names = list(predicates)
    cells = Counter()
    for (scenario, action), recordings in classes.items():
        fields = dict(scenario=scenario, action=action)
        key = tuple(fields[p['native_field']] == p['equals'] for p in predicates.values())
        cells[key] += recordings
    if sum(cells.values()) != population:
        raise ValueError('incomplete class population')
    rows = []
    for bits in itertools.product((False, True), repeat=len(names)):
        env = dict(zip(names, bits, strict=True))
        rows.append(dict(atoms=env, recordings=cells[bits],
                         target={q: truth(e, env) for q, e in cfg['expressions'].items()}))
    return dict(population=population, native_classes=len(classes), rows=rows,
                source_sha256=source, model_calls=0,
                scope='TRAIN ontology projections only, not independent task labels or model scores')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'local'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    cfg, provenance = load_config(args.config, args.local)
    print(json.dumps(dict(**count(cfg), provenance=provenance), indent=2))
