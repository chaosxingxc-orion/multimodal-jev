"""Explicit population contracts for immutable rule inference, not training."""
import json
from pathlib import Path
from .baseline import file_hash

LEARNING_PROBE = 'learning-probe-216-v1'
SELECTION_FULL = 'selection-full-v1'


def configured_rule_rows(config):
    if config.get('population')=='transfer-full-evaluation-v1':
        from .span_transfer_evaluation import evaluation_rows
        return evaluation_rows(config)
    if config.get('population')=='transfer-version-gate-v1':
        from .span_transfer_runtime import version_rows
        return version_rows(config)
    return select_rule_rows(Path(config['release']),config.get('population',LEARNING_PROBE))


def select_rule_rows(release, population=LEARNING_PROBE):
    release=Path(release)
    manifest=json.loads((release/'MANIFEST.json').read_text())
    schema={LEARNING_PROBE:'jev-span-balanced-rule-controls-v1',
            SELECTION_FULL:'jev-span-selection-rules-v1'}.get(population)
    if schema is None or manifest['schema']!=schema:
        raise ValueError('explicit population and dataset schema must match')
    for name,digest in manifest['files'].items():
        if file_hash(release/name)!=digest:raise ValueError('dataset changed')
    name='learning.jsonl' if population==LEARNING_PROBE else 'selection.jsonl'
    rows=[json.loads(s) for s in (release/name).read_text().splitlines()]
    by_id={r['id']:r for r in rows}
    if not rows or len(by_id)!=len(rows):raise ValueError('nonempty unique rule population required')
    if population==SELECTION_FULL:
        if (manifest['training_admitted'] is not False or len(rows)!=manifest['records']
                or any(r['split']!='selection' or r.get('optimization_allowed') is not False for r in rows)):
            raise ValueError('selection remains excluded from optimization')
        return rows
    schedule=json.loads((release/'schedule.json').read_text())
    selected=[];seen=set()
    for batch in schedule['batches']:
        if batch['rule_pair_id'] in seen:continue
        seen.add(batch['rule_pair_id'])
        for arm in ('final_only','structured','reordered'):
            selected.extend(by_id[k] for k in batch['arms'][arm])
    if (len(seen)!=9 or len(selected)!=216 or len({r['id'] for r in selected})!=216
            or any(r['split']!='learning' for r in selected)):
        raise ValueError('nine rule pairs and 216 matched learning requests required')
    return selected


def validate_selection_protocol(config, rows):
    """A pinned protocol authorizes only immutable inference on a fixed roster."""
    path=config.get('evaluation_protocol')
    if not path or config.get('pins',{}).get(path)!=file_hash(path):
        raise ValueError('pinned selection inference protocol required')
    protocol=json.loads(Path(path).read_text())
    policy=config['base_id']+':'+(config.get('adapter_id') or 'no-adapter')
    if (protocol['schema']!='jev-selection-frozen-evaluation-v1'
            or protocol['release_manifest_sha256']!=file_hash(Path(config['release'])/'MANIFEST.json')
            or protocol['population']!=SELECTION_FULL or config['population']!=SELECTION_FULL
            or protocol['request_ids']!=[r['id'] for r in rows]
            or policy not in protocol['policy_ids'] or protocol['optimizer_updates']!=0
            or protocol['training_admitted'] is not False):
        raise ValueError('immutable selection inference differs from fixed protocol')
    return protocol
