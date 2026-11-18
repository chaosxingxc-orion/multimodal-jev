"""Bind existing label-free clean TEST inputs to the current source release.

Reuse the prior full processor admission. No TEST labels, model selection,
audio-cache writes or inference; TEST is source-domain evaluation, not transfer.
"""
from multimodal_jev.slurp_portable_release import load as load_release
from tools.reports.slurp_context_test_runtime import load_inputs as load_test
from tools.reports.preflight_slurp_context_test_inputs import reconstruct


def bind_source(rows, definitions, source):
    """Compare current TRAIN ontology and both source populations before inference."""
    ids={r['id'] for r in rows};audio={r['audio_sha256'] for r in rows}
    if len(ids)!=12991 or len(rows)!=12991:
        raise ValueError('complete unique clean TEST population required')
    counts={'learning':40513,'selection':9226}
    for role,purpose in (('learning','optimize'),('selection','evaluate')):
        population,task,manifest=load_release(source['path'],source['sha256'],role,purpose=purpose)
        expected=dict(candidates=[dict(id=c['id'],definition=c['description'])
            for c in task['schema']['candidates']],requirements=task['thinking']['requirements'],
            semantics='exclusive_event')
        if (definitions!=expected or len(population)!=counts[role]
                or manifest['counts'][role]['recordings']!=counts[role]):
            raise ValueError('clean TEST must retain current full source ontology and populations')
        if (ids & {r['id'] for r in population}
                or audio & {r['audio_sha256'] for r in population}):
            raise ValueError('clean TEST overlaps current source '+role)


def load_inputs(config):
    if config['dataset']!='SLURP_clean_TEST60':
        raise ValueError('explicit clean TEST source evaluation role required')
    source=config['source_release']
    if set(source)!={'path','sha256'}:
        raise ValueError('explicit current source release binding required')
    rows,receipts,definitions,manifest=load_test(config['inputs'])
    bind_source(rows,definitions,source)
    requests=[reconstruct(row,definitions) for row in rows]
    candidates=[c['id'] for c in definitions['candidates']]
    return rows,receipts,requests,candidates,manifest['processor_sha256']
