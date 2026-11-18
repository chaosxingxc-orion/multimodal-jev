#!/usr/bin/env python3
"""Materialize complete selection request identities; no predictions or new gold."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import string


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def key(*parts):
    return hashlib.sha256(json.dumps(parts,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def make_panel(panel, question_order='family_grouped_legacy'):
    if question_order not in ('family_grouped_legacy','interleaved_I_S'):
        raise ValueError('explicit known question order required')
    families=panel['families']
    questions=[{'operator':op,'definition':v} for f,op in [('intent','I_class'),('slot','S_present')]
               for v in families[f]['definitions']]
    if question_order=='interleaved_I_S':
        questions=[{'operator':op,'definition':families[f]['definitions'][i]}
                   for i in range(max(len(families[f]['definitions']) for f in ('intent','slot')))
                   for f,op in [('intent','I_class'),('slot','S_present')]
                   if i<len(families[f]['definitions'])]
    small=[{'operator':op,'definition':families[f]['definitions'][0]}
           for f,op in [('intent','I_class'),('slot','S_present')] if families[f]['definitions']]
    return dict(panel_id=panel['panel_id'],knowledge='same_complete_intent_and_panel_slot_neighbor_rules_across_arms',
        arms={'direct':[],'q_small':small,'q_full':questions})


def interventions(ids, seed=317):
    canonical=sorted(ids)
    if len(set(ids))!=len(ids):raise ValueError('distinct semantic IDs required')
    permuted=sorted(ids,key=lambda x:key('eval-order-v1',seed,x))
    rebind=sorted(ids,key=lambda x:key('eval-symbol-v1',seed,x))
    def layout(order,symbols):
        return [dict(position=i,symbol=symbols[x],semantic_id=x) for i,x in enumerate(order)]
    original_symbols={x:f'C{i:02d}' for i,x in enumerate(canonical)}
    rebound_symbols={x:f'C{i:02d}' for i,x in enumerate(rebind)}
    return {'canonical':layout(canonical,original_symbols),
            'candidate_order':layout(permuted,original_symbols),
            'neutral_symbol_rebinding':layout(canonical,rebound_symbols)}


def explicit_readout_design(layout):
    """Declare independent display positions and token bindings, without rendering.

    Cxx values in the existing design are neutral candidate aliases, not native
    single-token logits. A..z0..9 are the corresponding readout symbols. The
    production encoder currently assigns those symbols by list position; this
    declaration does not implement or accept a replacement encoder.
    """
    alphabet=string.ascii_uppercase+string.ascii_lowercase+string.digits
    n=len(layout)
    if not 2<=n<=len(alphabet):raise ValueError('two to 62 candidates required')
    if any(set(x)!={'position','symbol','semantic_id'} for x in layout):
        raise ValueError('exact candidate layout fields required')
    if (any(type(x['position']) is not int or type(x['symbol']) is not str or
            type(x['semantic_id']) is not str or not x['semantic_id'] for x in layout) or
            sorted(x['position'] for x in layout)!=list(range(n)) or
            {x['symbol'] for x in layout}!={f'C{i:02d}' for i in range(n)} or
            len({x['semantic_id'] for x in layout})!=n):
        raise ValueError('complete unique layout required')
    ordered=sorted(layout,key=lambda x:x['position'])
    entries=[dict(display_position=x['position'],semantic_id=x['semantic_id'],
                  candidate_alias=x['symbol'],output_symbol=alphabet[int(x['symbol'][1:])]) for x in ordered]
    return dict(entries=entries,output_symbol_to_candidate_alias={x['output_symbol']:x['candidate_alias'] for x in entries},
                scorer_alias_to_semantic_id={x['candidate_alias']:x['semantic_id'] for x in entries},
                fixed_head_symbol_order=list(alphabet[:n]),
                existing_position_bound_symbol_mismatches=sum(alphabet[i]!=x['output_symbol'] for i,x in enumerate(entries)),
                scope='Design only. Semantic IDs are scorer provenance, never automatically actor text. Candidate aliases and explicit output symbols are visible. Requires separate processor/backend acceptance.')


def main():
    p=argparse.ArgumentParser()
    for arg in ('release','panel-design','output'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--panel-sha256',required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.release/'MANIFEST.json')!=a.manifest_sha256 or sha(a.panel_design)!=a.panel_sha256:
        raise ValueError('pinned release and design required')
    manifest=json.loads((a.release/'MANIFEST.json').read_text())
    design=json.loads(a.panel_design.read_text())
    pins={str(a.release/'MANIFEST.json'):a.manifest_sha256,str(a.panel_design):a.panel_sha256}
    for name in ('selection.jsonl','learning.jsonl','tasks.json'):pins[str(a.release/name)]=manifest['files'][name]
    menu=Path(design['menu_report'])
    pins[str(menu)]=design['menu_report_sha256']
    pins.update(json.loads(menu.read_text())['pins'])
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('changed dependency '+path)
    readrows=lambda name:list(map(json.loads,(a.release/name).read_text().splitlines()))
    selection=readrows('selection.jsonl');learning=readrows('learning.jsonl')
    if len(selection)!=9226 or len({r['id'] for r in selection})!=9226:raise ValueError('full unique selection required')
    for field in ('id','source_group','audio_sha256'):
        if {r[field] for r in selection}&{r[field] for r in learning}:raise ValueError('split overlap: '+field)
    if any(r['role']!='selection' or r['official_split']!='train' or r['known_pending_source'] for r in selection):raise ValueError('selection role required')
    panels=[make_panel(p) for p in design['panels']]
    ids=list(json.loads((a.release/'tasks.json').read_text())['intent']['candidate_pairs'])
    if sorted(v for p in panels for v in [q['definition'] for q in p['arms']['q_full'] if q['operator']=='I_class'])!=sorted(ids):raise ValueError('complete intent panels required')
    layouts=interventions(ids)
    a.output.mkdir(parents=True)
    roster=[dict(id=r['id'],source_group=r['source_group'],audio_sha256=r['audio_sha256']) for r in selection]
    (a.output/'roster.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in roster))
    count=0
    with (a.output/'requests.jsonl').open('w') as out:
        for r in roster:
            for panel in panels:
                for arm in panel['arms']:
                    for transform in layouts:
                        item=dict(request_id=key('slurp-eval-design-v1',r['id'],panel['panel_id'],arm,transform),
                            recording_id=r['id'],panel_id=panel['panel_id'],arm=arm,transform=transform)
                        out.write(json.dumps(item,sort_keys=True)+'\n');count+=1
    report=dict(schema='slurp-selection-evaluation-design-v1',training_dispatch_allowed=False,
        execution_allowed=False,pins=pins,implementation_sha256=sha(__file__),
        recordings=len(roster),source_groups=len({r['source_group'] for r in roster}),
        panels=panels,layouts=layouts,requests_per_checkpoint=count,
        primary_canonical_requests=len(roster)*len(panels)*3,
        question_count_histograms={arm:dict(Counter(len(p['arms'][arm]) for p in panels)) for arm in panels[0]['arms']},
        group_size_histogram=dict(Counter(Counter(r['source_group'] for r in roster).values())),
        files={n:sha(a.output/n) for n in ('roster.jsonl','requests.jsonl')},
        scope='Candidate evaluation identities only; internal official-TRAIN selection, not official TEST; no new labels, predictions, semantic admission, model calls or runtime acceptance.')
    (a.output/'design.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('recordings','source_groups','requests_per_checkpoint','primary_canonical_requests','question_count_histograms')}))


if __name__=='__main__':main()
