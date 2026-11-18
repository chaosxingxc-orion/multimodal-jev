#!/usr/bin/env python3
"""Separate native class bindings, semantic predicates and Span positions."""
import argparse
import json
from pathlib import Path

from build_slurp_evaluation_design import interventions, explicit_readout_design
from consolidate_intent_presence_references import read, sha, require


def compile_design(panels, native_ids):
    require(len(native_ids)==60 and len(set(native_ids))==60,'complete native60 required')
    ids={f:sorted(q['definition_id'] for p in panels for q in p['process_questions']
                  if q['operator']==f) for f in ('I_present','S_present')}
    require(len(ids['I_present'])==65 and len(ids['S_present'])==55,'complete semantic support required')
    require(all(len(v)==len(set(v)) for v in ids.values()),'duplicate semantic coordinates')
    aliases={(f,k):('I' if f=='I_present' else 'S')+f'{i:02d}'
             for f,values in ids.items() for i,k in enumerate(values)}
    layouts={}
    for name,layout in interventions(native_ids).items():
        d=explicit_readout_design(layout)
        for e in d['entries']:e['native_id']=e.pop('semantic_id')
        d['scorer_alias_to_native_id']=d.pop('scorer_alias_to_semantic_id')
        layouts[name]=d
    result=[]
    for panel in panels:
        qs=[dict(question_alias=f'Q{i:02d}',operator=q['operator'],
                 definition_alias=aliases[q['operator'],q['definition_id']],
                 definition_id_offline=q['definition_id'],span_position=i,
                 boolean_output_symbols={'A':False,'B':True})
            for i,q in enumerate(panel['process_questions'])]
        result.append(dict(panel_id=panel['panel_id'],semantic_questions=qs,
            arms={arm:dict(process_span_positions=[] if arm=='Direct' else list(range(len(qs))),
                final_span_position=0 if arm=='Direct' else len(qs),
                final_binding_namespace='native_class_layout',
                active_process_definitions=[] if arm=='Direct' else [q['definition_alias'] for q in qs])
                for arm in ('Direct','Q_input','Q_credit')}))
    return dict(native_layouts=layouts,panels=result,
        semantic_aliases=[dict(operator=f,definition_id_offline=k,definition_alias=v)
                          for (f,k),v in sorted(aliases.items())])


def build(menu,output):
    require(not output.exists(),'output exists')
    report=read(menu/'report.json')
    for n,h in report['files'].items():require(sha(menu/n)==h,'menu changed')
    result=compile_design(read(menu/'panels.json'),[x['native_id'] for x in read(menu/'native-menu.json')])
    result.update(schema='jev-semantic-readout-design-v2',status='design_only_not_production_encoder',
        pins={str(menu/'report.json'):sha(menu/'report.json'),
              str(menu/'panels.json'):sha(menu/'panels.json'),
              str(menu/'native-menu.json'):sha(menu/'native-menu.json')},
        implementation_sha256=sha(__file__),
        actor_contract=dict(native_class_aliases='C00..C59, bound to explicit category descriptors and output symbols',
            semantic_predicates='I00..I64/S00..S54 refer to supplied semantic cards only, never native C aliases',
            slot_neighbors='Full shared and incident boundary rules required; compiler still needs complete semantic field rendering',
            question_aliases='Q00..Q07 local to one call, mapped by explicit definition_alias and Span position',
            input_allowlist='Audio, class descriptors, semantic definitions/rules, questions and output symbol bindings; no gold/native labels for current example, transcript or offline review provenance.',
            order_intervention='Native display positions change, native alias/output mapping fixed; semantic Q and their bindings unchanged.',
            rebinding_intervention='Native aliases/output symbols change consistently; native display positions fixed, semantic Q unchanged.',
            boolean_symbols='A/B are local per semantic Span; Final A..Z/a..z/0..7 live in a distinct readout namespace.',
            reward_mask='No effect on action membership or Span positions. Unsupervised active Q remain joint-action/KL members.',
            direct_knowledge='Same semantic knowledge as paired structured panel; omits process questions/readouts only.'),
        remaining=['production_renderer_with_separate_namespaces','processor_symbol_token_identity',
                   'all_panel_context_and_audio_alignment','inverse_mapping_scorer_runtime',
                   'resident_actor_reload_vllm_parity'],
        training_dispatch_allowed=False,evaluation_dispatch_allowed=False,semantic_admission=0)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--menu',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=build(a.menu,a.output);print(json.dumps({'panels':len(r['panels']),'semantic_aliases':len(r['semantic_aliases']),'layouts':list(r['native_layouts'])}))
