#!/usr/bin/env python3
"""Materialize target-task output bindings; no audio, labels or model execution."""
import argparse
import hashlib
import json
from pathlib import Path

from build_slurp_evaluation_design import interventions, explicit_readout_design


def compile_design(protocol, native, semantic):
    cards = native['cards']
    native_ids = [c['semantic_id'] for c in cards]
    triples = [tuple(c['native_values'][k] for k in ('action', 'object', 'location')) for c in cards]
    semantic_ids = [c['semantic_id'] for c in semantic['cards']]
    if (len(native_ids) != 31 or len(set(native_ids)) != 31 or len(set(triples)) != 31
            or len(semantic_ids) != 31 or len(set(semantic_ids)) != 31
            or any(c['operator'] != 'I_present' or c['native_equivalence'] is not False
                   for c in semantic['cards'])):
        raise ValueError('distinct native31 and independent semantic31 required')
    panels = protocol['structure']['panels']
    planned = [s for p in panels for s in p['semantic_ids']]
    if (len(panels) != 8 or len({p['panel_id'] for p in panels}) != 8
            or [len(p['semantic_ids']) for p in panels] != [4]*7+[3]
            or sorted(planned) != sorted(semantic_ids)
            or protocol['structure']['arms'] != ['Direct', 'Q_input']):
        raise ValueError('complete fixed eight-panel target schedule required')
    aliases = {sid: f'I{i:02d}' for i, sid in enumerate(sorted(semantic_ids))}
    layouts = {}
    for name, layout in interventions(native_ids).items():
        d = explicit_readout_design(layout)
        for e in d['entries']:
            e['native_id'] = e.pop('semantic_id')
        d['scorer_alias_to_native_id'] = d.pop('scorer_alias_to_semantic_id')
        layouts[name] = d
    outputs = []
    for p in panels:
        questions = [dict(question_alias=f'Q{i:02d}', operator='I_present',
            definition_alias=aliases[sid], definition_id_offline=sid, span_position=i,
            boolean_output_symbols={'A': False, 'B': True})
            for i, sid in enumerate(p['semantic_ids'])]
        outputs.append(dict(panel_id=p['panel_id'], semantic_questions=questions,
            arms={arm: dict(process_span_positions=[] if arm == 'Direct' else list(range(len(questions))),
                final_span_position=0 if arm == 'Direct' else len(questions),
                final_binding_namespace='native_class_layout',
                active_process_definitions=[] if arm == 'Direct' else [q['definition_alias'] for q in questions])
                for arm in ('Direct', 'Q_input')}))
    return dict(native_layouts=layouts, panels=outputs,
        native_id_to_triple={c['semantic_id']: dict(c['native_values']) for c in cards},
        semantic_aliases=[dict(operator='I_present', definition_id_offline=k, definition_alias=v)
                          for k, v in sorted(aliases.items())],
        schema='fsc-semantic-readout-design-v1', status='task_design_not_production_renderer',
        primary_layout='canonical',
        diagnostic_layouts='candidate_order and neutral_symbol_rebinding are design-only; separate cost and dispatch registration required',
        namespace_contract='Cxx/native output symbols and Ixx/boolean output symbols are independent; each input supplies its own binding. Never reuse SLURP native meanings or infer semantic truth from a native ID.',
        native_descriptor_scope='Native source file supplies identities/triples only; its historical I_class prose is not imported as semantic truth or certified sufficient definitions.',
        action_contract='One forward per panel/target arm. Missing semantic supervision never removes a configured action. Direct has Final only; Q_credit identifies source training, not a third target arm.',
        population_cost='16 canonical requests per target recording per checkpoint; diagnostic layouts are not included',
        remaining=['native31_and_semantic_full_field_renderer', 'actual_processor_symbol_token_identity',
                   'full_audio_context_and_backend_parity', 'frozen_test_roster_and_semantic_truth',
                   'authenticated_prediction_parser_and_scorer'],
        semantic_admission=0, training_dispatch_allowed=False, evaluation_dispatch_allowed=False)


def build(protocol, native, semantic, output):
    if output.exists():
        raise FileExistsError(output)
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    cfg = read(protocol)
    if (cfg['target_definitions_sha256'] != sha(semantic)
            or cfg['training_dispatch_allowed'] is not False
            or cfg['evaluation_dispatch_allowed'] is not False):
        raise ValueError('pinned design-only protocol required')
    result = compile_design(cfg, read(native), read(semantic))
    result['pins'] = {str(p): sha(p) for p in (protocol, native, semantic)}
    result['implementation_sha256'] = sha(Path(__file__))
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('protocol', 'native', 'semantic', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    r = build(a.protocol, a.native, a.semantic, a.output)
    print(json.dumps(dict(panels=len(r['panels']), native_classes=len(r['native_id_to_triple']),
                         semantic_definitions=len(r['semantic_aliases']), execution_allowed=False)))
