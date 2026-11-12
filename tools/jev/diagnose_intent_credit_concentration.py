#!/usr/bin/env python3
"""Analyze fixed additive credit from accepted intent probabilities; no training."""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.slurp_intent_pilot import load_pilot
from multimodal_jev.intent_actor_inputs import schedule
from multimodal_jev.span_credit_concentration import sensitivity, concentration
from queue_span_native_actor_npu import read, completed
from queue_intent_finite_npu import verify_gate


def run(gate, pilot, manifest_sha256):
    gate = Path(gate); launch = read(gate/'launch.json')
    if not completed(launch['container_id'], launch['image']):
        raise ValueError('completed accepted gate required')
    audit = verify_gate(gate, pilot, manifest_sha256)
    raw = read(gate/'run/vllm/result.json'); records = {r['id']: r for r in raw['records']}
    results = {}
    for arm in ('D', 'Q', 'P'):
        items, admission = load_pilot(pilot, manifest_sha256, arm)
        wire = 'D' if arm == 'D' else 'Q'
        groups = {'all_learning': items, 'first_batch': schedule(items, 317, 0, 8)}
        summaries = {}
        for group, selected in groups.items():
            denominator = sum(r['target']['request_weight'] for r in selected)
            spans = {}
            for j, question in enumerate(selected[0]['request'].questions):
                rows = []
                for item in selected:
                    target = item['target']; probabilities = records[item['id']+'::'+wire]['candidate_logprobs'][j]
                    weight = target['span_weights'][j]*target['request_weight']/denominator
                    if not target['supervised'][j]: weight = 0.
                    info = sensitivity(probabilities, target['targets'][j], weight)
                    rows.append(dict(id=item['id'], gold=target['target_values'][j],
                        gold_probability=info['gold_probability'], norm=info['norm'], weight=weight,
                        input_mass_error=info['input_mass_error']))
                spans[question.id] = dict(summary=concentration(rows), rows=rows)
            final = spans[selected[0]['request'].questions[-1].id]['summary']['sum_norms']
            auxiliary = sum(v['summary']['sum_norms'] for k, v in spans.items()
                            if k != selected[0]['request'].questions[-1].id)
            summaries[group] = dict(spans=spans, auxiliary_to_final_sum_norm_ratio=auxiliary/final if final else None)
        results[arm] = summaries
    paths = [gate/'config.json', gate/'run/vllm/result.json', Path(__file__).resolve(),
             Path(__file__).resolve().parents[2]/'src/multimodal_jev/span_credit_concentration.py']
    paths += [Path(pilot)/(a+'-targets.json') for a in ('D', 'Q', 'P')]
    return dict(arms=results, gate_audit=audit, adapter_id=raw['adapter_id'], files={str(p): file_hash(p) for p in paths},
        scope='Accepted89-source learning initialization, independent output-logit coordinates only. '
              'Not shared-backbone gradient, Adam update, final-task benefit or held-out result. '
              'Candidate mass renormalized for analytic diagnostics; training objective unchanged.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('gate','pilot','manifest-sha256','output'): p.add_argument('--'+name, required=True)
    a = p.parse_args(); out = Path(a.output)
    if out.exists(): raise FileExistsError('immutable diagnosis exists')
    atomic_json(out, run(a.gate, a.pilot, a.manifest_sha256))
