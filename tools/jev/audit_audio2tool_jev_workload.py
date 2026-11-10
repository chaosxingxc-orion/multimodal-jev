#!/usr/bin/env python3
"""Input-only full native/Jev workload arithmetic, not measured throughput."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from multimodal_jev.audio2tool_inference import read_population, write_once
from multimodal_jev.baseline import file_hash

CONTEXT_SHA = '10d23786e92bc6300dfa3b9da4ba6e0855523be4977a125bcba2d1b51852aec2'


def run(args):
    result_path = args.context_run / 'result.json'
    if file_hash(result_path) != CONTEXT_SHA:
        raise ValueError('accepted full Jev context audit required')
    context = json.loads(result_path.read_text())
    profiles_path = args.context_run / 'profiles.jsonl'
    if file_hash(profiles_path) != context['profiles_sha256']:
        raise ValueError('complete real-tokenizer profiles required')
    rows, budgets = read_population(args.inputs, args.budgets)
    groups = defaultdict(list)
    for r, b in zip(rows, budgets, strict=True):
        inp = r['input']
        groups[(inp['mode'], tuple((a.get('turn'), a.get('role')) for a in inp['audios']))].append(b)
    totals = defaultdict(lambda:dict(decisions=0, text_tokens=0, audio_tokens=0))
    seen = set(); decision_coverage = defaultdict(int)
    for line in profiles_path.open():
        p = json.loads(line); key = (p['mode'], tuple(tuple(v) for v in p['roles']))
        pair = (key, p['decision_id'])
        if pair in seen or p['records'] != len(groups[key]):
            raise ValueError('complete unique layout/decision profiles required')
        seen.add(pair); decision_coverage[p['decision_id']] += p['records']
        t = totals[p['decision_type']]
        t['decisions'] += p['records']
        t['text_tokens'] += p['text_tokens'] * p['records']
        t['audio_tokens'] += sum(sum(b['audio_tokens']) for b in groups[key])
    if len(seen) != 2742 or len(decision_coverage) != 457 or set(decision_coverage.values()) != {30733}:
        raise ValueError('full 30733 by 457 coverage required')
    for t in totals.values():
        t['input_tokens'] = t['text_tokens'] + t['audio_tokens']
        t['output_token_cap_at_1024'] = t['decisions'] * 1024
    inputs = sum(t['input_tokens'] for t in totals.values())
    decisions = sum(t['decisions'] for t in totals.values())
    native_tokens = sum(b['input_tokens'] for b in budgets)
    frames = sum(sum(b['frames']) for b in budgets)
    audio_segments = sum(len(r['input']['audios']) for r in rows)
    return dict(status='input_workload_accounted', records=len(rows), independent_decisions=decisions,
        by_decision_type=dict(totals), total_input_tokens=inputs, mean_input_tokens=inputs/decisions,
        output_token_cap_at_1024=decisions*1024,
        native=dict(requests=len(rows), input_tokens=native_tokens, output_token_cap_at_1024=len(rows)*1024,
                    audio_segment_occurrences=audio_segments, padded_audio_seconds=frames/100),
        request_multiplier=457, input_token_multiplier=inputs/native_tokens,
        repeated_audio_segment_occurrences_without_reuse=audio_segments*457,
        repeated_padded_audio_seconds_without_reuse=frames*457/100,
        elapsed_days_at_hypothetical_decisions_per_second={str(rate):decisions/rate/86400 for rate in (.1,1,10,50)},
        result_sha256=CONTEXT_SHA, profiles_sha256=file_hash(profiles_path),
        inputs_sha256=file_hash(args.inputs), budgets_sha256=file_hash(args.budgets),
        runner_sha256=file_hash(__file__), targets_read=False, model_loaded=False,
        scope='Exact aggregate logical token/audio workload for full independent Jev decisions, before any verified cache reuse. Output tokens are caps, not generated counts. Hypothetical rates are sensitivity scenarios, not NPU forecasts. No truncation, subsetting, inference launch or label access.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('context-run', 'inputs', 'budgets', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve workload audit')
    write_once(args.output, run(args))
