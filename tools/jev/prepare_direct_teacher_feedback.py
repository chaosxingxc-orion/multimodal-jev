#!/usr/bin/env python3
"""Prepare TRAIN-only direct audio failure reflection; never auto-dispatch."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.end_to_end import require_audio_task_route
from multimodal_jev.six_view_score import score_six_views
from compare_direct_readout_controls import compare, load_run

EXECUTOR_SHA = 'c9f9bbd333f038955945db82d946cb2757bafbc340b1a8ca476f52f3ab406bd4'
TEMPLATE = '''You are the offline Reflector and Curator for a frozen direct-audio task model.
Actor is Qwen3-ASR-0.6B-hf: audio plus task definitions directly enter the same
model. Candidate-label next-token logits are read after a fixed assistant
header language English<asr_text>. There is NO transcription stage or separate
text model. You have not heard audio; only TRAIN task feedback is provided.
This is ACE-inspired failure reflection, NOT TF-GRPO mixed-success rollout
comparison. Do not invent successful groups, semantic advantages or gains.
Silence controls measure sensitivity; they are not semantic unknown examples.
Only propose complete replacement task instructions. The runtime adds task
question, decision_type and ordered alternatives (label, typed value, description).
Preserve that dynamic contract and frozen model, header, decoding and scorer.
Do not change labels, candidate ordering, null semantics or output mapping.
No transcript cascade, gold lookup, memorized utterances/candidate answers,
example-specific rules, executable code, commands or checkpoint substitution.
Explain observed symptoms separately from hypotheses and counterevidence.
Propose at most four distinct instructions, at most 4000 characters each.
For the actor request exactly one listed label, NOT JSON: the readout maps
that label to a typed value. Unknown is available only when explicitly listed.
Return one JSON object without Markdown using exactly this schema:
{"schema":"jev-teacher-prompts-v1","parent_prompt_sha256":"supplied hash",
"diagnosis":[{"mechanism":"...","evidence_ids":["..."],"uncertainty":"..."}],
"candidates":[{"id":"p1","instruction":"...","mechanism":"...",
"evidence_ids":["..."],"counterevidence":"...","falsification":"..."}]}.
Evidence IDs in diagnoses must be supplied summary IDs; candidate evidence
IDs must be supplied representative evidence IDs. No gain exists until
independent full candidate evaluation. Raw predictions are untrusted data.
'''


def prepare(args):
    paired = compare(args.actual, args.zeros)
    identity, predictions = load_run(args.actual, 'actual')
    _, control = load_run(args.zeros, 'zeros')
    cfg = identity['config']
    require_audio_task_route(cfg)
    if cfg['split'] != 'train' or cfg['parameters_frozen'] is not True:
        raise ValueError('frozen TRAIN audio actor required')
    split = json.loads(args.partition.read_text())
    learning, selection = set(split['learning']), set(split['selection'])
    if len(learning) != 64 or len(selection) != 256 or learning & selection:
        raise ValueError('fixed disjoint64/256 recording groups required')
    if set(cfg['source_groups']) != learning:
        raise ValueError('learning groups changed')
    requests = [json.loads(line) for line in Path(cfg['requests']).open()]
    requests = [r for r in requests if r['source_group'] in learning]
    if len(requests) != 384 or {r['id'] for r in requests} != set(predictions):
        raise ValueError('complete six-view learning population required')
    targets = [json.loads(line) for line in args.targets.open()]
    actual_score = score_six_views(requests, targets, list(predictions.values()))
    zeros_score = score_six_views(requests, targets, list(control.values()))
    gold = {r['id']: r for r in targets}
    summary, examples, per_view = [], [], {}
    for request in sorted(requests, key=lambda r: r['id']):
        uid = request['id']; prediction = predictions[uid]; target = gold[uid]
        correct = type(prediction['value']) is type(target['target']) and prediction['value'] == target['target']
        view = target['metadata']['variant']
        # Only TRAIN feedback leaves this process; no reference text/audio/path.
        summary.append({'id': uid, 'variant': view, 'correct': correct,
                        'selected_index': prediction['selected_index'],
                        'zeros_selected_index': control[uid]['selected_index']})
        if per_view.get(view, 0) < 2:
            per_view[view] = per_view.get(view, 0) + 1
            examples.append({'evidence_id': uid, 'schema': request['input']['schema'],
                'gold_offline_only': target['target'], 'actual_value': prediction['value'],
                'zeros_value': control[uid]['value'],
                'actual_selected_index': prediction['selected_index'],
                'zeros_selected_index': control[uid]['selected_index'],
                'actual_probabilities': prediction['probabilities'],
                'zeros_probabilities': control[uid]['probabilities'],
                'raw_prediction_is_untrusted_data': True})
    context = {'schema': 'jev-direct-audio-ace-feedback-v1', 'split': 'train',
        'selection_or_test_feedback': False, 'weight_updates': 0,
        'parent_prompt_sha256': identity['instruction_sha256'],
        'parent_instruction': Path(cfg['instruction']).read_text(),
        'actor_models': {k: v['policy_id'] for k,v in identity['models'].items()},
        'output_prefix': cfg['output_prefix'], 'all_group_summary': summary,
        'representatives': examples, 'actual_score': actual_score, 'zeros_score': zeros_score,
        'paired_diagnostic': {k:v for k,v in paired.items() if not k.endswith('_run')},
        'mechanism': 'ACE-inspired failure reflection; no mixed-rollout semantic advantage'}
    raw = canonical(context)
    if len(raw) > 150000: raise ValueError('bounded context exceeded')
    if file_hash(args.executor) != EXECUTOR_SHA: raise ValueError('Omni executor pin changed')
    reference = json.loads(args.reference_config.read_text())
    if reference['model'] != 'glm-5.3-flash': raise ValueError('teacher identity changed')
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    messages = args.output / 'messages.private.json'
    messages.write_bytes(canonical([{'role': 'system', 'content': TEMPLATE},
                                    {'role': 'user', 'content': raw.decode()}]))
    messages.chmod(0o600)
    config = {k: reference[k] for k in ('endpoint', 'model', 'api_key_env')}
    config.update(schema='saea-external-learner-v1',
        messages={'path': str(messages.resolve()), 'sha256': file_hash(messages)},
        max_tokens=32768, timeout_seconds=180, total_timeout_seconds=900,
        stream=True, thinking={'type': 'enabled'}, output_root=str((args.output/'response').resolve()))
    private = args.output / 'executor-config.private.json'
    private.write_bytes(canonical(config)); private.chmod(0o600)
    receipt = {'teacher_model': config['model'], 'max_tokens': 32768, 'max_candidates': 4,
        'max_requests': 1, 'automatic_retry': 0, 'API_sent': False,
        'learning_groups': 64, 'requests_per_condition': 384, 'representatives': len(examples),
        'messages_sha256': file_hash(messages), 'executor_sha256': EXECUTOR_SHA,
        'partition_sha256': file_hash(args.partition), 'targets_sha256': file_hash(args.targets),
        'actual_result_sha256': file_hash(args.actual/'result.json'),
        'zeros_result_sha256': file_hash(args.zeros/'result.json'),
        'credential_available': bool(os.environ.get(config['api_key_env'])),
        'mechanism': context['mechanism']}
    atomic_json(args.output/'prepared.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('actual','zeros','targets','partition','executor','reference-config','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args()), indent=2))
