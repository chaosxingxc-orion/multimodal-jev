#!/usr/bin/env python3
"""Bind prepared Omni learning feedback to one bounded remote teacher call."""
import argparse
import json
import os
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical

EXECUTOR_SHA = 'c9f9bbd333f038955945db82d946cb2757bafbc340b1a8ca476f52f3ab406bd4'


def configure(root, reference, executor):
    prepared = json.loads((root/'prepared.json').read_text())
    messages = root/'messages.private.json'
    if (file_hash(messages) != prepared['messages_sha256']
            or prepared['API_sent'] is not False
            or prepared['learning_groups'] != 64 or prepared['summary_records'] != 384
            or prepared['selection_or_test_feedback_used'] is not False
            or prepared['max_candidates'] != 2 or prepared['max_instruction_characters'] != 1500):
        raise ValueError('complete bounded Omni learning feedback required')
    context = json.loads(json.loads(messages.read_text())[1]['content'])
    if (context['schema'] != 'jev-generative-audio-ace-feedback-v1'
            or context['split'] != 'train' or context['selection_or_test_feedback'] is not False
            or context['weight_updates'] != 0 or context['zeros_semantic_gold_available'] is not False
            or context['actor_model'] != 'Qwen/Qwen3-Omni-30B-A3B-Instruct'):
        raise ValueError('frozen generative Omni TRAIN context required')
    if file_hash(executor) != EXECUTOR_SHA:
        raise ValueError('Omni executor pin changed')
    prior = json.loads(reference.read_text())
    if prior['model'] != 'glm-5.3-flash':
        raise ValueError('registered teacher model required')
    cfg = {k:prior[k] for k in ('endpoint','model','api_key_env')}
    cfg.update(schema='saea-external-learner-v1',
        messages=dict(path=str(messages.resolve()), sha256=file_hash(messages)),
        max_tokens=32768, timeout_seconds=180, total_timeout_seconds=900,
        stream=True, thinking={'type':'enabled'}, output_root=str((root/'response').resolve()))
    if (root/'response').exists() or (root/'teacher-call-intent.json').exists():
        raise FileExistsError('preserve prior teacher attempt; no automatic retry')
    destination = root/'executor-config.private.json'
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(canonical(cfg))
    receipt = dict(schema='jev-omni-teacher-call-intent-v1', API_sent=False,
        executor_sha256=EXECUTOR_SHA, config_sha256=file_hash(destination),
        messages_sha256=file_hash(messages), teacher_model=cfg['model'],
        max_new_requests=1, automatic_retry=False, max_tokens=32768,
        timeout_seconds=180, total_timeout_seconds=900,
        max_candidates=2, max_instruction_characters=1500,
        credential_available=bool(os.environ.get(cfg['api_key_env'])),
        selection_or_test_feedback_used=False,
        scope='ACE-inspired prompt reflection; no TF-GRPO estimate; no dispatch by this tool')
    with (root/'teacher-call-intent.json').open('xb') as handle:
        handle.write(canonical(receipt))
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root','reference','executor'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(configure(a.root, a.reference, a.executor), indent=2))
