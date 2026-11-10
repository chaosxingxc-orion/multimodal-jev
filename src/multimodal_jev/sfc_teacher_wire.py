"""Bounded SFC teacher requests and local wire-custody verification.

Local hashes establish consistency of persisted executor records, not a provider
signature, semantic approval, or evidence that the teacher heard the audio.
"""
import hashlib
from pathlib import Path

from .baseline import file_hash
from .consumption import canonical
from .sfc_jev_runtime import INSTRUCTION
from .sfc_prompt_candidates import validate
from .sfc_teacher_feedback import TEMPLATE
from .teacher_candidates import strict_json

EXECUTOR_SHA = 'c9f9bbd333f038955945db82d946cb2757bafbc340b1a8ca476f52f3ab406bd4'


def read(path):
    return strict_json(Path(path).read_text())


def private_write(path, value):
    # Exclusive creation preserves partial/failed attempts for reconciliation.
    import os
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(canonical(value))
        handle.flush()
        os.fsync(handle.fileno())


def prepared_feedback(root, prepared_hash, partition_hash):
    if file_hash(root/'prepared.json') != prepared_hash:
        raise ValueError('registered prepared feedback changed')
    prepared = read(root/'prepared.json')
    if (prepared['partition_sha256'] != partition_hash or prepared['API_sent'] is not False
            or prepared['selection_or_test_feedback_used'] is not False
            or prepared['bundles'] != 6 or prepared['decisions'] != 78
            or prepared['max_candidates'] != 2 or prepared['max_instruction_characters'] != 1500
            or file_hash(root/'messages.private.json') != prepared['messages_sha256']):
        raise ValueError('accepted six-turn 78-decision learning feedback required')
    messages = read(root/'messages.private.json')
    if (type(messages) is not list or len(messages) != 2
            or messages[0] != dict(role='system', content=TEMPLATE)
            or set(messages[1]) != {'role', 'content'} or messages[1]['role'] != 'user'):
        raise ValueError('exact SFC teacher messages required')
    context = strict_json(messages[1]['content'])
    if (len(messages[1]['content'].encode()) != prepared['context_bytes']
            or prepared['context_bytes'] > 150000
            or context['schema'] != 'sfc-jev-offline-teacher-feedback-v1'
            or context['split'] != 'train_id' or context['selection_or_test_feedback'] is not False
            or context['weight_updates'] != 0 or context['teacher_has_audio'] is not False
            or context['parent_instruction'] != INSTRUCTION
            or context['parent_prompt_sha256'] != hashlib.sha256(INSTRUCTION.encode()).hexdigest()
            or context['source_groups'] != 4 or context['counts']['bundles'] != 6
            or len(context['all_decision_summary']) != 78
            or len(context['representatives']) != prepared['representatives']):
        raise ValueError('pinned SFC learning context required')
    evidence = [r['evidence_id'] for r in context['all_decision_summary']]
    if len(set(evidence)) != 78:
        raise ValueError('unique complete learning decision evidence required')
    return messages, context


def configure(root, reference, executor, prepared_hash, partition_hash):
    root, executor = Path(root).resolve(), Path(executor).resolve()
    if any((root/n).exists() for n in ('executor-config.private.json', 'teacher-call-intent.json',
                                      'process-launch.json', 'process-terminal.json', 'response')):
        raise FileExistsError('existing teacher attempt; no automatic retry')
    prepared_feedback(root, prepared_hash, partition_hash)
    if file_hash(executor) != EXECUTOR_SHA:
        raise ValueError('registered Omni executor changed')
    prior = read(reference)
    if prior['model'] != 'glm-5.3-flash':
        raise ValueError('registered teacher model required')
    cfg = {k: prior[k] for k in ('endpoint', 'model', 'api_key_env')}
    cfg.update(schema='saea-external-learner-v1',
        messages=dict(path=str(root/'messages.private.json'), sha256=file_hash(root/'messages.private.json')),
        max_tokens=32768, timeout_seconds=180, total_timeout_seconds=900, stream=True,
        thinking={'type': 'enabled'}, output_root=str(root/'response'))
    private_write(root/'executor-config.private.json', cfg)
    intent = dict(schema='sfc-jev-teacher-call-intent-v1', API_sent=False,
        executor=str(executor), executor_sha256=EXECUTOR_SHA,
        config_sha256=file_hash(root/'executor-config.private.json'),
        prepared_sha256=prepared_hash, partition_sha256=partition_hash,
        messages_sha256=cfg['messages']['sha256'], parent_prompt_sha256=hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        max_new_requests=1, automatic_retry=False, teacher_model=cfg['model'],
        max_tokens=32768, thinking='enabled', timeout_seconds=180, total_timeout_seconds=900,
        max_candidates=2, max_instruction_characters=1500, selection_or_test_feedback_used=False,
        scope='One bounded SFC learning-only prompt reflection; no dispatch or candidate approval.')
    private_write(root/'teacher-call-intent.json', intent)
    return intent


def check_config(root, intent_hash):
    root = Path(root).resolve()
    if file_hash(root/'teacher-call-intent.json') != intent_hash:
        raise ValueError('registered teacher intent changed')
    intent = read(root/'teacher-call-intent.json')
    if (intent['schema'] != 'sfc-jev-teacher-call-intent-v1'
            or intent['executor_sha256'] != EXECUTOR_SHA
            or file_hash(intent['executor']) != EXECUTOR_SHA
            or file_hash(root/'executor-config.private.json') != intent['config_sha256']
            or intent['max_new_requests'] != 1 or intent['automatic_retry'] is not False
            or intent['selection_or_test_feedback_used'] is not False
            or intent['max_candidates'] != 2 or intent['max_instruction_characters'] != 1500
            or intent['max_tokens'] != 32768 or intent['thinking'] != 'enabled'
            or intent['teacher_model'] != 'glm-5.3-flash'
            or intent['timeout_seconds'] != 180 or intent['total_timeout_seconds'] != 900
            or intent['parent_prompt_sha256'] != hashlib.sha256(INSTRUCTION.encode()).hexdigest()):
        raise ValueError('teacher executor or configuration custody changed')
    cfg = read(root/'executor-config.private.json')
    messages, context = prepared_feedback(root, intent['prepared_sha256'], intent['partition_sha256'])
    expected = dict(schema='saea-external-learner-v1', model='glm-5.3-flash',
        messages=dict(path=str(root/'messages.private.json'), sha256=file_hash(root/'messages.private.json')),
        max_tokens=32768, timeout_seconds=180, total_timeout_seconds=900, stream=True,
        thinking={'type': 'enabled'}, output_root=str(root/'response'))
    if (set(cfg) != set(expected) | {'endpoint', 'api_key_env'}
            or any(cfg.get(k) != v for k, v in expected.items())
            or intent['messages_sha256'] != cfg['messages']['sha256']):
        raise ValueError('bounded SFC executor contract changed')
    return cfg, intent, messages, context


def parse_wire(wire):
    """Rebuild only final content and accounting, never private reasoning."""
    chunks, finishes, pending = [], [], []
    response_id = model = usage = None
    done = False
    for line in wire.splitlines():
        if done:
            if line.strip():
                raise ValueError('data after terminal SSE marker')
            continue
        if line.startswith('data:'):
            pending.append(line[5:].lstrip(' '))
        elif not line and pending:
            data = '\n'.join(pending); pending = []
            if data == '[DONE]':
                done = True
                continue
            event = strict_json(data)
            if type(event) is not dict or 'error' in event:
                raise ValueError('provider error or invalid SSE event')
            for key, previous in (('id', response_id), ('model', model)):
                value = event.get(key)
                if value is not None and (type(value) is not str or not value or previous not in (None, value)):
                    raise ValueError('stream identity changed')
            response_id = event.get('id') or response_id
            model = event.get('model') or model
            if event.get('usage') is not None:
                if usage is not None and usage != event['usage']:
                    raise ValueError('stream accounting changed')
                usage = event['usage']
            for choice in event.get('choices', []):
                if type(choice.get('index')) is not int or choice['index'] != 0 or finishes:
                    raise ValueError('one completion with one terminal finish required')
                delta = choice.get('delta', {})
                if delta.get('tool_calls') or delta.get('function_call'):
                    raise ValueError('teacher must return text, not tool calls')
                if delta.get('content') is not None:
                    if type(delta['content']) is not str:
                        raise ValueError('text completion required')
                    chunks.append(delta['content'])
                if choice.get('finish_reason') is not None:
                    finishes.append(choice['finish_reason'])
    if pending or not done or finishes != ['stop'] or not response_id or not model:
        raise ValueError('complete identified nontruncated SSE response required')
    if usage is not None:
        if (type(usage) is not dict or any(type(usage.get(k)) is not int or usage[k] < 0
                for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
                or usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']):
            raise ValueError('invalid stream token accounting')
    return dict(id=response_id, model=model, usage=usage, content=''.join(chunks))


def load_proposal(root, intent_hash):
    root = Path(root).resolve()
    cfg, intent, messages, context = check_config(root, intent_hash)
    terminal = read(root/'process-terminal.json')
    launched = read(root/'process-launch.json')
    if (terminal['exit_code'] != 0 or terminal['intent_sha256'] != intent_hash
            or launched['intent_sha256'] != intent_hash or type(launched.get('pid')) is not int):
        raise ValueError('successful exact teacher process required')
    response_root = root/'response'
    if read(response_root/'intent.json') != dict(config=cfg, messages_sha256=cfg['messages']['sha256']):
        raise ValueError('executor dispatch identity differs')
    expected_body = dict(model=cfg['model'], messages=messages, max_tokens=cfg['max_tokens'],
                         stream=True, stream_options={'include_usage': True}, thinking=cfg['thinking'])
    if read(response_root/'request.json') != expected_body:
        raise ValueError('actual teacher request differs from registered learning messages')
    result = read(response_root/'result.json')
    if (result['state'] != 'response_received' or result['http_status'] != 200
            or result['finish_reasons'] != ['stop']
            or file_hash(response_root/'response.sse') != result['response_sha256']):
        raise ValueError('successful complete wire response required')
    wire = parse_wire((response_root/'response.sse').read_text())
    response = read(response_root/'response.json')
    if (wire['model'] != cfg['model'] or result['model'] != wire['model']
            or response.get('id') != wire['id'] or response.get('model') != wire['model']
            or response.get('usage') != wire['usage'] or result.get('usage') != wire['usage']
            or len(response['choices']) != 1):
        raise ValueError('teacher model, identity or accounting differs from wire')
    choice = response['choices'][0]
    if (choice.get('index') != 0 or choice['finish_reason'] != 'stop'
            or choice['message'].get('role') != 'assistant'
            or choice['message']['content'] != wire['content']):
        raise ValueError('teacher final content differs from wire')
    payload = strict_json(wire['content'])
    return context, payload, dict(intent_sha256=intent_hash, messages_sha256=cfg['messages']['sha256'],
        request_sha256=file_hash(response_root/'request.json'), response_sha256=result['response_sha256'],
        response_id=wire['id'], teacher_model=wire['model'], usage=wire['usage'],
        usage_known=wire['usage'] is not None,
        output_budget_exceeded=None if wire['usage'] is None else wire['usage']['completion_tokens'] > cfg['max_tokens'])


def stage_wire(root, output, intent_hash):
    context, payload, evidence = load_proposal(root, intent_hash)
    candidates = validate(payload, context)
    if evidence['output_budget_exceeded'] is True:
        raise ValueError('provider exceeded registered output budget; preserve spend, do not promote')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    private_write(output/'proposal.private.json', payload)
    entries = []
    for row in candidates:
        path = output/(row['id']+'.txt')
        with path.open('x') as handle:
            handle.write(row['instruction'])
        path.chmod(0o600)
        entries.append(dict(id=row['id'], instruction_sha256=file_hash(path), characters=len(row['instruction'])))
    receipt = dict(schema='sfc-jev-teacher-wire-staging-v1', **evidence, candidates=entries,
        parent_prompt_sha256=context['parent_prompt_sha256'], wire_authenticated=True,
        semantic_review='pending', candidate_accepted=False, inference_dispatched=False,
        scope='Persisted executor/request/SSE custody verified; not cryptographic provider authentication. '
              'Private reasoning excluded. No semantic approval, inference, or improvement claim.')
    private_write(output/'staged.json', receipt)
    return receipt
