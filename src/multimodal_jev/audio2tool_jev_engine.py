"""Audio-only Jev engine requests bound to accepted real processor boundaries."""
import json
from pathlib import Path

from .audio2tool_audio import load_audio
from .audio2tool_engine import validate_request
from .audio2tool_inference import digest
from .audio2tool_jev import task_bank
from .audio2tool_jev_runtime import messages
from .baseline import file_hash

CONTEXT_SHA = '10d23786e92bc6300dfa3b9da4ba6e0855523be4977a125bcba2d1b51852aec2'
MANIFEST_SHA = '56921792b06636d83bd6b8a21e49ae0f887f3de68a9813d75fe1b98d83c0f062'


def boundaries(context_run, release, registry_bytes):
    """Only actor inputs and prior processor proofs; no gold/availability reads."""
    context_run, release = Path(context_run), Path(release)
    if file_hash(context_run / 'result.json') != CONTEXT_SHA or file_hash(release / 'MANIFEST.json') != MANIFEST_SHA:
        raise ValueError('accepted full Jev processor result and release required')
    result = json.loads((context_run / 'result.json').read_text())
    manifest = json.loads((release / 'MANIFEST.json').read_text())
    if (json.loads((context_run / 'terminal.json').read_text()) != {'exit_code': 0}
            or result['status'] != 'passed' or result['manifest_sha256'] != MANIFEST_SHA):
        raise ValueError('complete successful processor dependency required')
    for name, key in [('task-bank.json', 'task_bank_file_sha256'), ('external_test.requests.jsonl', 'requests_sha256')]:
        if file_hash(release / name) != manifest[key]:
            raise ValueError('unchanged full actor inputs required')
    bank = json.loads((release / 'task-bank.json').read_text())
    if bank != task_bank(registry_bytes) or digest(bank) != manifest['identity']['task_bank_sha256']:
        raise ValueError('registered full tool bank required')
    rows = [json.loads(line) for line in (release / 'external_test.requests.jsonl').open()]
    by_id = {r['id']:r for r in rows}
    if len(rows) != len(by_id) or len(rows) != 30733:
        raise ValueError('full unique release roster required')
    cases = result['real_processor_cases']
    expected = {mode + ':' + kind for mode in ('single_audio', 'dialogue_last_user', 'foreground_mixture', 'seven_audio')
                for kind in ('noul', 'typed_call_list', 'typed_tool_sequence')}
    if len(cases) != 12 or {c['case'] for c in cases} != expected:
        raise ValueError('all twelve Jev processor boundaries required')
    selected = []
    for case in cases:
        if (case != json.loads((context_run / (case['case'].replace(':', '-') + '.json')).read_text())
                or case['token_ids_exact'] is not True or case['bf16_features_exact'] is not True):
            raise ValueError('unchanged real processor boundary evidence required')
        selected.append((by_id[case['id']], case))
    return rows, bank, selected


def prepare_request(processor, row, decision_id, root, bank, registry_bytes):
    if processor.feature_extractor.hop_length != 160:
        raise ValueError('registered processor hop required')
    # Validate the full input/bank/decision before opening audio files.
    messages(row, decision_id, [''] * len(row['input']['context']['audios']), bank, registry_bytes)
    waves, receipts = load_audio(row['input']['context'], root)
    prompt = processor.apply_chat_template(messages(row, decision_id, waves, bank, registry_bytes),
                                           tokenize=False, add_generation_prompt=True)
    if prompt.count('<|audio_pad|>') != len(waves):
        raise ValueError('complete ordered audio placeholders required')
    return dict(prompt=prompt, multi_modal_data={'audio':[(w, 16000) for w in waves]}), receipts


def validate_boundary(request, receipts, case):
    validate_request(request, receipts, case)
    if [r['expected_frames'] for r in receipts] != case['frames']:
        raise ValueError('complete Jev audio frames required')


def accept_response(responses, case):
    if len(responses) != 1 or not responses[0].finished or len(responses[0].outputs) != 1:
        raise ValueError('one terminal Jev engine response required')
    response = responses[0]; answer = response.outputs[0]
    ids = list(response.prompt_token_ids)
    if len(ids) != case['input_tokens'] or digest(ids) != case['input_ids_sha256']:
        raise ValueError('actual engine IDs differ from accepted Jev processors')
    if (len(answer.token_ids) != 1 or type(answer.token_ids[0]) is not int or answer.token_ids[0] < 0
            or answer.finish_reason not in ('stop', 'length') or type(answer.text) is not str):
        raise ValueError('one-token capacity generation required')
    return dict(input_tokens=len(ids), input_ids_sha256=digest(ids), token_ids=list(answer.token_ids),
                raw_text=answer.text, finish_reason=answer.finish_reason)
