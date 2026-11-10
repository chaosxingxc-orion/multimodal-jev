"""Input-only independent prompt, audio-receipt and raw-output ledger audit."""
import hashlib
from pathlib import Path
import re

from .audio2tool_inference import digest
from .audio2tool_input import validate
from .audio2tool_jev_inference import layout
from .audio2tool_jev_ledger import _strict_json, records
from .audio2tool_jev_runtime import messages
from .baseline import file_hash


def prompt_expander(processor, bank, registry, profiles):
    """Cache only fixed prompt tokens, never model states or sample predictions.

    Token-level audio expansion is supported by the already pinned 2742-profile
    special-token expansion audit. A separate fixed-image check must compare
    reconstructed IDs to real processor evidence before actor acceptance.
    """
    audio_id = processor.tokenizer.convert_tokens_to_ids('<|audio_pad|>')
    if type(audio_id) is not int or audio_id < 0:
        raise ValueError('original audio placeholder token required')
    templates = {}

    def expected(row, budget, decision_id):
        inp = row['input']['context']; validate(inp)
        key = (layout(inp), decision_id); profile = profiles[key]
        lengths = budget['audio_tokens']
        if (budget['id'] != row['id'] or len(lengths) != len(inp['audios'])
                or any(type(n) is not int or n <= 0 for n in lengths)):
            raise ValueError('original complete ordered audio budget required')
        if key not in templates:
            if len(templates) >= 2742:
                raise ValueError('registered layout/decision template bound exceeded')
            prompt = processor.apply_chat_template(messages(row, decision_id, [''] * len(lengths), bank, registry),
                                                   tokenize=False, add_generation_prompt=True)
            tokens = processor.tokenizer(prompt, add_special_tokens=False)['input_ids']
            if (tokens.count(audio_id) != len(lengths) or prompt.count('<|audio_pad|>') != len(lengths)
                    or len(tokens) - len(lengths) != profile['text_tokens']
                    or hashlib.sha256(prompt.encode()).hexdigest() != profile['prompt_sha256']):
                raise ValueError('real tokenizer/template differs from accepted complete profile')
            templates[key] = prompt, tuple(tokens)
        prompt, tokens = templates[key]; ids = []; position = 0
        for token in tokens:
            if token == audio_id:
                ids.extend([token] * lengths[position]); position += 1
            else:
                ids.append(token)
        if position != len(lengths) or len(ids) != profile['text_tokens'] + sum(lengths) or len(ids)+1024 > 32768:
            raise ValueError('complete bounded audio token expansion required')
        return prompt, ids
    return expected


def check_receipts(row, budget, receipts, headers):
    audios = row['input']['context']['audios']
    if len(receipts) != len(audios) or len(receipts) != len(budget['frames']):
        raise ValueError('complete ordered audio receipts required')
    for receipt, audio, frames in zip(receipts, audios, budget['frames'], strict=True):
        h = headers[audio['path']]; samples = h['frames']
        target = (samples * 16000 + 23999) // 24000; aligned = (target + 159) // 160 * 160
        if (set(receipt) != {'path','sha256','bytes','conversion','right_zero_padding_samples',
                            'actor_samples','expected_frames','actor_waveform_sha256'}
                or h['manifest_sha256'] != audio['sha256'] or h['sample_rate'] != 24000
                or h['channels'] != 1 or h['sample_width_bytes'] != 2
                or any(receipt[k] != audio[k] for k in ('path','sha256','bytes'))
                or receipt['conversion'] != dict(source_rate=24000, source_channels=1, source_samples=samples,
                    target_rate=16000, target_samples=target,
                    conversion='channel mean then scipy.signal.resample_poly when needed')
                or receipt['actor_samples'] != aligned or receipt['right_zero_padding_samples'] != aligned-target
                or receipt['expected_frames'] != aligned//160 or frames != aligned//160
                or not re.fullmatch('[a-f0-9]{64}', receipt['actor_waveform_sha256'])):
            raise ValueError('source audio identity or complete resampling arithmetic differs')


def audit_ledger(ledger, rows, bank, budgets, prediction_path, headers, expected_prompt, decode):
    """Consume the full ledger and exact raw prediction projection before scores.

    Waveforms are not redecoded. Receipt authenticity also requires successful
    execution of the independently pinned byte-verifying actor source. Token
    and raw-text equality is independently recomputed, never JSON repaired.
    """
    if len(rows) != len(budgets) or not rows:
        raise ValueError('full ordered request/budget population required')
    prediction_path = Path(prediction_path); pin = file_hash(prediction_path)
    count = 0; length_limited = 0; seconds = 0.; output_tokens = 0; current_row = None
    transcript = hashlib.sha256()
    with prediction_path.open() as predictions:
        for item in records(ledger, rows, bank):
            attempt, response = item['attempt'], item['response']
            row_index, column = divmod(count, len(bank['decisions']))
            row, budget = rows[row_index], budgets[row_index]
            if budget['id'] != row['id']:
                raise ValueError('ordered native audio budget binding differs')
            if current_row != row_index:
                check_receipts(row, budget, item['context']['audio_receipts'], headers)
                current_row = row_index
            prompt, ids = expected_prompt(row, budget, bank['decisions'][column]['id'])
            if (attempt['prompt_sha256'] != hashlib.sha256(prompt.encode()).hexdigest()
                    or attempt['input_ids_sha256'] != digest(ids) or attempt['input_tokens'] != len(ids)):
                raise ValueError('independently rebuilt prompt or token IDs differ')
            if decode(response['token_ids']) != response['raw_text']:
                raise ValueError('raw text differs from original output token decode')
            line = predictions.readline()
            if not line or not line.endswith('\n'):
                raise ValueError('complete raw prediction projection required')
            # Strict parser also rejects duplicate keys and nonfinite values.
            p = _strict_json(line)
            expected = {k:attempt[k] for k in ('example_id','decision_id','request_sha256')}
            expected.update(raw_text=response['raw_text'], error=response['error'])
            if p != expected:
                raise ValueError('scorer projection differs from original terminal')
            transcript.update(bytes.fromhex(digest(item)))
            count += 1; length_limited += response['length_limited']
            seconds += item['seconds']; output_tokens += len(response['token_ids'])
        if predictions.readline():
            raise ValueError('extra scorer projection rows')
    if count != len(rows)*len(bank['decisions']) or file_hash(prediction_path) != pin:
        raise ValueError('complete unchanged prediction population required')
    return dict(records=len(rows), decisions=count, length_limited=length_limited, output_tokens=output_tokens,
                generation_seconds=seconds, predictions_sha256=pin, audited_records_sha256=transcript.hexdigest(),
                ledger_complete_sha256=file_hash(Path(ledger)/'complete.json'),
                targets_read=False, full_waveforms_recomputed=False)
