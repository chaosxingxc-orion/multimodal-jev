"""Full independent-decision Jev generation, with frozen input-budget evidence."""
from collections import defaultdict
import json
from pathlib import Path
import time

from .audio2tool_audio import load_audio
from .audio2tool_inference import accept_generation, digest, read_population
from .audio2tool_jev_engine import CONTEXT_SHA, boundaries
from .audio2tool_jev_ledger import DecisionLedger, export_predictions
from .audio2tool_jev_runtime import messages
from .baseline import file_hash


def layout(inputs):
    return inputs['mode'], tuple((a.get('turn'), a.get('role')) for a in inputs['audios'])


def population(release, context_run, registry, native_inputs, native_budgets):
    """Read full pinned inputs and all 2742 profiles, never targets."""
    rows, bank, _ = boundaries(context_run, release, registry)
    native, budgets = read_population(native_inputs, native_budgets)
    result = json.loads((Path(context_run) / 'result.json').read_text())
    profiles_path = Path(context_run) / 'profiles.jsonl'
    if file_hash(profiles_path) != result['profiles_sha256']:
        raise ValueError('registered complete Jev profiles required')
    counts = defaultdict(int)
    for row, original, budget in zip(rows, native, budgets, strict=True):
        if (row['id'] != original['id'] or row['tier'] != original['tier'] or row['split'] != 'external_test'
                or row['native_request_sha256'] != digest(original)
                or row['input']['context'] != original['input']):
            raise ValueError('unchanged native/Jev full audio-only pairing required')
        counts[layout(original['input'])] += 1
    profiles = {}
    decisions = {d['id']:d['decision_type'] for d in bank['decisions']}
    for line in profiles_path.open():
        p = json.loads(line); key = ((p['mode'], tuple(tuple(r) for r in p['roles'])), p['decision_id'])
        if (key in profiles or p['decision_type'] != decisions[p['decision_id']]
                or p['records'] != counts[key[0]] or type(p['text_tokens']) is not int or p['text_tokens'] <= 0
                or p['max_input_tokens'] + 1024 > 32768):
            raise ValueError('complete bounded unique Jev profiles required')
        profiles[key] = p
    if set(profiles) != {(key, d) for key in counts for d in decisions} or len(profiles) != 2742:
        raise ValueError('complete layout by decision Cartesian coverage required')
    return rows, bank, budgets, profiles


def generate_all(processor, engine, sampling, rows, bank, budgets, profiles, audio_root,
                 registry, ledger_root, progress, finite):
    """Every release/decision gets its own engine call; no score-based filtering.

    One row's authenticated waves are held across its 457 calls, but each actual
    HF processing and engine generation remains independent. No KV/feature cache
    is introduced here. finite is torch.isfinite(...).all() supplied by runner.
    """
    if processor.feature_extractor.hop_length != 160:
        raise ValueError('registered processor hop required')
    total = len(rows) * len(bank['decisions']); done = 0
    with DecisionLedger(ledger_root, rows, bank) as ledger:
        for row, budget in zip(rows, budgets, strict=True):
            if row['id'] != budget['id']:
                raise ValueError('ordered complete native budget binding required')
            context = row['input']['context']
            # Validate renderer before reading audio and keep originals intact.
            messages(row, bank['decisions'][0]['id'], [''] * len(context['audios']), bank, registry)
            progress('audio', done=done, total=total, id=row['id'])
            waves, receipts = load_audio(context, audio_root)
            for decision in bank['decisions']:
                progress('input', done=done, total=total, id=row['id'], decision=decision['id'])
                prompt = processor.apply_chat_template(messages(row, decision['id'], waves, bank, registry),
                                                       tokenize=False, add_generation_prompt=True)
                expected_length = profiles[(layout(context), decision['id'])]['text_tokens'] + sum(budget['audio_tokens'])
                batch = processor(text=[prompt], audio=waves, sampling_rate=16000,
                                  return_tensors='pt', padding=True, truncation=False)
                ids = batch['input_ids'][0].tolist()
                frames = batch['feature_attention_mask'].sum(-1).tolist()
                if (frames != budget['frames'] or frames != [r['expected_frames'] for r in receipts]
                        or len(ids) != expected_length or len(ids) + 1024 > 32768
                        or not finite(batch['input_features'])):
                    raise ValueError('real complete HF input differs from registered Jev budget')
                del batch
                request = dict(prompt=prompt, multi_modal_data={'audio':[(w, 16000) for w in waves]})
                attempt = ledger.begin(row, decision['id'], prompt, receipts, ids)
                begin = time.monotonic()
                progress('generation', done=done, total=total, id=row['id'], decision=decision['id'])
                try:
                    response = accept_generation(engine.generate([request], sampling, use_tqdm=False), ids)
                except Exception as exc:
                    ledger.failure(attempt, exc, time.monotonic()-begin)
                    raise
                ledger.finish(attempt, response, time.monotonic()-begin); done += 1
                del request
            del waves
        summary = ledger.complete()
    return summary


def export(root, rows, bank, output):
    return export_predictions(root, rows, bank, output)
