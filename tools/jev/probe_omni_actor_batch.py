#!/usr/bin/env python3
"""Real processor/verl batching on TRAIN audio; synthetic response tokens only."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.omni_actor_batch import prepare, evidence, make_batch
from multimodal_jev.sfc_jev_runtime import INSTRUCTION
from multimodal_jev.sfc_reward_batch import digest
from preflight_sfc_jev_omni import select


def run(model, release, partition_path, partition_hash, out):
    import torch
    from transformers import Qwen3OmniMoeProcessor
    from verl.utils.model import extract_multi_modal_inputs
    torch.set_num_threads(2)
    start = time.monotonic()
    manifest = json.loads((release / 'MANIFEST.json').read_text())
    train_path = release / 'train_id.requests.jsonl'
    target_path = release / 'train_id.targets.jsonl'
    if file_hash(train_path) != manifest['splits']['train_id']['requests_sha256']:
        raise ValueError('TRAIN inputs changed')
    if file_hash(target_path) != manifest['splits']['train_id']['targets_sha256']:
        raise ValueError('TRAIN targets changed')
    if file_hash(partition_path) != partition_hash:
        raise ValueError('partition changed')
    partition = json.loads(partition_path.read_text())
    all_rows = [json.loads(line) for line in train_path.open()]
    rows = select(all_rows)
    if not {r['source_group'] for r in rows} <= set(partition['learning']):
        raise ValueError('engineering samples must belong to learning')
    wanted = {r['id'] for r in rows}
    targets = []
    with target_path.open() as handle:
        for row, line in zip(all_rows, handle, strict=True):
            if row['id'] in wanted:
                target = json.loads(line)
                if target['id'] != row['id']:
                    raise ValueError('learning target order changed')
                targets.append(target)
    processor = Qwen3OmniMoeProcessor.from_pretrained(model, local_files_only=True)
    prepared = prepare(processor, rows)
    pins = evidence(prepared)
    records = []
    # Synthetic outcomes deliberately include unequal response lengths and format failures.
    for row in rows:
        for decision in row['input']['decisions']:
            for rep, text in enumerate(('false', 'invalid output')):
                ids = processor.tokenizer.encode(text, add_special_tokens=False)
                ids.append(processor.tokenizer.eos_token_id)
                raw = processor.tokenizer.decode(ids, skip_special_tokens=True)
                records.append(dict(example_id=row['id'], decision_id=decision['id'],
                    request_sha256=digest(row), replicate=rep, policy_sha256='a' * 64,
                    instruction_sha256=hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
                    **pins[row['id'], decision['id']], raw_text=raw, token_ids=ids,
                    log_probs=[-0.25 - 0.01 * i for i in range(len(ids))], finish_reason='stop'))
    random.Random(17).shuffle(records)
    args = dict(all_requests=all_rows, partition=partition, requests=rows, targets=targets,
                records=records, policy_sha256='a' * 64,
                instruction_sha256=hashlib.sha256(INSTRUCTION.encode()).hexdigest(), group_size=2)
    batch, report = make_batch(processor, prepared, **args)
    pmax = batch.batch['prompts'].shape[-1]
    rmax = batch.batch['responses'].shape[-1]
    for i, record in enumerate(records):
        item = prepared[record['example_id'], record['decision_id']]
        plen, length = len(item['input_ids']), len(record['token_ids'])
        assert batch.batch['input_ids'][i, pmax-plen:pmax].tolist() == item['input_ids'].tolist()
        assert batch.batch['responses'][i, :length].tolist() == record['token_ids']
        assert batch.batch['response_mask'][i].sum().item() == length
        assert batch.batch['attention_mask'][i].sum().item() == plen + length
        assert batch.batch['token_level_rewards'][i, length-1].item() == report['rewards'][i]
        assert torch.count_nonzero(batch.batch['token_level_rewards'][i, length:]) == 0
        torch.testing.assert_close(batch.batch['old_log_probs'][i, :length], torch.tensor(record['log_probs']))
        # Independent causal readout: predictor at token t must target token t+1.
        predictors = batch.batch['input_ids'][i, -rmax-1:-1]
        assert predictors[0].item() == item['input_ids'][-1].item()
        assert predictors[1:length].tolist() == record['token_ids'][:-1]
        assert batch.non_tensor_batch['record_identity'][i] == (record['example_id'], record['decision_id'], record['replicate'])
    # Real verl slicing/reordering and multimodal concatenation across single/double audio.
    by_id = {r['id']: r for r in rows}
    indices = [next(i for i, r in enumerate(records) if len(by_id[r['example_id']]['audio_sha256']) == count)
               for count in (2, 1)]
    micro = batch.select_idxs(indices)
    merged = extract_multi_modal_inputs(micro.non_tensor_batch['multi_modal_inputs'])
    offset = 0
    for i in indices:
        record = records[i]
        item = prepared[record['example_id'], record['decision_id']]
        n, _, width = item['input_features'].shape
        torch.testing.assert_close(merged['input_features'][offset:offset+n, :, :width], item['input_features'], rtol=0, atol=0)
        assert torch.equal(merged['feature_attention_mask'][offset:offset+n, :width], item['feature_attention_mask'])
        assert torch.count_nonzero(merged['feature_attention_mask'][offset:offset+n, width:]) == 0
        offset += n
    assert offset == 3
    rejected = []
    for kind in ('missing', 'logprob', 'text', 'prompt', 'context', 'prepared_tokens'):
        mutated = deepcopy(records)
        supplied = prepared
        budget = 4096
        if kind == 'missing': mutated.pop()
        if kind == 'logprob': mutated[0]['log_probs'][0] = float('nan')
        if kind == 'text': mutated[0]['raw_text'] = 'changed'
        if kind == 'prompt': mutated[0]['prompt_sha256'] = 'c' * 64
        if kind == 'context': budget = 1
        if kind == 'prepared_tokens':
            supplied = dict(prepared)
            key = records[0]['example_id'], records[0]['decision_id']
            supplied[key] = dict(prepared[key])
            supplied[key]['input_ids'] = prepared[key]['input_ids'].clone()
            supplied[key]['input_ids'][0] += 1
        try:
            make_batch(processor, supplied, context_limit=budget, **{**args, 'records': mutated})
        except ValueError:
            rejected.append(kind)
        else:
            raise AssertionError('negative control accepted: ' + kind)
    result = dict(scope='Real CPU Omni processor and verl DataProto on six TRAIN turns; synthetic responses/logprobs, no model/gradient/optimizer or engine parity',
        bundles=len(rows), decisions=len(prepared), responses=len(records), prompt_groups=report['prompt_groups'],
        tensor_shape=list(batch.batch['input_ids'].shape), reordered_microbatch_audio_count=offset,
        rejected_controls=rejected, elapsed_seconds=time.monotonic()-start,
        manifest_sha256=file_hash(release/'MANIFEST.json'), partition_sha256=partition_hash,
        processor_assets={p.name:file_hash(p) for p in model.iterdir() if p.is_file() and p.suffix in ('.json','.txt','.jinja')},
        source_files={name:file_hash(Path(__file__).parents[2]/name) for name in (
            'src/multimodal_jev/omni_actor_batch.py', 'src/multimodal_jev/sfc_reward_batch.py',
            'tools/jev/probe_omni_actor_batch.py')},
        targets_read='selected learning turns only; whole TRAIN target bytes hashed',
        model_loaded=False, weight_updates=0)
    atomic_json(out/'result.json',result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for key in ('model','release','partition','output'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--partition-hash',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try: print(json.dumps(run(a.model,a.release,a.partition,a.partition_hash,a.output),indent=2))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
