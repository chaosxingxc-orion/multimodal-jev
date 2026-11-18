"""Size a TP1 BF16 full-attention KV probe from model and actual encode batches.

This is a memory lower-bound calculation, not device admission or throughput
prediction. Model weights, activation/workspace and encoder cache are additional.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config


def capacity(text_config, batches, *, block_size, max_queries, max_model_len,
             max_batch_tokens, proposed_bytes):
    keys = ('num_hidden_layers', 'num_key_value_heads', 'head_dim')
    if any(type(text_config.get(k)) is not int or text_config[k] <= 0 for k in keys):
        raise ValueError('explicit positive full-attention model dimensions required')
    layers = text_config['num_hidden_layers']
    if (text_config.get('layer_types') != ['full_attention']*layers
            or text_config.get('use_sliding_window', False)):
        raise ValueError('only uniform full-attention TP1 supported')
    if any(type(v) is not int or v <= 0 for v in
           (block_size, max_queries, max_model_len, max_batch_tokens, proposed_bytes)):
        raise ValueError('positive integer capacity limits required')
    if not batches or any(not b or len(b) > max_queries or sum(b) > max_batch_tokens
            or any(type(n) is not int or not 0 < n <= max_model_len for n in b) for b in batches):
        raise ValueError('complete bounded token batches required')
    per_token = 2*2*layers*text_config['num_key_value_heads']*text_config['head_dim']
    per_block = block_size*per_token
    actual = [sum((n+block_size-1)//block_size for n in b)*per_block for b in batches]
    # Conservative full query-capacity bound (each query at maximum length).
    # This may exceed the token budget; it is deliberately not a packing claim.
    worst = max_queries*((max_model_len+block_size-1)//block_size)*per_block
    return dict(dtype='bfloat16',tensor_parallel_size=1,kv_bytes_per_token=per_token,
        block_size=block_size,bytes_per_block=per_block,actual_batch_required_bytes=actual,
        maximum_actual_batch_required_bytes=max(actual),full_query_capacity_bound_bytes=worst,
        proposed_kv_cache_memory_bytes=proposed_bytes,
        proposed_token_capacity=(proposed_bytes//per_block)*block_size,
        covers_observed_batches=proposed_bytes >= max(actual),
        covers_full_query_capacity_bound=proposed_bytes >= worst,
        scope='KV-only lower bound for configured dense TP1 BF16 model; excludes weights, '
              'activations, encoder cache, allocator overhead and device admission')


def run(c):
    for p, digest in c['pins'].items():
        if file_hash(p) != digest: raise ValueError('capacity source changed: '+p)
    model = json.loads(Path(c['model_config']).read_text())
    config = json.loads(Path(c['probe_config']).read_text())
    measured = json.loads(Path(c['measurement']).read_text())
    if model.get('dtype') != 'bfloat16': raise ValueError('BF16 source required')
    p = measured['passes'][1]; policy = config['policy']
    if (p['kind'] != 'timed' or measured['policy_id'] != policy['policy_id']
            or measured['input_sha256'] != config['input_sha256']):
        raise ValueError('actual timed measurement and policy required')
    records = p['records']; order = p['order']
    if sorted(order) != list(range(len(records))) or len(records) != 128:
        raise ValueError('complete TRAIN128 measurement required')
    lengths = [r['request_identity']['inputs']['input_ids']['shape'][-1] for r in records]
    # score_rows sorts unique rows by length before bounded encode submission.
    if len({r['request_id'] for r in records}) != len(records):
        raise ValueError('unique measured requests required')
    sorted_lengths = sorted(lengths); counts = p['plan']['batch_queries']; batches = []
    cursor = 0
    for count in counts:
        batches.append(sorted_lengths[cursor:cursor+count]); cursor += count
    if cursor != len(records) or [sum(b) for b in batches] != p['plan']['batch_tokens']:
        raise ValueError('actual encode plan differs from measured shapes')
    forwards = p['telemetry']['counts']['language']
    if Counter(n for f in forwards for n in f) != Counter(lengths):
        raise ValueError('actual model forwards do not cover inputs')
    result = capacity(model['text_config'], batches, block_size=c['block_size'],
        max_queries=policy['max_batch_queries'], max_model_len=policy['max_model_len'],
        max_batch_tokens=policy['max_batch_tokens'], proposed_bytes=c['proposed_kv_cache_memory_bytes'])
    return dict(result, observed_forward_sizes=dict(Counter(map(len,forwards))),
        current_kv_cache_memory_bytes=policy['kv_cache_memory_bytes'],
        source_sha256={k:file_hash(c[k]) for k in ('model_config','probe_config','measurement')},
        dispatch_allowed=False,
        next_probe=dict(change_only='kv_cache_memory_bytes',
            required_checks=['actual forward batch coverage', 'peak worker memory',
                'fresh audio encoding each timed pass', 'logits/probability/reverse-order comparison',
                'fixed-input latency', 'subsequent matched100 training before baseline promotion']))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    args=p.parse_args();c,provenance=load_config(args.config,args.local)
    result=run(c);result['config_provenance']=provenance
    output=Path(c['output']);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
