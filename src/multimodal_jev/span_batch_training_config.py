"""Explicit isolated true-batch prefix configuration and payload permutation."""
from .span_length_groups import length_group_indices


PROFILE = 'qwen3-asr-bf16-fixed128x8-packed-inner2-candidate-v1'


def batch_layout(config):
    micro = config.get('microbatch_per_rank')
    if (config.get('candidate_profile') != PROFILE
            or type(config.get('global_batch')) is not int or config['global_batch'] != 128
            or type(micro) is not int or micro not in (8, 16)
            or config.get('batch_order') not in ('original', 'length_grouped')):
        raise ValueError('explicit candidate profile, global128, DP4 micro8/16 and order required')
    return dict(global_batch=128, world_size=4, local_batch=32,
                microbatch_per_rank=micro, accumulation_per_rank=32 // micro)


def reorder_global_items(items, config):
    layout = batch_layout(config)
    if len(items) != layout['global_batch']:
        raise ValueError('complete global optimizer batch required')
    if len({item['request_id'] for item in items}) != len(items):
        raise ValueError('distinct request slots including padding required')
    lengths = [(int(item['inputs']['attention_mask'].sum()),
                int(item['inputs']['input_features_mask'].sum())) for item in items]
    ranks = length_group_indices(lengths, microbatch=layout['microbatch_per_rank'],
                                 world_size=4, order=config['batch_order'])
    return [items[i] for rank in ranks for i in rank], ranks
