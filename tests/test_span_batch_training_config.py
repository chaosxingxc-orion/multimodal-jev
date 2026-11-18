import pytest
import torch

from multimodal_jev.span_batch_training_config import PROFILE, batch_layout, reorder_global_items


def config(micro=8):
    return dict(candidate_profile=PROFILE, global_batch=128, microbatch_per_rank=micro,
                batch_order='length_grouped')


@pytest.mark.parametrize('micro', [8, 16])
def test_full_payload_permutation_preserves_zero_weight_and_identity(micro):
    items = [dict(request_id=str(i), inputs=dict(attention_mask=torch.ones(1, 80+i%13),
                  input_features_mask=torch.ones(1, 20+i%37)),
                  targets=object(), request_weight=0. if i == 127 else float(i+1),
                  real_request=i != 127, behavior=object()) for i in range(128)]
    ordered, ranks = reorder_global_items(items, config(micro))
    assert len(ranks) == 4 and all(len(r) == 32 for r in ranks)
    assert sorted(i for r in ranks for i in r) == list(range(128))
    assert all(item is items[i] for item, i in zip(ordered, [i for r in ranks for i in r]))
    assert sum(i['request_weight'] for i in ordered) == sum(i['request_weight'] for i in items)
    assert sum(i['real_request'] for i in ordered) == 127
    assert batch_layout(config(micro))['accumulation_per_rank'] == 32 // micro


@pytest.mark.parametrize('field,value', [('microbatch_per_rank', 32), ('microbatch_per_rank', True),
                                        ('candidate_profile', 'production'), ('global_batch', 256),
                                        ('batch_order', 'label_sorted')])
def test_unadmitted_configuration_rejected(field, value):
    c = config(); c[field] = value
    with pytest.raises(ValueError):
        batch_layout(c)
