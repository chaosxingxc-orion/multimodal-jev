import json

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.card_training_admission import admit_graph, completed_update


def test_dynamic_endpoint_uses_complete_epoch_cursors():
    train = dict(stop_update=10, batch_config=dict(epochs=2), token_buckets={})
    assert completed_update(train, dict(epoch_end_updates=[8, 17], updates=17)) == 17
    for ends in ([17], [17, 17], [18, 17], [8, 16]):
        with pytest.raises(ValueError):
            completed_update(train, dict(epoch_end_updates=ends, updates=17))


def test_formal_graph_requires_matching_finite_evidence(tmp_path):
    finite = dict(engineering_prefix=True, training_dispatch_allowed=False, parity_passes=4,
                  start_update=0, stop_update=2, training_graph='graph', physical_devices=[0,1],
                  source_policy_id='source', instruction_objective={}, token_buckets={}, training_acoustic_engine='engine',
                  packed_acoustics={}, class_logit_ema={}, schedule_sha256='s', stage1_layout={}, parity_probability_atol=1e-6)
    path = tmp_path/'config.json'
    path.write_text(json.dumps(finite))
    receipt = tmp_path/'receipt.json'
    receipt.write_text(json.dumps(dict(engineering_prefix=True, updates=2, training_graph='graph', config_sha256=file_hash(path))))
    pin = dict(path=str(receipt), sha256=file_hash(receipt))
    formal = dict(finite, engineering_prefix=False, training_dispatch_allowed=True, parity_passes=0)
    admit_graph(formal, pin)
    with pytest.raises(ValueError):
        admit_graph(dict(formal, physical_devices=[0]), pin)
    receipt.write_text('{}')
    with pytest.raises(ValueError):
        admit_graph(formal, pin)
