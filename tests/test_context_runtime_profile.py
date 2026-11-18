from types import SimpleNamespace as N
import pytest

from multimodal_jev.context_runtime_profile import PROFILE,install_rollout
from multimodal_jev.span_runtime_profile import RPCS


def test_named_profile_orders_every_component_before_serving():
    calls=[]
    def rpc(name):
        calls.append(name)
        if name=='align_jev_candidate_packed_attention':
            return [dict(layers=28,attention='TND_sparse3_inner2')]
        if name=='align_jev_candidate_fixed_base':
            return [dict(language_qkv=28,tile_rows=128,tile_batch=8)]
        return [dict(operation=name)]
    engine=N(llm_engine=N(collective_rpc=rpc))
    report=install_rollout(engine,PROFILE)
    assert calls==list(RPCS)+['align_jev_short_audio_padding',
        'align_jev_candidate_packed_attention','align_jev_candidate_fixed_base']
    assert report['profile']==PROFILE and not report['training_dispatch_allowed']
    with pytest.raises(ValueError):install_rollout(engine,'unknown')


def test_incomplete_worker_installation_rejected():
    engine=N(llm_engine=N(collective_rpc=lambda name: []))
    with pytest.raises(ValueError):install_rollout(engine,PROFILE)
