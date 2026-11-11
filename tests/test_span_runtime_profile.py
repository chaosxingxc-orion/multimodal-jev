from types import SimpleNamespace
import pytest
from multimodal_jev.span_runtime_profile import PROFILE, RPCS, install_rollout_profile


def test_profile_order_and_fail_closed_worker_installation():
    called=[]
    def rpc(name):
        called.append(name)
        if name == 'align_jev_language_qkv':
            raise ValueError('unsupported worker')
        return [{'installed':name}]
    engine=SimpleNamespace(llm_engine=SimpleNamespace(collective_rpc=rpc))
    with pytest.raises(ValueError,match='unsupported Span runtime'):
        install_rollout_profile(engine,'unreviewed-profile')
    assert not called
    with pytest.raises(ValueError,match='unsupported worker'):
        install_rollout_profile(engine,PROFILE)
    assert called == list(RPCS[:4])
    assert 'align_jev_language_attention' not in called


def test_profile_receipt_preserves_worker_results():
    engine=SimpleNamespace(llm_engine=SimpleNamespace(collective_rpc=lambda name:[{'rpc':name}]))
    receipt=install_rollout_profile(engine,PROFILE)
    assert receipt['profile']==PROFILE
    assert list(receipt['operations'])==list(RPCS)
    assert all(receipt['operations'][name]==[{'rpc':name}] for name in RPCS)


@pytest.mark.parametrize('replacement_module',['verl.models.transformers.npu_patch','unexpected.patch'])
def test_verl_import_restores_exact_original_functions_even_on_rejection(replacement_module):
    from multimodal_jev.span_runtime_profile import preserve_qwen3_import
    def original():
        return 'HF'
    def replacement():
        return 'fused'
    replacement.__module__=replacement_module
    owner=SimpleNamespace(forward=original)
    def importer():
        owner.forward=replacement
    if replacement_module=='unexpected.patch':
        with pytest.raises(ValueError,match='unexpected Qwen3'):
            preserve_qwen3_import(((owner,'forward'),),importer)
    else:
        receipt=preserve_qwen3_import(((owner,'forward'),),importer)
        assert len(receipt['restored'])==1
    assert owner.forward is original
    owner.forward=replacement;replacement.__module__='verl.models.transformers.npu_patch'
    with pytest.raises(ValueError,match='before importing'):
        preserve_qwen3_import(((owner,'forward'),),lambda:None)


def test_suite_admission_recomputes_probabilities_and_authenticates_policy(tmp_path):
    import itertools
    import json
    from multimodal_jev.baseline import file_hash
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.span_runtime_profile import verify_suite_admission
    base=tmp_path/'base';adapter=tmp_path/'adapter'
    for path in (base,adapter):
        path.mkdir();(path/'weights.safetensors').write_bytes(path.name.encode())
    options=list(itertools.product((False,True),repeat=3))
    result=dict(runtime_profile=PROFILE,passed=True,base_id=checkpoint_identity(base)[0],
                adapter_id=checkpoint_identity(adapter)[0],records=[dict(id=str(i),request_id=str(i),
                labels=options[i%8],supervised=[True]*3,
                hf_candidate_logprobs=[[-0.69314718056]*2]*3,
                candidate_logprobs=[[-0.69314718056]*2]*3) for i in range(48)])
    path=tmp_path/'acceptance.json'
    config=dict(runtime_profile=PROFILE,model_path=str(base),initial_adapter=str(adapter),
                runtime_acceptance=str(path),pins={})
    def save():
        path.write_text(json.dumps(result));config['pins'][str(path)]=file_hash(path)
    save()
    assert verify_suite_admission(config)['records']==48
    # Even a re-pinned artifact marked passed cannot hide changed candidates.
    result['records'][0]['candidate_logprobs'][0]=[-0.51082562377,-0.91629073187]
    save()
    with pytest.raises(ValueError,match='candidate parity failed'):
        verify_suite_admission(config)
    result['records'][0]['candidate_logprobs']=[[-0.69314718056]*2]*3
    save();(adapter/'weights.safetensors').write_bytes(b'new-policy')
    with pytest.raises(ValueError,match='policy identity differs'):
        verify_suite_admission(config)
    path.write_text('{}')
    with pytest.raises(ValueError,match='pinned runtime acceptance'):
        verify_suite_admission(config)
