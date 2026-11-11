import pytest
torch = pytest.importorskip('torch')
from multimodal_jev.span_trace import SpanTrace, compare_traces


def test_trace_copies_values_and_unwraps_residual_tuple():
    trace = SpanTrace()
    x = torch.ones(1,3,2,dtype=torch.bfloat16)
    trace.capture('norm',(x,torch.zeros_like(x)))
    x.zero_()
    assert trace.tensors['norm'].shape == (3,2)
    assert trace.tensors['norm'].sum() == 6
    with pytest.raises(ValueError,match='repeated'):
        trace.capture('norm',x)


@pytest.mark.parametrize('prefix',['layer.0.','audio.layer.0.'])
def test_packed_qkv_compares_matching_sections_and_detects_drift(prefix):
    actor = {'layer.0.q_proj':torch.ones(3,4,dtype=torch.bfloat16),
             'layer.0.k_proj':torch.ones(3,2,dtype=torch.bfloat16)*2,
             'layer.0.v_proj':torch.ones(3,2,dtype=torch.bfloat16)*3}
    actor = {name.replace('layer.0.',prefix):value for name,value in actor.items()}
    packed = torch.cat(list(actor.values()),-1)
    assert all(r['maximum']==0 for r in compare_traces(actor,{prefix+'qkv_proj':packed}).values())
    packed[1,5] += 1
    results = compare_traces(actor,{prefix+'qkv_proj':packed})
    assert results[prefix+'k_proj']['maximum'] == 1
    assert results[prefix+'q_proj']['maximum'] == 0
    with pytest.raises(ValueError,match='points differ'):
        compare_traces(actor,{})


def test_hf_order_norm_residual_and_weight_rounding():
    from types import SimpleNamespace
    from multimodal_jev.span_norm_alignment import hf_order_rms_norm
    torch.manual_seed(17)
    x = torch.randn(3,64,dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(64,dtype=torch.bfloat16)
    module = SimpleNamespace(weight=weight,variance_epsilon=1e-6)
    summed = x + residual
    normalized = summed.float()*torch.rsqrt(summed.float().pow(2).mean(-1,keepdim=True)+1e-6)
    expected = weight*normalized.to(torch.bfloat16)
    actual,saved_residual = hf_order_rms_norm(module,x,residual)
    assert torch.equal(actual,expected) and torch.equal(saved_residual,summed)
    assert torch.equal(hf_order_rms_norm(module,summed),expected)
    assert not torch.equal(actual,(weight.float()*normalized).to(torch.bfloat16))


def test_hf_chunk_layout_preserves_values_and_matches_actual_hf_strides():
    from multimodal_jev.span_conv_alignment import hf_chunk_layout
    x=torch.arange(3*128*100).reshape(3,100,128).transpose(1,2).unsqueeze(1)
    actual=hf_chunk_layout(x)
    assert torch.equal(actual,x)
    assert actual.stride()==(100,38400,300,1)


def test_ranked_paths_are_unique_and_reject_invalid_ranks():
    from multimodal_jev.span_trace import rank_trace_path
    paths={rank_trace_path('/tmp/trace.pt',rank,4) for rank in range(4)}
    assert len(paths)==4 and all(p.parent.as_posix()=='/tmp' for p in paths)
    for rank in (-1,4,True):
        with pytest.raises(ValueError):rank_trace_path('/tmp/trace.pt',rank,4)


def test_tp4_reconstruction_preserves_qkv_order_and_checks_every_replica():
    from multimodal_jev.span_trace import compare_tp4_traces
    torch.manual_seed(51)
    actor={name:torch.randn(3,width,dtype=torch.bfloat16) for name,width in
           [('layer.0.q_proj',16),('layer.0.k_proj',8),('layer.0.v_proj',8),
            ('layer.0.o_proj.input',16),('layer.0.down_proj.input',32),('final_norm',8)]}
    ranks=[]
    for rank in range(4):
        ranks.append({'layer.0.qkv_proj':torch.cat([actor['layer.0.'+k].chunk(4,-1)[rank]
                      for k in ('q_proj','k_proj','v_proj')],-1),
                      **{name:value.chunk(4,-1)[rank] for name,value in actor.items() if name.endswith('.input')},
                      'final_norm':actor['final_norm'].clone()})
    report=compare_tp4_traces(actor,ranks)
    assert all(v['maximum']==0 for v in report['reconstructed'].values())
    ranks[3]['final_norm'][0,0]+=1
    report=compare_tp4_traces(actor,ranks)
    assert report['replicas']['0']['final_norm']['maximum']==0
    assert report['replicas']['3']['final_norm']['maximum']>0
    with pytest.raises(ValueError,match='four aligned'):
        compare_tp4_traces(actor,ranks[:3])
