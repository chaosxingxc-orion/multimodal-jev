import pytest
import torch

from multimodal_jev.span_fixed_shape_linear import fixed_shape_linear


@pytest.mark.parametrize('shape', [(1,1,5),(2,13,5),(3,8,5),(17,19,5)])
def test_matches_dense_values_and_input_weight_bias_gradients(shape):
    torch.manual_seed(317)
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    w = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    b = torch.randn(7, dtype=torch.float64, requires_grad=True)
    expected = torch.nn.functional.linear(x, w, b)
    actual = fixed_shape_linear(x, w, b, tile_rows=8, tile_batch=4)
    upstream = torch.randn_like(actual)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual, (x,w,b), upstream, retain_graph=True)
    gb = torch.autograd.grad(expected, (x,w,b), upstream)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b)


def test_varied_neighbors_padding_and_row_permutations():
    torch.manual_seed(317)
    x=torch.randn(3,13,5);w=torch.randn(7,5)
    ref=fixed_shape_linear(x[:1],w,tile_rows=8,tile_batch=4)
    for order in ([0,1,2],[1,0,2],[1,2,0]):
        batch=x[order];index=order.index(0)
        output=fixed_shape_linear(batch,w,tile_rows=8,tile_batch=4)
        torch.testing.assert_close(output[index:index+1],ref)
        padded=torch.nn.functional.pad(batch,(0,0,0,12))
        output=fixed_shape_linear(padded,w,tile_rows=8,tile_batch=4)
        torch.testing.assert_close(output[index:index+1,:13],ref)


@pytest.mark.parametrize('fault',['rows','batch','dtype','bias','shape'])
def test_invalid_contract_rejected(fault):
    x=torch.ones(2,3,5);w=torch.ones(7,5);b=None;kw={}
    if fault=='rows':kw['tile_rows']=0
    if fault=='batch':kw['tile_batch']=True
    if fault=='dtype':w=w.double()
    if fault=='bias':b=torch.ones(6)
    if fault=='shape':x=x[0]
    with pytest.raises(ValueError):fixed_shape_linear(x,w,b,**kw)
