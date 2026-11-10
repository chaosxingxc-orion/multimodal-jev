"""Experimental CPU expert projection precision control, never enabled by default."""


def fp32_grouped_linear(inputs, weight, offsets, bias=None, is_transposed=False):
    """Project each expert's complete group in FP32, then restore input dtype.

    Keeps token grouping, order, routes and autograd. It does not split by sample
    or force individual-token GEMMs. This is a diagnostic candidate, not a claim
    of batch-invariant FP32 arithmetic or accelerator compatibility.
    """
    import torch

    if (inputs.device.type != 'cpu' or weight.device.type != 'cpu'
            or offsets.device.type != 'cpu' or (bias is not None and bias.device.type != 'cpu')):
        raise ValueError('CPU-only projection candidate')
    if (inputs.ndim != 2 or weight.ndim != 3 or offsets.ndim != 1
            or offsets.dtype not in (torch.int32, torch.int64)
            or offsets.numel() != weight.shape[0]
            or inputs.dtype not in (torch.bfloat16, torch.float32)
            or weight.dtype != inputs.dtype):
        raise ValueError('aligned BF16/FP32 grouped inputs, weights and integer offsets required')
    ends = offsets.tolist()
    if not ends or ends[-1] != len(inputs) or any(a > b for a, b in zip([0]+ends, ends)):
        raise ValueError('nondecreasing offsets must cover all routed tokens')
    input_dim = weight.shape[1] if is_transposed else weight.shape[2]
    output_dim = weight.shape[2] if is_transposed else weight.shape[1]
    if inputs.shape[1] != input_dim or (bias is not None and bias.shape != (len(inputs), output_dim)):
        raise ValueError('projection dimensions or per-routed-token bias differ')
    pieces = []
    begin = 0
    # Explicitly disable ambient BF16 autocast; float() alone is insufficient.
    with torch.autocast('cpu', enabled=False):
        for expert, end in enumerate(ends):
            if end > begin:
                matrix = weight[expert] if is_transposed else weight[expert].t()
                projected = torch.mm(inputs[begin:end].float(), matrix.float())
                pieces.append(projected.to(inputs.dtype))
            begin = end
        if pieces:
            output = torch.cat(pieces, dim=0)
        else:
            # Preserve the zero-sized input's autograd link.
            output = inputs.sum(-1, keepdim=True).expand(0, output_dim)
        if bias is not None:
            output = output + bias
    return output
