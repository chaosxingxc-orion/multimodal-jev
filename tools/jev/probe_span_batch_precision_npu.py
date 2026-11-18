#!/usr/bin/env python3
"""Process-isolated NPU precision options; retain BF16 model/LoRA tensors.

Diagnostic only: these options do not imply FP32 accumulation for all BF16
kernels. Their effect must be established from measurements and profiler/docs.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--variant',choices=['keep_dtype','fp32_add'])
    args=p.parse_args();root=Path(args.output)
    if args.variant:
        import torch
        import torch_npu
        from torch_npu.npu.npu_config import _CubeMathType
        from multimodal_jev.baseline import atomic_json
        from multimodal_jev.span_device_scope import validate_tp1
        validate_tp1(json.loads(Path(args.config).read_text()))
        torch.npu.matmul.allow_hf32=False
        torch.npu.conv.allow_hf32=False
        value=_CubeMathType.KEEP_DTYPE if args.variant=='keep_dtype' else _CubeMathType.USE_FP32_ADD
        torch.npu.matmul.cube_math_type=value
        torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'})
        keys=['CUBE_MATH_TYPE','ALLOW_MATMUL_HF32','ALLOW_CONV_HF32','ACL_PRECISION_MODE']
        receipt={k:str(torch_npu._C._npu_getOption(k)) for k in keys}
        import diagnose_span_audio_batch_npu
        sys.argv=[sys.argv[0],'--config',args.config,'--output',args.output]
        diagnose_span_audio_batch_npu.main()
        atomic_json(root/'precision-options.json',dict(variant=args.variant,options=receipt,
            model_execution_dtype='bfloat16',scope='Configured options only, not proof of kernel accumulation precision'))
        return
    root.mkdir(parents=True,exist_ok=False)
    results=[]
    for variant in ['keep_dtype','fp32_add']:
        command=[sys.executable,__file__,'--config',args.config,'--output',str(root/variant),'--variant',variant]
        with (root/(variant+'.log')).open('w') as f:
            code=subprocess.call(command,stdout=f,stderr=subprocess.STDOUT)
        results.append(dict(variant=variant,exit_code=code))
        (root/'progress.json').write_text(json.dumps(dict(results=results))+'\n')
    (root/'result.json').write_text(json.dumps(dict(completed=True,results=results,
        training_promotion=False,scope='Isolated precision-option experiments, failures retained; inspect each raw result.'),indent=2)+'\n')


if __name__=='__main__':main()
