"""Official package/import gate, with optional allocated CUDA BF16 execution."""
import argparse
import importlib.metadata
import json
import platform

EXPECTED = {
    'torch': '2.8.0+cu128', 'torchvision': '0.23.0+cu128',
    'transformers': '4.57.6', 'peft': '0.17.1', 'numpy': '2.2.6',
    'scipy': '1.15.3', 'soundfile': '0.13.1', 'accelerate': '1.10.1',
    'safetensors': '0.6.2', 'huggingface-hub': '0.34.4', 'librosa': '0.11.0',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    if versions != EXPECTED or platform.machine() != 'x86_64':
        raise RuntimeError('registered official baseline runtime differs')
    import torch
    import torchvision
    import peft
    import librosa
    import soundfile
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    report = {'versions': versions, 'architecture': platform.machine(),
              'cuda_build': torch.version.cuda, 'cuda_execution_checked': False,
              'scope': 'dependency/import check only; no model inference'}
    if torch.version.cuda != '12.8':
        raise RuntimeError('CUDA 12.8 wheel required')
    if args.require_cuda:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError('allocated CUDA device with native BF16 required')
        with torch.inference_mode():
            value = torch.ones((1, 1, 4, 16), dtype=torch.bfloat16, device='cuda:0')
            output = torch.nn.functional.scaled_dot_product_attention(value, value, value)
            if not torch.isfinite(output).all():
                raise RuntimeError('nonfinite CUDA BF16 SDPA output')
            torch.cuda.synchronize()
        report.update(cuda_execution_checked=True, device=torch.cuda.get_device_name(0),
                      capability=list(torch.cuda.get_device_capability(0)),
                      scope='dependency/import and small CUDA BF16 SDPA check; no model inference')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
