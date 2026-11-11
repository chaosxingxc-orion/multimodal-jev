#!/usr/bin/env python3
"""Installed PEFT/safetensors BF16 optimizer resume test; synthetic CPU model."""
import argparse
import json
from pathlib import Path
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from multimodal_jev.bf16_lora import BF16MasterAdamW, cast_lora_bf16
from multimodal_jev.span_checkpoint import save_span_checkpoint, restore_span_optimizer


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(8, 4, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.projection(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    base = TinyModel()
    base_state = {k:v.clone() for k,v in base.state_dict().items()}
    adapted = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.,
                            target_modules=['projection']), autocast_adapter_dtype=False).eval()
    cast_lora_bf16(adapted)
    params = [p for p in adapted.parameters() if p.requires_grad]
    optimizer = BF16MasterAdamW(params, lr=1e-4)
    x = torch.randn(3, 8, dtype=torch.bfloat16)

    def step(model, opt):
        opt.zero_grad()
        model(x).float().square().mean().backward()
        opt.step()

    for _ in range(7):
        step(adapted, optimizer)
    save_span_checkpoint(args.output, adapted, adapted, optimizer, 'synthetic-base-v1', 7)
    base2 = TinyModel()
    base2.load_state_dict(base_state)
    restored = PeftModel.from_pretrained(base2, args.output/'adapter', is_trainable=True,
                                         autocast_adapter_dtype=False).eval()
    cast_lora_bf16(restored)
    optimizer2 = BF16MasterAdamW([p for p in restored.parameters() if p.requires_grad], lr=.1)
    restore_span_optimizer(args.output, restored, optimizer2, 'synthetic-base-v1')
    assert torch.equal(adapted(x), restored(x))
    for _ in range(9):
        step(adapted, optimizer)
        step(restored, optimizer2)
        assert torch.equal(adapted(x), restored(x))
        assert all(torch.equal(a,b) for a,b in zip(optimizer.masters, optimizer2.masters))
    print(json.dumps(dict(status='passed', saved_updates=7, resumed_updates=9,
        exact_execution_outputs=True, exact_fp32_masters=True,
        scope='Real PEFT BF16 safetensors and Adam resume on a synthetic CPU linear model; no speech/NPU/RL claim.')))
