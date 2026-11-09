#!/usr/bin/env python3
"""Validate installed dependencies and public runtime imports."""
import importlib
import importlib.metadata as md
import json
import platform
from pathlib import Path

m=json.loads(Path('/opt/multimodal-jev/IMAGE-MANIFEST.json').read_text())
packages={name: md.version(name) for name in m['required_packages']}
assert packages==m['required_packages'], (packages,m['required_packages'])
for module, symbol in [('transformers','Qwen3ASRForConditionalGeneration'),
                       ('transformers','Qwen3ASRProcessor'),('verl','DataProto')]:
    getattr(importlib.import_module(module),symbol)
from vllm.model_executor.models.registry import ModelRegistry
assert 'Qwen3ASRForConditionalGeneration' in ModelRegistry.get_supported_archs()
from multimodal_jev import Candidate, DecisionSchema, JevExample, score_rollout
example=JevExample('smoke','train','fixture.wav',DecisionSchema('s','choice','Pick',
                    (Candidate('a','activate'),Candidate('b','deactivate'))),'a')
assert score_rollout(example,'a')['total']>score_rollout(example,None)['total']
print(json.dumps({'status':'passed','python':platform.python_version(),'packages':packages,
                  'scope':'imports, registry and CPU contract; no NPU or model inference'},indent=2))
