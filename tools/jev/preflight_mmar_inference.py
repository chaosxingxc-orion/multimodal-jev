#!/usr/bin/env python3
"""Exercise the inference input helper and actual vLLM configuration on CPU."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.mmar_inference import ENGINE, population, capacity_indices, prepare_checked, ids_hash
from multimodal_jev.audio2tool_engine import VERSIONS
# Import the real entrypoint too, without loading weights or dispatching an engine.
import infer_mmar_omni


def run(args):
    start = time.monotonic()
    versions = {k:importlib.metadata.version(k) for k in VERSIONS}
    if versions != VERSIONS:
        raise ValueError('fixed inference image versions required')
    rows, prior, identity = population(args.inputs, args.processor_run)
    for name,pin in identity['processor_assets'].items():
        if file_hash(args.model/name) != pin:
            raise ValueError('accepted processor assets required')
    import torch
    from transformers import Qwen3OmniMoeProcessor
    from vllm import SamplingParams
    from vllm.config import ModelConfig
    torch.set_num_threads(2)
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model,local_files_only=True)
    config = ModelConfig(model=str(args.model),tokenizer=str(args.model),dtype=ENGINE['dtype'],
        max_model_len=ENGINE['max_model_len'],enforce_eager=True,limit_mm_per_prompt=ENGINE['limit_mm_per_prompt'])
    if config.max_model_len != 4096:
        raise ValueError('actual context configuration differs')
    settings = []
    for budget in (1,256):
        params = SamplingParams(temperature=0.,max_tokens=budget,n=1,seed=17)
        values = {k:getattr(params,k) for k in ('temperature','max_tokens','n','seed')}
        if values != dict(temperature=0.,max_tokens=budget,n=1,seed=17):
            raise ValueError('actual sampling parameters differ')
        settings.append(values)
    records = []
    for i in capacity_indices(prior):
        request,receipt,ids = prepare_checked(processor,rows[i],args.audio_root,prior[i])
        records.append(dict(id=rows[i]['id'],audio=receipt,input_tokens=len(ids),input_ids_sha256=ids_hash(ids)))
    atomic_json(args.output/'result.json',dict(status='passed',population=len(rows),boundaries=records,
        versions=versions,engine=ENGINE,sampling=settings,elapsed_seconds=time.monotonic()-start,
        runner_sha256=file_hash(infer_mmar_omni.__file__),model_loaded=False,targets_read=False,
        scope='Actual fixed-image imports, ModelConfig/SamplingParams, full population admission and shared boundary waveform/processor input. No LLM engine, NPU generation or weight update.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('inputs','processor-run','audio-root','model','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    try:run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(args.output/'terminal.json',dict(exit_code=0))
