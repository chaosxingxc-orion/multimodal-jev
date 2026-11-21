"""Policy-free context audio cache; labels/behavior bind after preparation.

Uses the same local immutable cache and ordered whole-batch prefetch primitive
as the established span pipeline. Cache keys bind this compiler's marker,
dtype and context budget in addition to processor/input/preprocessing identity.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import threading

import torch

from .audio_policy import load_audio
from .baseline import file_hash
from .context_decision_contract import encode_context_request
from .encoded_audio_cache import binding,canonical
from .slurp_training import content_hash


def processor_cache_identity(policy):
    """Match train/warmup cache keys using processor assets, never policy weights."""
    root=Path(policy)
    manifest=json.loads((root/'MANIFEST.json').read_text())
    assets={k:v for k,v in manifest['files'].items() if k.startswith('processor/')}
    if not {'processor/tokenizer.json','processor/tokenizer_config.json'}<=assets.keys():
        raise ValueError('complete pinned processor assets required')
    if any(file_hash(root/name)!=digest for name,digest in assets.items()):
        raise ValueError('processor assets changed')
    return content_hash(assets)


@dataclass(frozen=True)
class PreparedContextInput:
    request: dict
    inputs: dict
    positions: torch.Tensor
    audio_receipt: dict
    cache_hit: bool


class CachedContextPreparer:
    def __init__(self,processor_factory,cache,*,processor_sha256,decision_token_id,
                 audio_token_id,max_tokens,preprocessing=None,audio_mirror=None,prompt_layout=None,
                 provider_namespace=None,encode=None):
        if (any(type(v) is not int or v<0 for v in (decision_token_id,audio_token_id)) or
                decision_token_id==audio_token_id or type(max_tokens) is not int or max_tokens<1):
            raise ValueError('explicit distinct marker/audio IDs and context budget required')
        self.processor_factory=processor_factory;self.cache=cache
        self.processor_sha256=processor_sha256;self.preprocessing=preprocessing
        self.audio_mirror=audio_mirror
        self.encoding=dict(decision_token_id=decision_token_id,audio_token_id=audio_token_id,max_tokens=max_tokens)
        if prompt_layout is not None:
            from .context_decision_contract import PREFIX_LAYOUT
            if prompt_layout!=PREFIX_LAYOUT:raise ValueError("unknown prompt layout")
            self.encoding["prompt_layout"]=prompt_layout
        self.encode=encode_context_request if encode is None else encode
        self.provider_namespace=provider_namespace
        self.local=threading.local()

    def identity(self,request,expected_audio_sha256):
        identity=binding(request=request,audio_sha256=expected_audio_sha256,
                         processor_sha256=self.processor_sha256,preprocessing=self.preprocessing)
        identity['context_encoding']=dict(self.encoding,features_dtype='bfloat16',version='context-cache-v1')
        if self.provider_namespace is not None:
            identity['model_provider_sha256']=self.provider_namespace
        return identity

    def __call__(self,job):
        request,expected_audio_sha256=job
        audio_path=(request.audio if self.audio_mirror is None else
                    self.audio_mirror.resolve(expected_audio_sha256))
        if self.audio_mirror is None and file_hash(audio_path)!=expected_audio_sha256:
            raise ValueError('source audio changed')
        identity=self.identity(request,expected_audio_sha256)
        value=self.cache.read(identity);hit=value is not None
        if value is None:
            if not hasattr(self.local,'processor'):self.local.processor=self.processor_factory()
            audio,receipt=load_audio(audio_path,self.preprocessing,return_receipt=True)
            inputs,positions=self.encode(self.local.processor,request,audio,**self.encoding)
            inputs={k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in inputs.items()}
            self.cache.write(identity,inputs,positions,receipt)
        else:inputs,positions,receipt=value
        ids=inputs['input_ids']
        actual=(ids[0]==self.encoding['decision_token_id']).nonzero().flatten()
        if (ids.shape[-1]>self.encoding['max_tokens'] or positions.shape!=(1,len(request.questions)) or
                not torch.equal(positions[0],actual) or inputs['input_features'].dtype!=torch.bfloat16):
            raise ValueError('cached context representation differs')
        return PreparedContextInput(canonical(request.model_input()),inputs,positions,receipt,hit)
