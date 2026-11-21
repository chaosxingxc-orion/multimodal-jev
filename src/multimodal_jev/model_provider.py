"""Select the same Jev model implementation for every role of a frozen run."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re

from .baseline import file_hash

ABI = 'jev-audio-model-v1'
DEFAULT = 'multimodal_jev.qwen3_asr_provider'
METHODS = ('initialize_policy', 'initialize_actor_runtime', 'adapter_transition', 'load_actor', 'load_reference',
           'processor', 'collation', 'choice_compiler', 'encode', 'forward')


def binding(config):
    if 'model_provider' not in config:
        return None  # Frozen Qwen3-ASR configurations predate explicit bindings.
    spec = config['model_provider']
    if (not isinstance(spec, dict) or set(spec) != {'abi', 'module', 'sha256', 'options'}
            or spec['abi'] != ABI or not isinstance(spec['options'], dict)
            or not isinstance(spec['module'], str)
            or not all(part.isidentifier() for part in spec['module'].split('.'))
            or not isinstance(spec['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', spec['sha256'])):
        raise ValueError('invalid model_provider binding')
    return spec


def cache_namespace(config):
    spec = binding(config)
    if spec is None:
        return None
    raw = json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def require_same_provider(*configs):
    first, *others = (binding(c) for c in configs)
    if any(other != first for other in others):
        raise ValueError('model_provider differs between bound artifacts')


def resolve_provider(config):
    spec = binding(config)
    name = spec['module'] if spec else DEFAULT
    if spec:
        location = importlib.util.find_spec(name)
        if (not location or not location.origin or Path(location.origin).suffix != '.py'
                or file_hash(location.origin) != spec['sha256']):
            raise ValueError(f'model_provider source differs: {name}')
    module = importlib.import_module(name)
    if getattr(module, 'ABI', None) != ABI:
        raise ValueError(f'model_provider ABI differs: {name}')
    provider = module.create(spec['options'] if spec else {})
    if any(not callable(getattr(provider, method, None)) for method in METHODS):
        raise ValueError(f'incomplete model_provider: {name}')
    profile = getattr(provider, 'runtime_profile', None)
    if not isinstance(profile, str) or not profile:
        raise ValueError(f'missing model_provider runtime profile: {name}')
    for key in ('source_contract', 'contract'):
        if key in config and profile != config[key]['runtime_profile']:
            raise ValueError(f'model_provider incompatible with policy runtime profile: {name}')
    return provider
