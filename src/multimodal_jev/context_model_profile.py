"""Derive Jev model geometry from an HF config without loading weights.

Qwen3-ASR and Qwen2.5-Omni use the same execution schema.  Their thinker text
configuration and LoRA module prefix are derived from the checkpoint instead of
being hardcoded in a runner.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .baseline import file_hash

SCHEMA = 'jev-context-model-profile-v1'
GEOMETRY = ('hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
            'head_dim', 'intermediate_size', 'vocab_size', 'audio_d_model',
            'audio_encoder_layers', 'audio_output_dim')
SUPPORTED_LAYER_TYPES = ('full_attention',)
HEAD_HEADS = 8
HEAD_RESIDUAL_SCALE = 0.1


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError('positive integer ' + name + ' required')
    return value


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ModelProfile:
    """Immutable declared geometry plus the config digest it was derived from."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    audio_d_model: int
    audio_encoder_layers: int
    audio_output_dim: int
    config_sha256: str
    model_type: str = 'qwen3_asr'
    target_prefix: str = 'language_model'

    def __post_init__(self):
        for name in GEOMETRY:
            _positive(getattr(self, name), name)
        if type(self.model_type) is not str or not self.model_type:
            raise ValueError('explicit model type required')
        if type(self.config_sha256) is not str or len(self.config_sha256) != 64:
            raise ValueError('config digest required')

    def geometry(self):
        return {name: getattr(self, name) for name in GEOMETRY}

    def lora_targets(self):
        """Every text layer's q/v projection, in the frozen lexicographic order."""
        return sorted(f'{self.target_prefix}.layers.{i}.self_attn.{p}_proj'
                      for i in range(self.num_hidden_layers) for p in ('q', 'v'))

    def decision_head(self):
        """Head dimensions for the shared sigmoid branch on this backbone."""
        if self.hidden_size % HEAD_HEADS:
            raise ValueError('hidden size must be divisible by the decision head count')
        return dict(hidden_size=self.hidden_size, num_heads=HEAD_HEADS,
                    ffn_size=self.hidden_size, residual_scale=HEAD_RESIDUAL_SCALE)

    def identity(self):
        """Bind geometry to the exact config and implementation that produced it."""
        payload = dict(schema=SCHEMA, model_type=self.model_type,
                       config_sha256=self.config_sha256,
                       implementation_sha256=file_hash(Path(__file__)),
                       **self.geometry())
        if self.model_type == 'qwen2_5_omni':
            payload['target_prefix'] = self.target_prefix
        payload['geometry_sha256'] = _canonical(self.geometry())
        return payload


def from_hf(path):
    """Read and validate one HF model config; symlinks and unknown layouts fail."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('ordinary HF model directory required')
    config_path = root/'config.json'
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError('ordinary config.json required')
    config = json.loads(config_path.read_text())
    if type(config) is not dict:
        raise ValueError('object config required')
    model_type = config.get('model_type')
    if model_type == 'qwen2_5_omni':
        thinker = config.get('thinker_config')
        text = thinker.get('text_config') if isinstance(thinker, dict) else None
        audio = thinker.get('audio_config') if isinstance(thinker, dict) else None
        target_prefix = 'thinker.model'
    else:
        text = config.get('text_config')
        audio = config.get('audio_config')
        target_prefix = 'language_model'
    if type(text) is not dict or type(audio) is not dict:
        raise ValueError('text and audio config sections required')
    if type(model_type) is not str or not model_type:
        raise ValueError('explicit model type required')
    layers = _positive(text.get('num_hidden_layers'), 'num_hidden_layers')
    layer_types = text.get('layer_types')
    if model_type == 'qwen2_5_omni' and layer_types is None and text.get('use_sliding_window') is False:
        layer_types = ['full_attention'] * layers
    # Uniform full-attention layers are what make one q/v naming scheme valid
    # for every layer; anything else needs a per-layer target derivation.
    if (type(layer_types) is not list or len(layer_types) != layers or
            any(t not in SUPPORTED_LAYER_TYPES for t in layer_types)):
        raise ValueError('uniform full-attention text layers required')
    heads = _positive(text.get('num_attention_heads'), 'num_attention_heads')
    kv_heads = _positive(text.get('num_key_value_heads'), 'num_key_value_heads')
    if heads % kv_heads:
        raise ValueError('key/value head count must divide attention head count')
    hidden_size = _positive(text.get('hidden_size'), 'hidden_size')
    if hidden_size % heads:
        raise ValueError('hidden size must be divisible by attention head count')
    head_dim = text.get('head_dim', hidden_size // heads if model_type == 'qwen2_5_omni' else None)
    return ModelProfile(
        hidden_size=hidden_size,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=_positive(head_dim, 'head_dim'),
        intermediate_size=_positive(text.get('intermediate_size'), 'intermediate_size'),
        vocab_size=_positive(text.get('vocab_size'), 'vocab_size'),
        audio_d_model=_positive(audio.get('d_model'), 'audio d_model'),
        audio_encoder_layers=_positive(audio.get('encoder_layers'), 'audio encoder_layers'),
        audio_output_dim=_positive(audio.get('output_dim'), 'audio output_dim'),
        config_sha256=file_hash(config_path),
        model_type=model_type, target_prefix=target_prefix)


def build_head(profile):
    """Construct the decision branch for this backbone (imports torch lazily)."""
    from .context_decision_head import ContextDecisionHead
    return ContextDecisionHead(**profile.decision_head())


def validate_against_config(profile, config):
    """Compare one experiment config's declared geometry with the derived profile.

    Frozen configs keep their literal target lists; this check is what keeps those
    literals honest instead of deleting them. Declared keys that this function does
    not recognise are rejected rather than silently ignored.
    """
    if type(config) is not dict:
        raise ValueError('object experiment config required')
    expected_targets = profile.lora_targets()
    compared = {}
    for stage in ('stage1', 'stage2'):
        declared = config.get(stage)
        if declared is None:
            continue
        if type(declared) is not dict or type(declared.get('lora')) is not dict:
            raise ValueError('explicit LoRA block required in ' + stage)
        targets = declared['lora'].get('target_modules')
        if type(targets) is not list or list(targets) != expected_targets:
            raise ValueError(stage + ' LoRA targets differ from the model profile')
        compared[stage + '.lora.target_modules'] = len(targets)
    declared_profile = config.get('model_profile')
    if declared_profile is not None:
        if type(declared_profile) is not dict:
            raise ValueError('object model profile required')
        unknown = set(declared_profile) - set(GEOMETRY)
        if unknown:
            raise ValueError('unknown declared model geometry: ' + ','.join(sorted(unknown)))
        for name, value in declared_profile.items():
            if getattr(profile, name) != value:
                raise ValueError('declared ' + name + ' differs from the model config')
        compared['model_profile'] = dict(declared_profile)
    return dict(schema=SCHEMA, status='consistent', compared=compared,
                geometry_sha256=_canonical(profile.geometry()))


def record(profile):
    """Serialisable evidence row for a report; no model call is implied."""
    return dict(asdict(profile), schema=SCHEMA,
                lora_targets=profile.lora_targets(),
                decision_head=profile.decision_head())
