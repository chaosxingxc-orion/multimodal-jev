"""Audio-only Qwen2.5-Omni Thinker with the shared Jev decision head."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .context_choice_inputs import ChoiceInputs
from .context_decision_contract import DECISION_TOKEN, encode_context_request
from .context_decision_model import ContextDecisionModel
from .context_model_profile import build_head, from_hf
from .model_provider import ABI as ABI

PROFILE = 'provider:qwen2-5-omni-thinker-audio-v1'
REQUEST_VERSION = 'jev-context-exclusive_event-v1'


def register_decision_token(tokenizer, vocab_size):
    previous = set(tokenizer.all_special_tokens)
    if tokenizer.add_special_tokens({'extra_special_tokens': [DECISION_TOKEN]},
                                    replace_extra_special_tokens=False) != 1:
        raise ValueError('expected a fresh Jev marker')
    marker_id, = tokenizer.encode(DECISION_TOKEN, add_special_tokens=False)
    if not 0 <= marker_id < vocab_size or set(tokenizer.all_special_tokens) != previous | {DECISION_TOKEN}:
        raise ValueError('Omni marker registration changed the reserved tokens or exceeded the embedding')
    return marker_id


class ThinkerAudio(nn.Module):
    def __init__(self, thinker):
        super().__init__()
        self.thinker = thinker
        self.thinker.visual = None
        self.thinker.lm_head = None
        self.config = thinker.config

    def get_input_embeddings(self):
        return self.thinker.get_input_embeddings()

    def forward(self, *, input_ids, attention_mask, input_features, input_features_mask,
                use_cache, return_dict):
        if use_cache or not return_dict:
            raise ValueError('expected one uncached audio decision request')
        thinker = self.thinker
        with torch.no_grad():
            acoustic = thinker.get_audio_features(
                input_features, feature_attention_mask=input_features_mask, return_dict=True
            ).last_hidden_state
            text = self.get_input_embeddings()(input_ids)
            audio_tokens = input_ids.eq(self.config.audio_token_id)
            text = text.masked_scatter(audio_tokens.unsqueeze(-1), acoustic.to(text.dtype))
            # Audio without images/video uses the upstream Thinker position path.
            positions, _ = thinker.get_rope_index(input_ids, attention_mask=attention_mask)
        return thinker.model(inputs_embeds=text, attention_mask=attention_mask,
                             position_ids=positions, use_cache=False, return_dict=True)


class AudioProcessor:
    """Keep native audio expansion and serialize only the valid mel frames."""
    def __init__(self, native):
        self.native = native
        self.tokenizer = native.tokenizer

    def apply_chat_template(self, *args, **kwargs):
        return self.native.apply_chat_template(*args, **kwargs)

    def __call__(self, *, text, audio, sampling_rate, return_tensors):
        waveform, = audio
        frontend = self.native.feature_extractor
        if not 0 < len(waveform) <= frontend.n_samples:
            raise ValueError('audio exceeds the Omni processor limit or is empty')
        # Keep a full FFT window of tail padding for the last valid frame.
        hop = frontend.hop_length
        limit = min(frontend.n_samples, ((len(waveform) + frontend.n_fft + hop - 1) // hop) * hop)
        batch = dict(self.native(text=text, audio=audio, return_tensors=return_tensors,
                                 audio_kwargs=dict(sampling_rate=sampling_rate, max_length=limit)))
        mask = batch.pop('feature_attention_mask')
        frames = int(mask.sum())
        batch['input_features'] = batch['input_features'][..., :frames].contiguous()
        batch['input_features_mask'] = mask[..., :frames].contiguous()
        return batch


def _load_backbone(path):
    from transformers import Qwen2_5OmniConfig, Qwen2_5OmniThinkerForConditionalGeneration

    config = Qwen2_5OmniConfig.from_pretrained(path, local_files_only=True).thinker_config
    thinker, report = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        path, config=config, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='eager', output_loading_info=True)
    if (any(value for key, value in report.items() if key != 'unexpected_keys')
            or any(not key.startswith(('talker.', 'token2wav.')) for key in report['unexpected_keys'])):
        raise ValueError('unmatched Omni Thinker weights')
    return ThinkerAudio(thinker).requires_grad_(False).eval()


def _model(path, profile, marker_id, rank, alpha, device):
    from peft import LoraConfig, get_peft_model

    from .bf16_lora import cast_lora_bf16

    base = _load_backbone(path)
    targets = profile.lora_targets()
    if not set(targets) <= dict(base.named_modules()).keys():
        raise ValueError('missing Omni Thinker LoRA projections')
    adapted = get_peft_model(
        base, LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets, bias='none'),
        autocast_adapter_dtype=False)
    cast_lora_bf16(adapted)
    head = build_head(profile)
    return ContextDecisionModel(adapted.get_base_model(), head, marker_id).to(
        device=device, dtype=torch.bfloat16).eval()


class Qwen25Omni:
    runtime_profile = PROFILE
    encode = staticmethod(encode_context_request)
    choice_compiler = staticmethod(ChoiceInputs)

    @staticmethod
    def processor(path):
        from transformers import Qwen2_5OmniProcessor

        return AudioProcessor(Qwen2_5OmniProcessor.from_pretrained(path, local_files_only=True))

    def initialize_policy(self, config, output):
        from .cached_context_preparation import processor_cache_identity
        from .context_policy_artifacts import read_execution_bundle
        from .context_policy_snapshot import export_policy, lora_contract
        from .policy_identity import checkpoint_identity

        torch.set_num_threads(config['cpu_workers'])
        base_id, _ = checkpoint_identity(config['model'], io_workers=config['io_workers'])
        if base_id != config['base_id']:
            raise ValueError('Omni base identity differs')
        profile = from_hf(config['model'])
        processor = self.processor(config['model']).native
        tokenizer = processor.tokenizer
        symbols = [tokenizer.encode(value, add_special_tokens=False) for value in config['marker_init_symbols']]
        if len(symbols) != 2 or any(len(value) != 1 for value in symbols):
            raise ValueError('two single-token marker initialization symbols required')
        marker_id = register_decision_token(tokenizer, profile.vocab_size)
        torch.manual_seed(config['seed'])
        lora = config['lora']
        model = _model(config['model'], profile, marker_id, lora['rank'], lora['alpha'], 'cpu')
        embedding = model.backbone.get_input_embeddings().weight
        with torch.no_grad():
            embedding[marker_id].copy_(embedding[[value[0] for value in symbols]].float().mean(0))
        processor_path = output / 'processor-init'
        processor.save_pretrained(processor_path)
        restored = self.processor(processor_path)
        if (restored.tokenizer.encode(DECISION_TOKEN, add_special_tokens=False) != [marker_id]
                or set(restored.tokenizer.all_special_tokens) != set(tokenizer.all_special_tokens)):
            raise ValueError('Omni processor lost reserved tokens')
        branch = dict(profile.decision_head(), layers=1, dropout=0.,
                      position_encoding='inherited_backbone_states_no_extra_branch_encoding',
                      readout='decision_marker_itself',
                      attention_scope='same_request_valid_causal_prefix_including_marker')
        contract = dict(base_id=base_id, runtime_profile=PROFILE, request_version=REQUEST_VERSION,
                        branch=branch, lora=lora_contract(model.backbone),
                        token=dict(id=marker_id, text=DECISION_TOKEN, row_trainable=False,
                                   registration_status='admitted'),
                        output=dict(field='prob', activation='sigmoid', normalize_across_candidates=False))
        policy_path = output / 'policy'
        policy_id = export_policy(policy_path, model, processor_path, contract)
        read_execution_bundle(policy_path, expected_policy_id=policy_id,
                              expected_contract=contract, expected_base_id=base_id)
        return dict(status='stage1_fresh_cpu_policy_verified', base_id=base_id, policy_id=policy_id,
                    geometry=profile.identity(), processor_sha256=processor_cache_identity(policy_path),
                    audio_token_id=model.backbone.config.audio_token_id, npu_executed=False,
                    formal_training_admitted=False)

    @staticmethod
    def initialize_actor_runtime():
        import torch_npu
        from verl.trainer.ppo import core_algos  # noqa: F401

        torch.npu.config.allow_internal_format = False
        if torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT') != b'disable':
            raise ValueError('actor internal-format configuration failed')

    @staticmethod
    def adapter_transition(config, fresh):
        if not fresh:
            raise ValueError('Omni provider requires a single-stage policy')
        return {'stage2': Qwen25Omni.lora_spec(config['source_contract'])}

    @staticmethod
    def lora_spec(contract):
        values = {(spec['rank'], spec['alpha']) for spec in contract['lora'].values()}
        if len(values) != 1:
            raise ValueError('uniform Omni LoRA rank and alpha required')
        rank, alpha = values.pop()
        return dict(rank=rank, alpha=alpha)

    def load_snapshot(self, config, device):
        from .context_policy_snapshot import load_policy

        profile = from_hf(config['model'])
        contract = config['contract']
        if (contract['runtime_profile'] != PROFILE or contract['request_version'] != REQUEST_VERSION
                or set(contract['lora']) != set(profile.lora_targets())
                or any(contract['branch'][k] != value for k, value in profile.decision_head().items())):
            raise ValueError('Omni policy geometry or runtime differs')
        model = _model(config['model'], profile, contract['token']['id'], **self.lora_spec(contract),
                       device=device)
        load_policy(config['snapshot'], model, expected_policy_id=config['policy_id'],
                    expected_contract=contract, expected_base_id=config['base_id'])
        return model

    def load_actor(self, config, device, output, execution, transition):
        from .baseline import file_hash
        from .local_checkpoint import commit_json
        from .policy_identity import checkpoint_identity
        from tools.reports.stage2_view_actor import wrap_actor

        if checkpoint_identity(config['model'])[0] != config['base_id']:
            raise ValueError('Omni actor base changed')
        app = dict(config, snapshot=config['source_policy'], policy_id=config['source_policy_id'],
                   contract=config['source_contract'])
        if transition['stage2'] != self.lora_spec(app['contract']):
            raise ValueError('Omni actor adapter differs')
        model = self.load_snapshot(app, device)
        processor = self.processor(Path(app['snapshot']) / 'processor')
        commit_json(output / 'runtime.json', dict(profile=PROFILE, source_sha256=file_hash(__file__),
                                                  execution_dtype='bfloat16', engine='HF eager'))
        wrapped, optimizer, dtypes = wrap_actor(model, device, execution)
        return model.backbone, processor, model, wrapped, optimizer, dtypes, None

    def load_reference(self, config, bundle, output):
        import torch_npu  # noqa: F401

        from .local_checkpoint import commit_json

        if config.get('process_device_scope') is not None:
            from tools.jev.context_configurable_device_scope import validate_tp1
            validate_tp1(config)
        elif (len(config['physical_devices']) != 1 or
              {p.name for p in Path('/dev').glob('davinci[0-9]*')} !=
              {f"davinci{config['physical_devices'][0]}"}):
            raise ValueError('one isolated Omni NPU is required')
        if bundle['contract'] != config['contract'] or bundle['policy_id'] != config['policy_id']:
            raise ValueError('Omni reference policy differs')
        torch.npu.config.allow_internal_format = False
        torch.npu.set_device(0)
        device = torch.device('npu', 0)
        model = self.load_snapshot(config, device)
        commit_json(output / 'hf-runtime.json', dict(profile=PROFILE, execution_dtype='bfloat16', engine='HF eager'))
        return model, device, model.backbone, None

    @staticmethod
    def collation(base, processor, model):
        return dict(decision_token_id=model.decision_token_id, pad_token_id=processor.tokenizer.pad_token_id,
                    audio_token_id=base.config.audio_token_id)

    @staticmethod
    def forward(model, items, *, device, dtype, policy_id, **collation):
        inputs, positions = collate_audio(items, policy_id=policy_id, **collation)
        inputs = {key: value.to(device, dtype=dtype if value.is_floating_point() else value.dtype)
                  for key, value in inputs.items()}
        return model(inputs, positions.to(device), return_logits=True)


def collate_audio(items, *, policy_id, decision_token_id, pad_token_id, audio_token_id):
    required = {'input_ids', 'attention_mask', 'input_features', 'input_features_mask'}
    if not items or len({item['request_id'] for item in items}) != len(items):
        raise ValueError('distinct Omni audio requests required')
    for item in items:
        inputs, positions = item['inputs'], item['positions']
        if item['policy_id'] != policy_id or inputs.keys() != required:
            raise ValueError('complete input-only Omni request required')
        ids, mask = inputs['input_ids'], inputs['attention_mask']
        features, frames = inputs['input_features'], inputs['input_features_mask']
        if (ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.long
                or mask.shape != ids.shape or not mask.eq(1).all()
                or features.ndim != 3 or features.shape[0] != 1 or not features.is_floating_point()
                or frames.shape != (1, features.shape[-1]) or not frames.eq(1).all()):
            raise ValueError('one unpadded token sequence and contiguous valid audio frames required')
        markers = ids[0].eq(decision_token_id).nonzero().flatten()
        audio_indices = ids[0].eq(audio_token_id).nonzero().flatten()
        if (positions.shape != items[0]['positions'].shape or positions.dtype != torch.long
                or not markers.numel() or not torch.equal(positions, markers[None, :])
                or not audio_indices.numel() or audio_indices[-1] >= markers[0]
                or audio_indices.numel() != ((features.shape[-1] + 1) // 2) // 2):
            raise ValueError('Omni audio placeholders or decision positions differ')
    inputs = {}
    for key in sorted(required):
        values = [item['inputs'][key] for item in items]
        length = max(value.shape[-1] for value in values)
        fill = pad_token_id if key == 'input_ids' else 0
        inputs[key] = torch.cat([F.pad(value, (0, length - value.shape[-1]), value=fill) for value in values])
    return inputs, torch.cat([item['positions'] for item in items])


def create(options):
    if options:
        raise ValueError('unexpected Qwen2.5-Omni provider options')
    return Qwen25Omni()
