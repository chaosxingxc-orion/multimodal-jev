"""Official Qwen3-ASR implementation of the typed audio Jev interface."""
from .model_provider import ABI
from .context_runtime_profile import PROFILE


class Qwen3ASR:
    runtime_profile = PROFILE

    @staticmethod
    def initialize_policy(config, output):
        from tools.reports.initialize_context_stage1_policy import initialize_qwen
        return initialize_qwen(config, output)

    @staticmethod
    def initialize_actor_runtime():
        from .span_runtime_profile import initialize_verl_preserving_qwen3, configure_actor_profile, SHORT_AUDIO_PROFILE
        initialize_verl_preserving_qwen3()
        configure_actor_profile(SHORT_AUDIO_PROFILE)

    @staticmethod
    def adapter_transition(config, fresh):
        from .context_stage_lora import admit_stage2, from_policy
        if fresh:
            return {'stage2': from_policy(config['source_contract'])}
        return admit_stage2(config['stage_lora_transition'], config['source_contract'])

    @staticmethod
    def load_actor(*args):
        from tools.reports.stage2_view_actor import load_actor_stack
        return load_actor_stack(*args)

    @staticmethod
    def load_reference(*args):
        from tools.reports.context_transfer_macro_parity import load_reference_model
        return load_reference_model(*args)

    @staticmethod
    def load_vllm(config, bundle):
        from .context_resident_rollout import FixedContextRollout
        return FixedContextRollout(config)

    @staticmethod
    def processor(path):
        from transformers import Qwen3ASRProcessor
        return Qwen3ASRProcessor.from_pretrained(path, local_files_only=True)

    @staticmethod
    def collation(base, processor, model):
        audio = base.config.audio_config
        return dict(decision_token_id=model.decision_token_id, pad_token_id=processor.tokenizer.pad_token_id,
                    audio_token_id=base.config.audio_token_id, audio_chunk_size=audio.n_window*2,
                    audio_infer_window=audio.n_window_infer)

    @staticmethod
    def choice_compiler(*args):
        from .context_choice_inputs import ChoiceInputs
        return ChoiceInputs(*args)

    @staticmethod
    def encode(*args, **kwargs):
        from .context_decision_contract import encode_context_request
        return encode_context_request(*args, **kwargs)

    @staticmethod
    def forward(*args, **kwargs):
        from .context_audio_collate import forward_context_audio_batch
        return forward_context_audio_batch(*args, **kwargs)


def create(options):
    if options:
        raise ValueError('unexpected Qwen3-ASR provider options')
    return Qwen3ASR()
