"""Fail-closed selection of accepted TP1 and experimental TP4 runtimes."""
from .span_runtime_profile import PROFILE

TP4_DIAGNOSTIC = 'qwen3-asr-bf16-native-tp4-diagnostic-v1'


def runtime_topology(config):
    size = config.get('tensor_parallel_size', 1)
    profile = config.get('runtime_profile')
    if type(size) is not int:
        raise ValueError('integer tensor parallel size required')
    if size == 1 and profile == PROFILE:
        return dict(size=1, devices=['davinci4'], architecture='Qwen3ASRForSpanDecisions',
                    diagnostic_only=False)
    if size == 4 and profile == TP4_DIAGNOSTIC:
        return dict(size=4, devices=['davinci4', 'davinci5', 'davinci6', 'davinci7'],
                    architecture='Qwen3ASRForSpanDecisionsTP4Diagnostic', diagnostic_only=True)
    raise ValueError('unsupported Span topology/profile pair')


def install_diagnostic_profile(engine, name):
    if name != TP4_DIAGNOSTIC:
        raise ValueError('unsupported diagnostic profile')
    receipts = engine.llm_engine.collective_rpc('align_jev_internal_format')
    if len(receipts) != 4 or any(r.get('after') != 'disable' for r in receipts):
        raise ValueError('four internal-format receipts required')
    return dict(profile=name, operations={'align_jev_internal_format': receipts},
                diagnostic_only=True, scope='Native TP4 kernels and LoRA; no training admission')
