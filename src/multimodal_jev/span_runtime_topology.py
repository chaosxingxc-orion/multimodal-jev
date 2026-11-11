"""Fail-closed selection of accepted TP1 and experimental TP4 runtimes."""
from .span_runtime_profile import PROFILE

TP4_DIAGNOSTIC = 'qwen3-asr-bf16-native-tp4-diagnostic-v1'
TP4_HF_ORDER = 'qwen3-asr-bf16-hf-order-tp4-diagnostic-v1'
TP4_COMPLETE_K = 'qwen3-asr-bf16-complete-k-tp4-diagnostic-v1'
TP4_FULL_WIDTH = 'qwen3-asr-bf16-full-width-tp4-diagnostic-v1'


def runtime_topology(config):
    size = config.get('tensor_parallel_size', 1)
    profile = config.get('runtime_profile')
    if type(size) is not int:
        raise ValueError('integer tensor parallel size required')
    devices=config.get('physical_devices',[4] if size==1 else [4,5,6,7])
    if (type(devices) is not list or any(type(d) is not int or d not in (4,5,6,7) for d in devices)
            or len(devices)!=size or len(set(devices))!=size):
        raise ValueError('distinct authorized physical devices required')
    if size == 1 and profile == PROFILE:
        return dict(size=1, devices=[f'davinci{devices[0]}'], architecture='Qwen3ASRForSpanDecisions',
                    diagnostic_only=False)
    if size == 4 and profile in (TP4_DIAGNOSTIC, TP4_HF_ORDER, TP4_COMPLETE_K, TP4_FULL_WIDTH):
        if devices!=[4,5,6,7]:raise ValueError('fixed TP4 rank order required')
        return dict(size=4, devices=['davinci4', 'davinci5', 'davinci6', 'davinci7'],
                    architecture='Qwen3ASRForSpanDecisionsTP4Diagnostic', diagnostic_only=True)
    raise ValueError('unsupported Span topology/profile pair')


def install_diagnostic_profile(engine, name):
    if name not in (TP4_DIAGNOSTIC, TP4_HF_ORDER, TP4_COMPLETE_K, TP4_FULL_WIDTH):
        raise ValueError('unsupported diagnostic profile')
    receipts = engine.llm_engine.collective_rpc('align_jev_internal_format')
    if len(receipts) != 4 or any(r.get('after') != 'disable' for r in receipts):
        raise ValueError('four internal-format receipts required')
    operations = {'align_jev_internal_format': receipts}
    if name in (TP4_HF_ORDER, TP4_COMPLETE_K, TP4_FULL_WIDTH):
        aligned = engine.llm_engine.collective_rpc('align_jev_tp4_hf_order')
        if len(aligned)!=4 or any(r.get('tensor_parallel_size')!=4 or
                                 r.get('diagnostic_only') is not True for r in aligned):
            raise ValueError('four TP4 alignment receipts required')
        operations['align_jev_tp4_hf_order'] = aligned
    if name in (TP4_COMPLETE_K, TP4_FULL_WIDTH):
        gathered=engine.llm_engine.collective_rpc('align_jev_tp4_complete_k')
        if (len(gathered)!=4 or sorted(r.get('rank',-1) for r in gathered)!=list(range(4))
                or any(r.get('tensor_parallel_size')!=4 or len(r.get('modules',[]))!=56 for r in gathered)):
            raise ValueError('four complete-K projection receipts required')
        operations['align_jev_tp4_complete_k']=gathered
    if name == TP4_FULL_WIDTH:
        padded=engine.llm_engine.collective_rpc('align_jev_tp4_full_width')
        if (len(padded)!=4 or sorted(r.get('rank',-1) for r in padded)!=list(range(4))
                or any(r.get('layers')!=28 or r.get('diagnostic_only') is not True for r in padded)):
            raise ValueError('four full-width projection receipts required')
        operations['align_jev_tp4_full_width']=padded
    return dict(profile=name, operations=operations, diagnostic_only=True,
                scope='TP4 diagnostic with native LoRA; projection/communication path in operation receipts; no training admission')
