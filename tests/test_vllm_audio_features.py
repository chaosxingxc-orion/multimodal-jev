import sys
from types import ModuleType,SimpleNamespace
import pytest
from multimodal_jev.vllm_audio_features import enable_feature_passthrough


def test_feature_passthrough_preserves_waveform_path_and_is_idempotent(monkeypatch):
    calls=[]
    class Processor:
        def _get_hf_mm_data(self,items):return items.data
        def _apply_hf_processor_main(self,items,kwargs):
            calls.append((items,kwargs));return 'original'
    vllm=ModuleType('vllm.model_executor.models.qwen3_asr')
    vllm.Qwen3ASRMultiModalProcessor=Processor
    hf=ModuleType('transformers.feature_extraction_utils');hf.BatchFeature=dict
    monkeypatch.setitem(sys.modules,vllm.__name__,vllm)
    monkeypatch.setitem(sys.modules,hf.__name__,hf)
    class Items:
        def __init__(self,data):self.data=data
        def get_all_counts(self):return {'audio':1}
        def select(self,keys):assert keys=={'audio'};return self
    features={'input_audio_features':object(),'audio_feature_lengths':object()}
    enable_feature_passthrough();first=Processor._apply_hf_processor_main
    enable_feature_passthrough();assert Processor._apply_hf_processor_main is first
    processor=Processor()
    output=processor._apply_hf_processor_main(Items(({},features)),{})
    assert output==features and not calls
    wave=Items(({'audios':['wave']},{}));kwargs={'x':1}
    assert processor._apply_hf_processor_main(wave,kwargs)=='original'
    assert calls==[(wave,kwargs)]
    with pytest.raises(ValueError,match='incomplete'):
        processor._apply_hf_processor_main(Items(({}, {'input_audio_features':object()})),{})
