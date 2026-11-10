import asyncio
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
import sample_sfc_omni_async as runner


@pytest.mark.parametrize('failure',[None,'sampling','audio'])
def test_complete_runner_persists_intents_and_shuts_down(tmp_path,monkeypatch,failure):
    requests=tmp_path/'requests';requests.mkdir()
    rows=[dict(id=str(i),audio_sha256=['pin'],input=dict(context=dict(audios=['wave']),
            decisions=[dict(id=str(j)) for j in range(13)])) for i in range(6)]
    prepared={(r['id'],d['id']):dict(input_ids=NS(tolist=lambda:[1,2]))
              for r in rows for d in r['input']['decisions']}
    engine=NS(model_config=NS(logprobs_mode='raw_logprobs',generation_config='vllm',max_model_len=4096))
    shutdown=[];calls=[]
    engine.shutdown=lambda **kw:shutdown.append(kw)
    fake=ModuleType('vllm.v1.engine.async_llm');fake.AsyncLLM=NS(from_engine_args=lambda _:engine)
    monkeypatch.setitem(sys.modules,'vllm.v1.engine.async_llm',fake)
    monkeypatch.setattr(runner,'file_hash',lambda _: 'bad' if failure=='audio' else 'pin')
    monkeypatch.setattr(runner,'load_sfc_audio',lambda _:([.1],16000))
    class Sampler:
        last_output=None
        def __init__(self,*args,**kwargs):pass
        async def sample(self,row,decision,*args,**kwargs):
            name=runner.digest((row['id'],decision))
            assert (requests/f'{name}.intent.json').exists()
            calls.append(kwargs['request_id'])
            self.last_output=NS(request_id=kwargs['request_id'],finished=failure is None,
                prompt_token_ids=[1,2],outputs=[])
            if failure=='sampling':raise RuntimeError('ambiguous generation failure')
            return [dict(example_id=row['id'],decision_id=decision,replicate=i) for i in range(2)]
    monkeypatch.setattr(runner,'OmniAsyncSampler',Sampler)
    task=runner.generate(NS(),NS(),rows,prepared,'policy',dict(instruction_sha256='instruction'),tmp_path)
    if failure is None:
        records=asyncio.run(task)
        assert len(records)==156 and len(set(calls))==78
        assert len(list(requests.glob('*.intent.json')))==78
        assert len(list(requests.glob('*.raw.json')))==78
        assert len(list(requests.glob('*.result.json')))==78
    else:
        with pytest.raises((RuntimeError,ValueError)):asyncio.run(task)
        assert len(calls)==(1 if failure=='sampling' else 0)
        if failure=='sampling':
            assert len(list(requests.glob('*.failure.json')))==1
            raw=json.loads(next(requests.glob('*.raw.json')).read_text())
            assert raw['output']['finished'] is False
        assert not list(requests.glob('*.result.json'))
    assert shutdown==[dict(timeout=30)]
