import copy
from types import SimpleNamespace
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev import simple_audio_batch as module


def fixture(monkeypatch):
    calls=[]
    class Request:
        questions=('Final',)
        def __init__(self,audio):self.audio=audio
        def model_input(self):return dict(audio=self.audio,questions=['fixed60'])
    def example(row,*args):
        return Request(row['audio']),dict(targets=[row['target']],supervised=[True],span_weights=[1.]),None
    monkeypatch.setattr(module,'example',example)
    monkeypatch.setattr(module,'file_hash',lambda path:'hash')
    def audio(path,**kw):
        calls.append(path)
        return path,dict(input_waveform_sha256='a'*64,output_waveform_sha256='b'*64,sample_rate=16000,config={})
    monkeypatch.setattr(module,'load_audio',audio)
    monkeypatch.setattr(module,'encode_span_request',lambda *args:(dict(input_ids=torch.tensor([[1,2,3]]),
        attention_mask=torch.ones(1,3,dtype=torch.long),input_features=torch.ones(1,2,3),
        input_features_mask=torch.ones(1,3,dtype=torch.long)),torch.tensor([[2]])))
    monkeypatch.setattr(module,'span_labels',lambda *args:[dict(token_ids=list(range(60)))])
    population={str(i):dict(id=str(i),audio=f'{i}.wav',audio_sha256='hash',role='learning',known_pending_source=False,
        source_group=str(i),target=i%60) for i in range(128)}
    slots=[dict(id=str(i),source_group=str(i),view='direct',variant='native',weight=1.,padding=False) for i in range(128)]
    return population,slots,SimpleNamespace(tokenizer=None),calls


def test_full_batch_shared_renderer_and_reference_no_gold(monkeypatch):
    population,slots,processor,calls=fixture(monkeypatch)
    slots[-1]=dict(slots[0],weight=0.,padding=True)
    entries,renderer,receipt=module.compile_batch(population,{},slots,processor,policy_id='base:actor',global_batch=128)
    assert receipt['global_real_requests']==127 and len(entries)==128 and len(renderer)==len(calls)==127
    assert entries[-1]['item']['inputs'] is entries[0]['item']['inputs']
    assert entries[0]['item']['inputs'] is renderer[0][1]
    ref=module.renderer_for_policy(renderer,'base:reference')
    assert ref[0][1] is renderer[0][1] and ref[0][4]!=renderer[0][4]
    assert ref[0][5]==dict(renderer[0][5],policy_id='base:reference')
    for row in population.values():row['target']=(row['target']+1)%60
    changed,again,_=module.compile_batch(population,{},slots,processor,policy_id='base:actor',global_batch=128)
    assert [r[4:6] for r in renderer]==[r[4:6] for r in again]
    assert not torch.equal(entries[0]['item']['targets'],changed[0]['item']['targets'])


@pytest.mark.parametrize('fault',['partial','selection','pending','group','variant','negative','padding',
    'interleaved_padding','duplicate','audio','nan','all_padding'])
def test_reject_nonlearning_or_changed_schedule(monkeypatch,fault):
    population,slots,processor,_=fixture(monkeypatch)
    if fault=='partial':slots.pop()
    elif fault=='selection':population['0']['role']='selection'
    elif fault=='pending':population['0']['known_pending_source']=True
    elif fault=='group':slots[0]['source_group']='other'
    elif fault=='variant':slots[0]['variant']='permute'
    elif fault=='negative':slots[0]['weight']=-1.
    elif fault=='padding':slots[-1]['padding']=True
    elif fault=='interleaved_padding':slots[0].update(padding=True,weight=0.)
    elif fault=='duplicate':slots[1]=copy.deepcopy(slots[0])
    elif fault=='audio':population['0']['audio_sha256']='changed'
    elif fault=='nan':slots[0]['weight']=float('nan')
    else:
        for s in slots:s.update(padding=True,weight=0.)
    with pytest.raises(ValueError):module.compile_batch(population,{},slots,processor,policy_id='base:actor',global_batch=128)
