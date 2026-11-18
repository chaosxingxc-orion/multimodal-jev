import pytest
from multimodal_jev import simple_selection_inputs as module


def fixture(monkeypatch):
    values=[str(i) for i in range(60)]
    class Question: pass
    q=Question();q.values=values
    class Request:
        questions=[q]
        def model_input(self):return {'audio':'/wave','questions':['Final']}
    monkeypatch.setattr(module,'compile_intent',lambda *args:Request())
    rows=[dict(id='one',audio='/wave',audio_sha256='hash')]
    artifact=dict(passed=True,population_mode='full_selection',release_sha256='release',recordings=1,requests=1,
        maximum_input_tokens=3,records=[dict(id='one',recording_id='one',audio_sha256='hash',view='direct',variant='native',
            request=Request().model_input(),input_tokens=3,positions=[[2]],input_ids={'shape':[1,3]},
            candidate_mapping=[dict(values=values,token_ids=list(range(60)))])])
    return rows,artifact


def test_complete_label_free_population(monkeypatch):
    rows,artifact=fixture(monkeypatch)
    assert module.verify(artifact,rows,{},'release')['requests']==1


@pytest.mark.parametrize('fault',['missing','duplicate','gold','values','tokens','audio','position'])
def test_wrong_or_incomplete_artifact(monkeypatch,fault):
    rows,a=fixture(monkeypatch);r=a['records'][0]
    if fault=='missing':a['records']=[]
    if fault=='duplicate':a['records']*=2
    if fault=='gold':r['request']['target']='class1'
    if fault=='values':r['candidate_mapping'][0]['values']=list(reversed(r['candidate_mapping'][0]['values']))
    if fault=='tokens':r['candidate_mapping'][0]['token_ids']=[0]*60
    if fault=='audio':r['audio_sha256']='other'
    if fault=='position':r['positions']=[[3]]
    with pytest.raises(ValueError):module.verify(a,rows,{},'release')
