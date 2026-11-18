"""Portable publication scoring must not reopen developer review directories."""
import hashlib
import json
import shutil

import pytest
from test_slurp_official_training import fixture
from multimodal_jev.baseline import file_hash
from multimodal_jev import slurp_official_training as legacy
from multimodal_jev import slurp_portable_release as portable
from tools.jev.evaluate_context_event_records import evaluate


def setup(tmp_path):
    origin=tmp_path/'origin';origin.mkdir();config,v1=fixture(origin)
    tasks=__import__('pathlib').Path(config['tasks'])
    task=json.loads(tasks.read_text())
    for i in range(57):
        name='fixture_'+str(i)
        task['intent']['schema']['candidates'].append(dict(id=name,description=name))
        task['intent']['candidate_pairs'][name]=['fixture',str(i)]
    tasks.write_text(json.dumps(task));config['pins'][str(tasks)]=file_hash(tasks)
    legacy.publish(config,v1)
    rows,task,_=legacy.load(v1,file_hash(v1/'MANIFEST.json'),'selection',purpose='evaluate')
    ids=[r['id'] for r in task['schema']['candidates']]
    records=[]
    for r in rows:
        identity=dict(policy_id='policy',positions=list(range(60)),inputs={'fixture':'no model execution'})
        rid=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        records.append(dict(id=r['id'],policy_id='policy',request_identity=identity,request_id=rid,
            prob=[.9 if candidate==r['intent'] else .1 for candidate in ids]))
    predictions=tmp_path/'predictions.json';predictions.write_text(json.dumps(dict(records=records)))
    return origin,v1,predictions


def test_same_metrics_after_relocation_and_removal_of_source_evidence(tmp_path):
    origin,v1,predictions=setup(tmp_path)
    before=evaluate(v1,file_hash(v1/'MANIFEST.json'),predictions,file_hash(predictions),'policy','selection')
    v2=tmp_path/'v2';portable.migrate(v1,file_hash(v1/'MANIFEST.json'),v2)
    moved=tmp_path/'relocated';v2.rename(moved);shutil.rmtree(origin)
    after=evaluate(moved,file_hash(moved/'MANIFEST.json'),predictions,file_hash(predictions),'policy','selection')
    assert before['metrics']==after['metrics']
    assert after['metrics']['queries']==2 and after['metrics']['query_macro_set_f1']==1


@pytest.mark.parametrize('fault',['missing','duplicate','stale','identity','training_role'])
def test_portability_preserves_complete_split_and_policy_checks(tmp_path,fault):
    origin,v1,predictions=setup(tmp_path)
    v2=tmp_path/'v2';portable.migrate(v1,file_hash(v1/'MANIFEST.json'),v2)
    data=json.loads(predictions.read_text());role='selection'
    if fault=='missing':data['records'].pop()
    if fault=='duplicate':data['records'][1]=data['records'][0]
    if fault=='stale':data['records'][0]['policy_id']='wrong'
    if fault=='identity':data['records'][0]['request_id']='wrong'
    if fault=='training_role':role='learning'
    predictions.write_text(json.dumps(data))
    with pytest.raises(ValueError):evaluate(v2,file_hash(v2/'MANIFEST.json'),predictions,file_hash(predictions),'policy',role)
