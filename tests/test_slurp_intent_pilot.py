import json
from pathlib import Path
import pytest
from test_slurp_intent_structure import fixture
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.slurp_intent_structure import training_targets
from multimodal_jev.slurp_intent_pilot import load_pilot
import multimodal_jev.slurp_intent_structure as compiler


def prepare(tmp_path):
    task,e=fixture();release=tmp_path/'release';release.mkdir();root=tmp_path/'pilot';root.mkdir()
    evidence=tmp_path/'evidence';evidence.write_text('Explicit alarm setting evidence')
    atomic_json(release/'tasks.json',{'intent':task})
    atomic_json(release/'review-ledger.json',{'evidence_files':{str(evidence):file_hash(evidence)}})
    (release/'train.jsonl').write_text(json.dumps(e.to_record())+'\n')
    atomic_json(release/'MANIFEST.json',dict(schema='slurp-reviewed-training-release-v1',review_scope='test',files={p.name:file_hash(p) for p in release.iterdir()}))
    for arm in ('D','Q','P'):
        request,target=training_targets(e,task,arm);row=dict(id=e.example_id,request=request.model_input())
        target.update(id=e.example_id,request_sha256=content_hash(row),request_weight=1.)
        atomic_json(root/f'{arm}-requests.json',[row]);atomic_json(root/f'{arm}-targets.json',[target])
    manifest=dict(schema='slurp-intent-dqp-pilot-v1',release=str(release),release_manifest_sha256=file_hash(release/'MANIFEST.json'),implementation_sha256=file_hash(compiler.__file__),files={p.name:file_hash(p) for p in root.iterdir()})
    atomic_json(root/'MANIFEST.json',manifest)
    return root,manifest


def test_load_rederives_from_reviewed_targets(tmp_path):
    root,_=prepare(tmp_path)
    for arm,weights in [('D',[1.]),('Q',[0.,0.,1.]),('P',[.1,.1,1.])]:
        rows,audit=load_pilot(root,file_hash(root/'MANIFEST.json'),arm,verify_audio=False)
        assert rows[0]['target']['span_weights']==weights
        assert audit['candidate_intents']==3
    with pytest.raises(FileNotFoundError):load_pilot(root,file_hash(root/'MANIFEST.json'),'D')


@pytest.mark.parametrize('defect',['target','weight','request','review_evidence'])
def test_rehashed_transport_cannot_override_admission(tmp_path,defect):
    root,manifest=prepare(tmp_path)
    if defect=='review_evidence':(tmp_path/'evidence').write_text('changed')
    else:
        name='P-requests.json' if defect=='request' else 'P-targets.json'
        path=root/name;rows=json.loads(path.read_text())
        if defect=='target':rows[0]['targets'][-1]=0
        elif defect=='weight':rows[0]['span_weights']=[10.,10.,1.]
        else:rows[0]['request']['requirements']=['Gold answer is alarm_set']
        atomic_json(path,rows);manifest['files'][name]=file_hash(path);atomic_json(root/'MANIFEST.json',manifest)
    with pytest.raises(ValueError):load_pilot(root,file_hash(root/'MANIFEST.json'),'P',verify_audio=False)
