import json
from pathlib import Path
import pytest
from multimodal_jev.baseline import select_examples, prepare_selection
from multimodal_jev.task_schema import Candidate, DecisionSchema, JevExample


def manifest(path, split):
    schema=DecisionSchema('s','choice','Choose',(Candidate('a','A'),Candidate('b','B')))
    records=[]
    for target in ('a','b'):
        for index in range(4):
            group=f'{split}-{target}-{index}'
            records.append(JevExample(group,split,'/audio.wav',schema,target,
                {'variant':'native','speaker_id':split,'source_group':group,'audio_sha256':group}).to_record())
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    return records


def test_selection_independent_of_source_order_and_stratified(tmp_path):
    path=tmp_path/'train.jsonl'; records=manifest(path,'train')
    first=select_examples(path,'train',2)
    path.write_text(''.join(json.dumps(r)+'\n' for r in reversed(records)))
    assert first==select_examples(path,'train',2)
    assert [ex.target for ex in first]==['a','a','b','b']
    with pytest.raises(ValueError): select_examples(path,'train',5)
    with pytest.raises(ValueError): select_examples(path,'valid',1)


def test_selection_rejects_speaker_leakage(tmp_path):
    train=tmp_path/'train';valid=tmp_path/'valid'
    manifest(train,'train');records=manifest(valid,'valid')
    cfg={'train_manifest':str(train),'valid_manifest':str(valid),'train_per_class':2,'valid_per_class':2}
    assert len(prepare_selection(cfg)[0])==4
    for record in records: record['metadata']['speaker_id']='train'
    valid.write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError,match='speaker'):prepare_selection(cfg)
