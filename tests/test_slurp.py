import json
from pathlib import Path
import pytest
from multimodal_jev.slurp import build,slot_target
from multimodal_jev.task_schema import JevExample


def fixture(tmp_path):
    root=tmp_path/'slurp';(root/'metadata').mkdir(parents=True);(root/'slurp_real').mkdir()
    config={'protocol':'fixture','heldout_scenarios':['calendar'],'slot_types':['person']}
    metadata={}
    for split,offset in [('train',0),('devel',10),('test',20)]:
        rows=[]
        for i,scenario in enumerate(['music','calendar','weather']):
            uid=offset+i;filename=f'{uid}.flac'
            sentence=f'ask {scenario} {split}'
            row={'slurp_id':uid,'scenario':scenario,'intent':scenario+'_query','action':'query',
                 'sentence':sentence,'tokens':[{'id':0,'surface':'ask'},{'id':1,'surface':'amy'}],
                 'entities':[{'type':'person','span':[1]}], 'recordings':[{'file':filename}]}
            rows.append(row);metadata[str(uid)]={'recordings':{filename:{'usrid':split}}}
            (root/'slurp_real'/filename).write_bytes(f'audio-{uid}'.encode())
        (root/'metadata'/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (root/'metadata/metadata.json').write_text(json.dumps(metadata))
    return root,config


def test_native_span_and_missing():
    row={'tokens':[{'id':0,'surface':'Amy'},{'id':1,'surface':'Bob'}],
         'entities':[{'type':'person','span':[1]},{'type':'person','span':[0]}]}
    assert slot_target(row,'person')==('Amy',[0])
    assert slot_target(row,'date')==(None,[])
    row['entities']=[{'type':'person','span':[9]}]
    with pytest.raises(ValueError,match='span'):slot_target(row,'person')


def test_domain_holdout_and_no_label_leakage(tmp_path):
    root,cfg=fixture(tmp_path)
    output,receipt=build(root,tmp_path/'out',cfg)
    train=[JevExample.from_record(json.loads(l)) for l in (output/'train.jsonl').read_text().splitlines()]
    assert len(train)==8 and all(x.metadata['scenario']!='calendar' for x in train)
    holdout=[JevExample.from_record(json.loads(l)) for l in (output/'heldout-test.jsonl').read_text().splitlines()]
    assert len(holdout)==4 and all(x.split=='test' for x in holdout)
    assert all('transcript_reference' not in x.model_input() for x in train)
    assert receipt['exclusions']['heldout_domain_train_recordings']==1
    assert train[0].target==train[1].target and train[0].target!=train[2].target
    assert receipt['native_intents_absent_from_train']['test']==[]
    with pytest.raises(FileExistsError):build(root,tmp_path/'out',cfg)


def test_duplicate_text_training_removed_and_audio_overlap_rejected(tmp_path):
    root,cfg=fixture(tmp_path)
    train_path=root/'metadata/train.jsonl'; rows=[json.loads(l) for l in train_path.read_text().splitlines()]
    rows[0]['sentence']='ASK music devel'  # normalized duplicate
    train_path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    out,receipt=build(root,tmp_path/'out',cfg)
    assert receipt['exclusions']['train_text_shared_with_evaluation_recordings']==1
    (root/'slurp_real/20.flac').write_bytes((root/'slurp_real/2.flac').read_bytes())
    with pytest.raises(ValueError,match='audio bytes'):build(root,tmp_path/'second',cfg)
