import json
from pathlib import Path
from multimodal_jev.sft import select_training
from multimodal_jev.task_schema import Candidate,DecisionSchema,JevExample


def test_training_selection_contains_complete_views_from_train(tmp_path):
    schema=DecisionSchema('s','choice','pick',(Candidate('a','A'),Candidate('b','B')))
    rows=[]
    for label in ['a','b']:
        for group in ['1','2']:
            for view in ['native','permutation','rebind','missing_candidate','noul.True','noul.False']:
                uid=label+group
                rows.append(JevExample(uid+view,'train','fixture.wav',schema,label,
                    {'source_group':uid,'variant':view}).to_record())
    path=tmp_path/'train.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    config={'train_manifest':str(path),'groups_per_class':1,'seed':17}
    selected=select_training(config)
    assert len(selected)==12 and len({x.metadata['source_group'] for x in selected})==2
    assert select_training(config)==selected
