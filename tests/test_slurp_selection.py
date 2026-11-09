import importlib.util
import json
from pathlib import Path
import pytest
from multimodal_jev.evaluate import evaluate
from multimodal_jev.task_schema import JevExample,DecisionSchema,Candidate


def rows():
    result=[]
    for i in range(4):
        for variant in ('native','permutation','rebind','slot.date','slot.time','slot.person','slot.place_name'):
            slot=variant.startswith('slot.')
            schema=DecisionSchema('s','open_slot' if slot else 'choice','Which?',
                                  () if slot else (Candidate('a','A'),Candidate('b','B')))
            target=('present' if i%2 else None) if slot else 'a'
            result.append(JevExample(f'{i}:{variant}','valid','fixture',schema,target,
                {'source_group':str(i),'scenario':'calendar' if i<2 else 'music',
                 'partition':'heldout-valid' if i<2 else 'valid','variant':variant}).to_record())
    return result


def test_selection_preserves_views_and_presence_buckets():
    path=Path(__file__).parents[1]/'tools/jev/build_slurp_eval_selection.py'
    spec=importlib.util.spec_from_file_location('selection',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    selected,buckets,support=m.select(rows())
    assert len(selected)==28 and len(buckets)==18
    assert m.select(list(reversed(rows())))[1]==buckets
    assert all(support[p,'slot.date',present]>0 for p in ('valid','heldout-valid') for present in (True,False))
    with pytest.raises(ValueError,match='seven-view'):m.select(rows()[:-1])


def test_domain_metrics_do_not_hide_null_imbalance(tmp_path):
    manifest=tmp_path/'manifest';predictions=tmp_path/'predictions'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows()))
    predictions.write_text(''.join(json.dumps({'example_id':r['example_id'],'value':None})+'\n' for r in rows()))
    result=evaluate(manifest,predictions)
    for cell in result['by_domain']:
        assert cell['accuracy']==(1 if cell['target_presence']=='null' else 0)
