import copy
import hashlib
import json
import pytest
from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_jev_full import expand_bundle,values_from_target
from multimodal_jev.slurp_jev_comparison import PairedAccumulator,require_full_roster
from test_slurp_jev_full import fixture


def predictions(bundle,bank,values):
    return [dict(example_id=r['id'],replicate=0,request_sha256=hashlib.sha256(canonical(r)).hexdigest(),
                 raw_text=json.dumps({'value':values[r['id'].rsplit('::',1)[-1]]}))
            for r in expand_bundle(bundle,bank)]


def test_strict_repairs_repeated_entities_and_group_dependence():
    bank,bundle,native=fixture();values=values_from_target(native,bank)
    gold=dict(id=bundle['id'],values=values,bundle_sha256=hashlib.sha256(canonical(bundle)).hexdigest())
    correct=predictions(bundle,bank,values);parent=copy.deepcopy(correct)
    for row in parent:
        if row['example_id'].endswith('entity.artist_name.values'):
            row['raw_text']='```json\n'+json.dumps({'value':[]})+'\n```'
        if row['example_id'].endswith('entity.person.values'):
            row['raw_text']=json.dumps({'value':['amy']})
        if row['example_id'].endswith('entity.person.present'):
            row['raw_text']=json.dumps({'value':1})
    scored=PairedAccumulator();scored.add(bundle,gold,bank,parent,correct)
    report=scored.report()
    assert report['by_task']['family.values']['fixed']==2
    assert report['by_task']['family.values.nonempty']['fixed']==1
    assert report['by_task']['family.values.empty']['fixed']==1
    assert report['by_task']['entity.person.present']['fixed']==1
    assert report['all_tasks_by_recording']['fixed']==1
    with pytest.raises(ValueError,match='duplicate'):scored.add(bundle,gold,bank,parent,correct)
    second={**bundle,'id':'same-group-second'}
    gold2={**gold,'id':second['id'],'bundle_sha256':hashlib.sha256(canonical(second)).hexdigest()}
    second_good=predictions(second,bank,values);second_bad=copy.deepcopy(second_good)
    second_bad[0]['raw_text']='invalid'
    scored.add(second,gold2,bank,second_good,second_bad)
    report=scored.report()
    assert report['recordings']==2 and report['source_groups']==1
    assert report['all_tasks_by_recording']['fixed']==1 and report['all_tasks_by_recording']['regressed']==1
    assert report['all_tasks_all_recordings_by_source_group']['both_wrong']==1
    with pytest.raises(ValueError,match='complete bundle'):
        PairedAccumulator().add(bundle,gold,bank,parent[:-1],correct)


def test_full_roster_rejects_missing_prompt_shard_and_overlap():
    plan=dict(recordings=3,source_groups=2,decisions=504,tasks_per_recording=168,shards=[
        dict(id='s0',ordered_bundle_ids=['a','b'],source_groups=['g0'],recordings=2,decisions=336),
        dict(id='s1',ordered_bundle_ids=['c'],source_groups=['g1'],recordings=1,decisions=168)])
    prompts={'parent':'a','p1':'b'};runs={p:{'s0':'path','s1':'path'} for p in prompts}
    require_full_roster(plan,prompts,runs)
    with pytest.raises(ValueError,match='every registered'):require_full_roster(plan,prompts,{'parent':runs['parent']})
    altered=copy.deepcopy(runs);del altered['p1']['s1']
    with pytest.raises(ValueError,match='every registered'):require_full_roster(plan,prompts,altered)
    for field,value in [('source_groups',['g0']),('ordered_bundle_ids',['a']),('decisions',1)]:
        altered=copy.deepcopy(plan);altered['shards'][1][field]=value
        with pytest.raises(ValueError,match='coverage'):require_full_roster(altered,prompts,runs)
