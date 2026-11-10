from copy import deepcopy
import json
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_jev_shards import make_plan,verify_plan,load_shard,materialize
from test_sfc_teacher_feedback import fixture as population_fixture


def fixture(tmp_path):
    rows,_,partition=population_fixture();release=tmp_path/'release';release.mkdir();pins={}
    for split in ('train_id','test_id','test_ood'):
        values=[json.loads(json.dumps(r).replace('train_id',split)) for r in rows]
        path=release/f'{split}.requests.jsonl';path.write_bytes(b''.join(map(canonical,values)))
        pins[split]=dict(bundles=4,source_groups=2,decisions=24,requests_sha256=file_hash(path))
    (release/'MANIFEST.json').write_bytes(canonical(dict(identity={'protocol':'sfc-jev-bundle-v1'},splits=pins)))
    partition.update(release=str(release.resolve()),manifest_sha256=file_hash(release/'MANIFEST.json'),
                     train_requests_sha256=file_hash(release/'train_id.requests.jsonl'))
    path=tmp_path/'partition.json';path.write_bytes(canonical(partition))
    return release,path


def test_whole_test_and_indivisible_group_can_exceed_target(tmp_path):
    release,_=fixture(tmp_path)
    # No targets exist at all: the plan and materializer cannot depend on them.
    plan=make_plan(release,'test_id',10);assert verify_plan(plan)==plan
    assert plan['bundles']==4 and plan['decisions']==24 and len(plan['shards'])==2
    assert plan['indivisible_group_exceeds_target'] is True
    seen=[]
    for shard in plan['shards']:
        rows=load_shard(plan,shard['id']);seen.extend(r['id'] for r in rows)
        assert [len(r['audio_sha256']) for r in rows]==[1,2]
        assert len({r['source_group'] for r in rows})==1 and shard['decisions']==12
    assert len(set(seen))==4
    path=tmp_path/'plan.json';path.write_bytes(canonical(plan))
    result=materialize(path,file_hash(path),'shard-0000',tmp_path/'shard')
    assert result['bundles']==2 and result['decisions']==12 and not result['targets_read']
    assert not list((tmp_path/'shard').glob('*targets*'))
    with pytest.raises(FileExistsError):materialize(path,file_hash(path),'shard-0000',tmp_path/'shard')


def test_train_learning_selection_are_exhaustive_disjoint_and_pinned(tmp_path):
    release,path=fixture(tmp_path);plans={}
    for part in ('learning','selection'):
        plans[part]=make_plan(release,'train_id',1024,partition_path=path,partition_hash=file_hash(path),part=part)
        assert plans[part]['bundles']==2 and plans[part]['decisions']==12
        assert verify_plan(plans[part])==plans[part]
    left=set(plans['learning']['shards'][0]['source_groups']);right=set(plans['selection']['shards'][0]['source_groups'])
    assert not left&right and len(left|right)==2
    with pytest.raises(ValueError):make_plan(release,'train_id')
    with pytest.raises(ValueError):make_plan(release,'test_id',partition_path=path,partition_hash=file_hash(path),part='learning')
    path.write_text('{}')
    with pytest.raises(ValueError,match='partition changed'):verify_plan(plans['learning'])


@pytest.mark.parametrize('change',['missing_shard','duplicate_shard','missing_turn','missing_group','decision_count','ids','digest','partition'])
def test_plan_cannot_silently_shrink_or_duplicate_population(tmp_path,change):
    release,_=fixture(tmp_path);plan=make_plan(release,'test_id',10)
    if change=='missing_shard':plan['shards'].pop()
    if change=='duplicate_shard':plan['shards'].append(deepcopy(plan['shards'][0]))
    if change=='missing_turn':plan['shards'][0]['ordered_bundle_ids'].pop()
    if change=='missing_group':plan['shards'][0]['source_groups']=[]
    if change=='decision_count':plan['shards'][0]['decisions']-=1
    if change=='ids':plan['shards'][0]['id']='other'
    if change=='digest':plan['shards'][0]['decision_keys_sha256']='a'*64
    if change=='partition':plan['partition']=dict(path='/unused',sha256='a'*64,part='learning')
    with pytest.raises(ValueError):verify_plan(plan)


@pytest.mark.parametrize('change',['missing_turn','changed_decision','prefix','wrong_group','supervision'])
def test_rehashed_broken_release_fails_structural_checks(tmp_path,change):
    release,_=fixture(tmp_path);path=release/'test_id.requests.jsonl'
    rows=list(map(json.loads,path.read_text().splitlines()))
    if change=='missing_turn':rows.pop()
    if change=='changed_decision':rows[0]['input']['decisions'].pop()
    if change=='prefix':rows[1]['audio_sha256'][0]='f'*64
    if change=='wrong_group':rows[0]['source_group']='other'
    if change=='supervision':rows[0]['answers']={'f00.present':True}
    path.write_bytes(b''.join(map(canonical,rows)))
    manifest=json.loads((release/'MANIFEST.json').read_text());manifest['splits']['test_id']['requests_sha256']=file_hash(path)
    (release/'MANIFEST.json').write_bytes(canonical(manifest))
    with pytest.raises(ValueError):make_plan(release,'test_id')


@pytest.mark.parametrize('target',[0,-1,True,1.5,None])
def test_invalid_shard_limits_rejected(tmp_path,target):
    release,_=fixture(tmp_path)
    with pytest.raises(ValueError):make_plan(release,'test_id',target)
