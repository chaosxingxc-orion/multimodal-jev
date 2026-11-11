import json
from copy import deepcopy
import pytest
from test_span_rule_transfer import populations
from multimodal_jev.span_rule_transfer import make_transfer_protocol
from multimodal_jev.span_transfer_evaluation import cell_metrics,evaluation_rows,POPULATION
from multimodal_jev.baseline import file_hash


def fixture(tmp_path):
    learning,selection=populations();study=make_transfer_protocol(learning,selection);study['pins']={}
    for role,rows in [('learning',learning),('selection',selection)]:
        root=tmp_path/role;root.mkdir();f=root/(role+'.jsonl');f.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (root/'MANIFEST.json').write_text(json.dumps({'files':{f.name:file_hash(f)}}))
        study[role+'_release']=str(root);study['pins'][str(root/'MANIFEST.json')]=file_hash(root/'MANIFEST.json')
    sp=tmp_path/'study.json';sp.write_text(json.dumps(study));ep=tmp_path/'evaluation.json'
    ep.write_text(json.dumps(dict(schema='jev-transfer-frozen-evaluation-v1',study_sha256=file_hash(sp),optimizer_updates=0,training_admitted=False,policy_ids=['base:initial','base:final'])))
    c=dict(evaluation_protocol=str(ep),transfer_protocol=str(sp),pins={str(p):file_hash(p) for p in (sp,ep)},population=POPULATION,evaluation_role='selection',base_id='base',adapter_id='final',release=study['selection_release'],request_ids=[r['id'] for r in selection])
    return study,learning,selection,c


def test_complete_roles_and_four_metrics(tmp_path):
    study,learning,selection,c=fixture(tmp_path)
    assert evaluation_rows(c)==json.loads(json.dumps(selection))
    rows=learning+selection
    a=[dict(id=r['id'],predictions=[False]*len(r['targets'])) for r in rows]
    b=[dict(id=r['id'],predictions=r['targets']) for r in rows]
    result=cell_metrics(study,rows,a,b)
    assert len(result)==4
    for cell in result.values():
        assert all(r['final_accuracy']==1 for r in cell['after']['per_rule'])
        assert all(r['after_accuracy']==1 for r in cell['paired_source_intervals']['arms'].values())


@pytest.mark.parametrize('fault',['unlisted_policy','wrong_role','missing','changed_protocol','training','duplicate'])
def test_evaluation_contract_rejects_changed_scope(tmp_path,fault):
    study,learning,selection,c=fixture(tmp_path)
    if fault=='unlisted_policy':c['adapter_id']='best-middle'
    elif fault=='wrong_role':c['evaluation_role']='test'
    elif fault=='missing':c['request_ids']=c['request_ids'][:-1]
    elif fault=='changed_protocol':c['pins']={}
    elif fault=='training':
        p=c['evaluation_protocol'];d=json.load(open(p));d['optimizer_updates']=1
        open(p,'w').write(json.dumps(d));c['pins'][p]=file_hash(p)
    else:c['request_ids'].append(c['request_ids'][0])
    with pytest.raises(ValueError):evaluation_rows(c)


@pytest.mark.parametrize('fault',['missing_prediction','duplicate_prediction','missing_cell','missing_arm'])
def test_cell_comparison_requires_complete_four_way_population(tmp_path,fault):
    study,learning,selection,_=fixture(tmp_path);rows=learning+selection
    a=[dict(id=r['id'],predictions=r['targets']) for r in rows];b=deepcopy(a)
    if fault=='missing_prediction':b.pop()
    elif fault=='duplicate_prediction':b.append(deepcopy(b[0]))
    elif fault=='missing_cell':study['evaluation_cells'].pop(next(iter(study['evaluation_cells'])))
    else:
        uid=rows[0]['id'];rows=rows[1:];a=a[1:];b=b[1:]
        for cell in study['evaluation_cells'].values():
            if uid in cell['request_ids']:cell['request_ids'].remove(uid)
    with pytest.raises(ValueError):cell_metrics(study,rows,a,b)
