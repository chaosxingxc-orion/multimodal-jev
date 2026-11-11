import json
from copy import deepcopy
import pytest
from test_span_rule_transfer import populations
from multimodal_jev.baseline import file_hash
from multimodal_jev.span_rule_balance import balanced_rule_schedule
from multimodal_jev.span_rule_transfer import make_transfer_protocol
from multimodal_jev.span_transfer_runtime import branch_protocol,step_rows,version_rows,cursor,validate_parent
from multimodal_jev.span_rule_population import select_rule_rows


def fixture(tmp_path):
    learning,selection=populations();release=tmp_path/'release';release.mkdir()
    (release/'learning.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in learning))
    (release/'schedule.json').write_text(json.dumps(balanced_rule_schedule(learning,17)))
    manifest=dict(schema='jev-span-balanced-rule-controls-v1',files={n:file_hash(release/n) for n in ['learning.jsonl','schedule.json']})
    (release/'MANIFEST.json').write_text(json.dumps(manifest))
    p=make_transfer_protocol(learning,selection);p.update(learning_release=str(release),pins={str(release/'MANIFEST.json'):file_hash(release/'MANIFEST.json')},training_arms=['structured','final_only'],methods=['sft','joint-grpo','exact-conditional-marginal-ppo'],auxiliary_weights=[0.,.2],runtime_profile='profile',optimizer={'lr':1e-4},joint_grpo_samples_per_request=32,initializer={'adapter_id':'initial'})
    path=tmp_path/'protocol.json';path.write_text(json.dumps(p))
    c=dict(transfer_protocol=str(path),pins={str(path):file_hash(path)},release=str(release),seed=17,arm='structured',method='conditional',auxiliary_weight=.2,runtime_profile='profile',lr=1e-4,samples_per_request=32,batch_index=0,adapter_id='initial')
    c['request_ids']=p['schedules']['17'][0]['arms']['structured']
    return p,c


def test_runtime_gate_covers_previous_and_next_step(tmp_path):
    p,c=fixture(tmp_path)
    for version in (0,1,12):
        ids=[r['id'] for r in select_rule_rows(c['release'])]
        for i in (version-1,version):
            if 0<=i<12:
                ids.extend(k for k in p['schedules']['17'][i]['arms']['structured'] if k not in ids)
        rows=version_rows({**c,'version_index':version,'request_ids':ids})
        assert [r['id'] for r in rows]==ids
        assert len(rows)>=216
    assert len(step_rows(c))==8


def test_resume_requires_exact_next_cursor_and_same_branch(tmp_path):
    p,c=fixture(tmp_path);validate_parent(c,None)
    c.update(batch_index=1,adapter_id='trained')
    receipt=dict(training_state=cursor(c,1),completed_updates=1,adapter_id='trained')
    validate_parent(c,receipt)
    for key,value in [('seed',29),('method','sft'),('auxiliary_weight',0.),('batch_index',2),('adapter_id','other')]:
        bad={**c,key:value}
        with pytest.raises(ValueError):validate_parent(bad,receipt)


@pytest.mark.parametrize('fault',['wrong_batch','heldout','bad_lr','single_aux','bad_pin','skip_version','no_initial'])
def test_reject_contract_changes(tmp_path,fault):
    p,c=fixture(tmp_path)
    if fault=='wrong_batch':c['request_ids']=p['schedules']['17'][1]['arms']['structured']
    elif fault=='heldout':c['request_ids']=p['evaluation_cells']['learning_source:heldout_rule_group']['request_ids'][:8]
    elif fault=='bad_lr':c['lr']=.1
    elif fault=='single_aux':c['arm']='final_only'
    elif fault=='bad_pin':c['pins']={}
    elif fault=='skip_version':c['batch_index']=12
    elif fault=='no_initial':
        c['adapter_id']='rule-trained'
        with pytest.raises(ValueError):validate_parent(c,None)
        return
    with pytest.raises(ValueError):step_rows(c)
