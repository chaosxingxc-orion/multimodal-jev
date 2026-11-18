import copy
import json
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash
from tools.reports.context_attribute_execution_candidate import configurations,validate_pair,authenticate,DEPENDENCIES

BASE=Path(__file__).resolve().parents[1]/'.local/decision-objective-research/context-rl-grpo-arithmetic-v2-package'


def pair(arm='q_credit',algorithm='grpo_v1'):
    c=json.loads((BASE/'config.json').read_text());s=json.loads((BASE/'service-config.json').read_text())
    before=copy.deepcopy((c,s))
    result=configurations(c,s,root='/data/xingchao/multimodal-jev/simple-final-v1/context-attribute-fixture',
        source_root='/candidate/source',dataset='/candidate/data',schedule='/candidate/schedule',
        schedule_sha256='a'*64,arm=arm,algorithm=algorithm,pins={'/candidate/source':'b'*64})
    assert (c,s)==before
    return result


@pytest.mark.parametrize('arm',['direct','q_input','q_credit'])
@pytest.mark.parametrize('algorithm',['grpo_v1','rloo_v1'])
def test_shared_initial_configuration_preserves_model_budget(arm,algorithm):
    c,s=pair(arm,algorithm)
    assert validate_pair(c,s)
    assert c['stop_update']==2 and not c['training_dispatch_allowed']
    assert s['policy']['max_decisions']==(60 if arm=='direct' else 124)
    assert c['algorithm']==algorithm and c['reference_adapter']==c['initial_adapter']
    assert 'formal_release' not in c


@pytest.mark.parametrize('fault',['formal','budget','capacity','policy','contract','devices','scope','pins','socket','arithmetic'])
def test_incompatible_service_rejected(fault):
    c,s=pair()
    if fault=='formal':c['training_dispatch_allowed']=True
    if fault=='budget':c['stop_update']=634
    if fault=='capacity':s['policy']['max_decisions']=60
    if fault=='policy':s['policy']['policy_id']='wrong'
    if fault=='contract':s['policy']['contract']={}
    if fault=='devices':s['policy']['physical_devices']=[3]
    if fault=='scope':s['policy']['process_device_scope']['visible_ordinal']='7'
    if fault=='pins':s['pins']={}
    if fault=='socket':s['socket_path']='/wrong.sock'
    if fault=='arithmetic':c['binary_policy_arithmetic']='old'
    with pytest.raises(ValueError):validate_pair(c,s)


def test_resume_uses_update1_service_while_reference_stays_stage1(tmp_path):
    c,s=pair();reference=c['source_policy_id'];resume=tmp_path/'prior';resume.mkdir()
    policy=resume/'policy-0001';policy.mkdir();(policy/'MANIFEST.json').write_text(json.dumps({'policy_id':'updated'}))
    shards={}
    for rank in range(4):
        p=resume/f'rank-{rank}/checkpoint-0001.pt';p.parent.mkdir();p.write_bytes(b'synthetic checkpoint')
        shards[str(p.relative_to(resume))]=file_hash(p)
    commit=resume/'checkpoint-0001-committed.json'
    commit.write_text(json.dumps(dict(updates=1,rank_shards=shards,joint_policy_directory='policy-0001',
        joint_policy_manifest_sha256=file_hash(policy/'MANIFEST.json'))))
    c.update(start_update=1,resume_root=str(resume),initial_adapter=str(resume/'adapter-0001'))
    c['pins'][str(commit)]=file_hash(commit);s['pins']=copy.deepcopy(c['pins'])
    s['policy'].update(snapshot=str(policy),policy_id='updated',adapter=c['initial_adapter'])
    assert validate_pair(c,s) and c['source_policy_id']==reference
    (resume/'rank-3/checkpoint-0001.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError):validate_pair(c,s)


def test_exact_shared_runtime_identity(tmp_path):
    root=Path(__file__).resolve().parents[1]
    authenticate(root)
    for name in DEPENDENCIES:
        dest=tmp_path/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((root/name).read_bytes())
    authenticate(tmp_path)
    (tmp_path/'tools/jev/context_process_supervisor.py').write_text('changed')
    with pytest.raises(ValueError):authenticate(tmp_path)
