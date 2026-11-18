import importlib.util
import json
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.baseline import file_hash


def fixture(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('probe_compare',Path(__file__).parents[1]/'tools/jev/compare_simple_instruction_probe.py')
    tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
    from multimodal_jev import policy_identity
    from multimodal_jev.slurp_instruction_resume import FIELDS
    audits=[];roots=[]
    for tag in ('a','b'):
        root=tmp_path/tag;roots.append(root);(root/'run').mkdir(parents=True)
        config={k:'fixed' for k in FIELDS};config.update(global_batch=128)
        (root/'config.json').write_text(json.dumps(config))
        for step in (0,2):
            adapter=root/'run'/f'adapter-step{step:04d}';adapter.mkdir()
            (adapter/'adapter_model.safetensors').write_bytes(b'exact-test-weights')
            (adapter/'adapter_config.json').write_text(json.dumps(dict(r=4,target_modules=['q','v'] if tag=='a' else ['v','q'])))
        for step in range(2):
            (root/'run'/f'batch-{step:04d}.json').write_text(json.dumps(dict(records=[dict(audio='same',target=3)],policy_id=tag)))
        for rank in range(4):
            out=root/'run'/f'rank-{rank}';out.mkdir()
            t=torch.tensor([.1234567]);state=dict(world_size=4,rank=rank,updates=2,names=['w'],parameters={'w':t},
                optimizer=dict(param_groups=[dict(params=[0])],state={0:dict(step=torch.tensor(2.),exp_avg=t/10,exp_avg_sq=t/100)}))
            torch.save(state,out/'checkpoint-0002.pt')
        audit=root/'audit.json'
        audit.write_text(json.dumps(dict(passed=True,exact_master_export=True,exact_peft_reload=True,
            durable_commit_verified=True,updates=2,training_root=str(root),
            adapter_id=policy_identity.checkpoint_identity(root/'run/adapter-step0002')[0],
            files={str(root/'config.json'):file_hash(root/'config.json')})))
        audits.append(audit)
    return tool,roots,audits


def test_exact_state_across_bookkeeping_control(tmp_path,monkeypatch):
    tool,roots,audits=fixture(tmp_path,monkeypatch)
    assert tool.run(*audits)['exact_fp32_masters_and_adam']


@pytest.mark.parametrize('fault',['pin','micro','records','master','adapter','audit'])
def test_reject_false_control_identity_or_sub_bf16_drift(tmp_path,monkeypatch,fault):
    tool,roots,audits=fixture(tmp_path,monkeypatch);root=roots[1]
    audit=json.loads(audits[1].read_text())
    if fault in ('pin','micro'):
        p=root/'config.json';c=json.loads(p.read_text());c['microbatch_per_rank']=2;p.write_text(json.dumps(c))
        if fault=='micro':audit['files'][str(p)]=file_hash(p)
    elif fault=='records':(root/'run/batch-0000.json').write_text(json.dumps(dict(records=[])))
    elif fault=='master':
        p=root/'run/rank-2/checkpoint-0002.pt';s=torch.load(p,weights_only=True)
        s['parameters']['w']=torch.nextafter(s['parameters']['w'],torch.tensor([float('inf')]))
        torch.save(s,p)
    elif fault=='adapter':audit['adapter_id']='different'
    else:audit['exact_peft_reload']=False
    audits[1].write_text(json.dumps(audit))
    with pytest.raises(ValueError):tool.run(*audits)
