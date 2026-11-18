import copy
import json
from pathlib import Path

import pytest

from tools.jev import prepare_context_formal_package as module
from tools.jev.context_formal_runtime import validate


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    candidate = Path('.local/decision-objective-research/context-checkpoint-shared-npu-v1-package')
    c = json.loads((candidate/'config.json').read_text())
    # Packaging test: real pinned candidate source, simulated completed-source
    # and NPU evidence admission. No actual formal publication is implied.
    source = tmp_path/'initial';source.mkdir()
    for folder in ('policy','adapter'):
        (source/folder).mkdir();(source/folder/'fixture').write_text(folder)
    binding = dict(source_step=634, source_policy=str(source/'policy'), source_policy_id='full-stage1',
        reference_policy_id='full-stage1', source_contract=c['source_contract'], reset_optimizer_for_task_stage=True)
    (source/'binding.json').write_text(json.dumps(binding))
    (source/'PACKAGE.json').write_text('{}')
    source_digest = module.sha(source/'PACKAGE.json')
    training = tmp_path/'training';training.mkdir()
    train_config = copy.deepcopy(c)
    train_config.update(encoded_audio_cache=module.PREFIX+'context-learning-cache-v1/run/cache',
                        audio_mirror_root=module.PREFIX+'context-learning-audio-mirror-v1/run')
    train_config['pins'] = {c['model']+'/model.safetensors':'model',c['tasks']:'tasks',
        str(Path(train_config['encoded_audio_cache']).parent/'MANIFEST.json'):'cache',
        train_config['audio_mirror_root']+'/MANIFEST.json':'mirror'}
    (training/'config.json').write_text(json.dumps(train_config))
    source_spec = dict(training_root=str(training), selection_root=str(tmp_path/'selection'))
    for name, value in [('experiment',json.loads(Path('configs/releases/context-prob-v1/experiment.json').read_text())),
                        ('training_proof',dict(files={'config.json':module.sha(training/'config.json')})),
                        ('selection_proof',{})]:
        path=tmp_path/(name+'.json');path.write_text(json.dumps(value));source_spec[name]=str(path)
    spec_path=tmp_path/'source-spec.json';spec_path.write_text(json.dumps(source_spec))
    acceptance = dict(candidate_package=str(candidate),candidate_manifest_sha256=module.sha(candidate/'PREPARED.json'))
    accepted = dict(bindings={})
    for role in ('continuous','fresh_replay','checkpoint_frequency'):
        acceptance[role]={};accepted['bindings'][role]={}
        for kind in ('report','launch'):
            p=tmp_path/(role+'-'+kind+'.json');p.write_text(json.dumps({'container_id':'a'*64} if kind=='launch' else {}))
            acceptance[role][kind]=str(p);accepted['bindings'][role][kind+'_sha256']=module.sha(p)
    ap=tmp_path/'acceptance-spec.json';ap.write_text(json.dumps(acceptance))
    monkeypatch.setattr(module,'accept',lambda **kw:copy.deepcopy(accepted))
    monkeypatch.setattr(module,'bind',lambda **kw:copy.deepcopy(binding))
    monkeypatch.setattr(module,'verify_source',lambda *args:dict(source_policy_id='full-stage1'))
    monkeypatch.setattr(module,'verify_lock',lambda *args:dict(lock_id='fixture'))
    import multimodal_jev.local_runtime_storage as storage
    monkeypatch.setattr(storage,'check_paths',lambda *args:[])
    # Execute runtime path checks against the actual local package location.
    monkeypatch.setattr(module,'PREFIX',str(tmp_path)+'/')
    return dict(source_package=source,source_manifest_sha256=source_digest,source_specification=spec_path,
        acceptance_specification=ap,algorithm='grpo_v1',output=tmp_path/'context-rl-grpo-fixture',
        remote_root=str(tmp_path/'context-rl-grpo-fixture'))


@pytest.mark.parametrize('algorithm',['grpo_v1','rloo_v1'])
def test_package_shared_source_and_runtime_binding(inputs,algorithm):
    inputs['algorithm']=algorithm
    manifest=module.prepare(**inputs);root=inputs['output']
    c=json.loads((root/'config.json').read_text())
    assert c['source_policy_id']=='full-stage1' and c['stop_update']==634
    assert c['initial_adapter']==c['reference_adapter']==str(root/'initial/adapter')
    assert 'context-resident-switch' not in json.dumps(c)
    assert c['algorithm']==algorithm and manifest['algorithm']==algorithm
    assert validate(c)['release_id']==manifest['release_id']
    assert module.verify(root,module.sha(root/'PREPARED.json'))==manifest
    assert 'train_context_formal.py' in (root/'run.py').read_text()
    assert "if c['engineering_prefix'] is not True" in (root/'source/tools/jev/train_context_online_candidate.py').read_text()


@pytest.mark.parametrize('fault',['source','acceptance','experiment','algorithm','path'])
def test_incomplete_inputs_do_not_publish(inputs,monkeypatch,fault):
    def reject(*args,**kw):raise ValueError('incomplete source or acceptance')
    if fault=='source':monkeypatch.setattr(module,'bind',reject)
    elif fault=='acceptance':monkeypatch.setattr(module,'accept',reject)
    elif fault=='experiment':
        spec=json.loads(inputs['source_specification'].read_text());Path(spec['experiment']).write_text('{}')
    elif fault=='algorithm':inputs['algorithm']='ppo_critic_design_v1'
    else:inputs['remote_root']='/mnt/cephfs/not-hot'
    with pytest.raises(ValueError):module.prepare(**inputs)
    assert not inputs['output'].exists()


@pytest.mark.parametrize('fault',['code','config','service','extra','symlink','release'])
def test_post_publication_drift_rejected(inputs,fault):
    module.prepare(**inputs);root=inputs['output'];digest=module.sha(root/'PREPARED.json')
    paths=dict(code='source/tools/jev/train_context_formal.py',config='config.json',
               service='service-config.json',extra='extra',release='RELEASE.json')
    if fault=='symlink':(root/'link').symlink_to(root/'RELEASE.json')
    else:(root/paths[fault]).write_text('changed')
    with pytest.raises(ValueError):module.verify(root,digest)
