from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev import sfc_native_full as contract
from multimodal_jev.sfc_consumption import export
from test_sfc_consumption import source
from test_sfc_jev import load_tool


def fixture(tmp_path,monkeypatch):
    release,manifest=export(source(tmp_path),tmp_path/'native')
    monkeypatch.setattr(contract,'COUNTS',{'test_id':(2,1),'test_ood':(2,1)})
    cfg=dict(contract.RUNTIME,split='test_id',release=str(release),manifest_sha256=file_hash(release/'MANIFEST.json'),
        requests_sha256=manifest['splits']['test_id']['requests_sha256'])
    return cfg,release


def test_full_split_uses_only_requests_and_preserves_supplied_history(tmp_path,monkeypatch):
    cfg,release=fixture(tmp_path,monkeypatch)
    for path in release.glob('*.targets.jsonl'):path.unlink()
    rows=contract.select_inputs(cfg)
    assert len(rows)==2 and rows[1]['input']['messages'][1]['content']=='official history'
    assert all('supervision' not in r for r in rows)
    cfg['split']='test_ood';cfg['requests_sha256']=file_hash(release/'test_ood.requests.jsonl')
    assert len(contract.select_inputs(cfg))==2


@pytest.mark.parametrize('change',['subset','duplicate','split','group','prefix','label'])
def test_even_rehashed_malformed_population_is_rejected(tmp_path,monkeypatch,change):
    cfg,release=fixture(tmp_path,monkeypatch);path=release/'test_id.requests.jsonl'
    rows=list(map(json.loads,path.read_text().splitlines()))
    if change=='subset':rows.pop()
    if change=='duplicate':rows[1]=rows[0]
    if change=='split':rows[0]['split']='test_ood'
    if change=='group':rows[0]['source_group']='level_3_1/test_id/id_999'
    if change=='prefix':rows[1]['audio_sha256'][0]='f'*64
    if change=='label':rows[0]['target']='f(x=1)'
    path.write_bytes(b''.join(map(canonical,rows)))
    manifest=json.loads((release/'MANIFEST.json').read_text())
    manifest['splits']['test_id']['requests_sha256']=file_hash(path)
    (release/'MANIFEST.json').write_bytes(canonical(manifest))
    cfg.update(manifest_sha256=file_hash(release/'MANIFEST.json'),requests_sha256=file_hash(path))
    with pytest.raises(ValueError):contract.select_inputs(cfg)


@pytest.mark.parametrize('change',[{'split':'train_id'},{'max_new_tokens':128},{'route':'asr-text'},
                                  {'parameters_frozen':False},{'adapter':'lora'}])
def test_frozen_test_runtime_cannot_drift(tmp_path,monkeypatch,change):
    cfg,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError):contract.select_inputs({**cfg,**change})


def predictions(rows):
    return [dict(example_id=r['id'],split=r['split'],source_group=r['source_group'],request_sha256=contract.digest(r),
        audio_sha256=r['audio_sha256'],hf_vllm_prompt_ids_equal=True,input_tokens=100,raw_text='invalid task answer',
        token_ids=[1],finish_reason='stop',length_limited=False,expected_frames=[10]*len(r['audio_sha256']),
        observed_frames=[10]*len(r['audio_sha256']),prompt_sha256='a'*64,input_ids_sha256='b'*64) for r in rows]


@pytest.mark.parametrize('change',['missing','duplicate','binding','parity','frames','overflow','length','tokens'])
def test_output_contract_rejects_incomplete_or_changed_engine_evidence(tmp_path,monkeypatch,change):
    cfg,_=fixture(tmp_path,monkeypatch);rows=contract.select_inputs(cfg);ps=predictions(rows)
    contract.check_predictions(rows,ps)  # No semantic-quality threshold.
    if change=='missing':ps.pop()
    if change=='duplicate':ps[1]=ps[0]
    if change=='binding':ps[0]['request_sha256']='bad'
    if change=='parity':ps[0]['hf_vllm_prompt_ids_equal']=False
    if change=='frames':ps[0]['observed_frames']=[9]
    if change=='overflow':ps[0]['input_tokens']=3073
    if change=='length':ps[0]['length_limited']=True
    if change=='tokens':ps[0]['token_ids']=[True]
    with pytest.raises(ValueError):contract.check_predictions(rows,ps)


def test_engine_response_compares_exact_tokens_and_keeps_length_outputs():
    module=load_tool('infer_sfc_omni_full')
    answer=NS(text='unfinished',finish_reason='length',token_ids=[3]*1024)
    response=NS(finished=True,prompt_token_ids=[1,2],outputs=[answer])
    assert module.accept_response([response],[1,2]) is answer
    with pytest.raises(ValueError,match='IDs'):module.accept_response([response],[2,1])
    with pytest.raises(ValueError):module.accept_response([response,response],[1,2])
    response.finished=False
    with pytest.raises(ValueError):module.accept_response([response],[1,2])


def accepted_probe(tmp_path,cfg):
    root=tmp_path/'probe';(root/'run').mkdir(parents=True);(root/'score').mkdir()
    repository=Path(__file__).parents[1]
    names=('src/multimodal_jev/sfc_omni.py','src/multimodal_jev/sfc_audio.py')
    for name in names:
        path=root/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((repository/name).read_bytes())
    prior=dict(contract.RUNTIME,split='train_id',audio_model='/model',versions={'vllm':'pinned'})
    cfg.update(audio_model='/model',versions=prior['versions'],image='image',native_probe=str(root),
               native_probe_container_id='probe-container')
    (root/'config.json').write_bytes(canonical(prior))
    (root/'launch.json').write_bytes(canonical(dict(container_id='probe-container',image='image')))
    for path in (root/'run/terminal.json',root/'evaluation-terminal.json'):path.write_text('{"exit_code":0}')
    (root/'run/predictions.jsonl').write_text('synthetic probe predictions')
    (root/'score/metrics.json').write_text('{}')
    result=dict(status='passed',model_files_unchanged=True,weight_updates=0,requests=6,predictions=6,
        predictions_sha256=file_hash(root/'run/predictions.jsonl'))
    (root/'run/result.json').write_bytes(canonical(result))
    identity=dict(config=prior,source_files={n:file_hash(root/'source'/n) for n in names},model_files=['synthetic'])
    (root/'run/identity.json').write_bytes(canonical(identity))
    accepted=dict(container_id='probe-container',requests=6,source_groups=4,weight_updates=0,
        result_sha256=file_hash(root/'run/result.json'),score_sha256=file_hash(root/'score/metrics.json'))
    (root/'acceptance.json').write_bytes(canonical(accepted))
    cfg['native_probe_acceptance_sha256']=file_hash(root/'acceptance.json')
    return root,identity


@pytest.mark.parametrize('change',['acceptance','source','prediction','score','terminal','model'])
def test_train_probe_gate_cannot_be_bypassed_by_summary_alone(tmp_path,monkeypatch,change):
    cfg,_=fixture(tmp_path,monkeypatch);root,identity=accepted_probe(tmp_path,cfg)
    assert contract.check_probe(cfg)==identity
    if change=='acceptance':(root/'acceptance.json').write_text('{}')
    if change=='source':(root/'source/src/multimodal_jev/sfc_omni.py').write_text('changed')
    if change=='prediction':(root/'run/predictions.jsonl').write_text('changed')
    if change=='score':(root/'score/metrics.json').write_text('changed')
    if change=='terminal':(root/'evaluation-terminal.json').write_text('{"exit_code":1}')
    if change=='model':cfg['audio_model']='/other'
    with pytest.raises(ValueError):contract.check_probe(cfg)


def full_run(tmp_path,monkeypatch):
    cfg,release=fixture(tmp_path,monkeypatch);_,probe=accepted_probe(tmp_path,cfg)
    rows=contract.select_inputs(cfg);ps=predictions(rows)
    root=tmp_path/'full';(root/'run').mkdir(parents=True)
    module=load_tool('verify_sfc_omni_full');repo=Path(__file__).parents[1]
    for name in module.SOURCE_FILES:
        dest=root/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((repo/name).read_bytes())
    (root/'config.json').write_bytes(canonical(cfg))
    (root/'launch.json').write_bytes(canonical(dict(container_id='test-container',image='image',
        config_sha256=file_hash(root/'config.json'),source_files={n:file_hash(root/'source'/n) for n in module.SOURCE_FILES})))
    identity=dict(config=cfg,versions=cfg['versions'],model_files=probe['model_files'],targets_read=False,
        inputs_sha256=contract.digest(rows),native_probe_acceptance_sha256=cfg['native_probe_acceptance_sha256'],
        source_files={n:file_hash(root/'source'/n) for n in module.SOURCE_FILES})
    (root/'run/identity.json').write_bytes(canonical(identity))
    (root/'run/predictions.jsonl').write_bytes(b''.join(map(canonical,ps)))
    (root/'run/selection.json').write_bytes(canonical(dict(ids=[r['id'] for r in rows])))
    result=dict(status='passed',weight_updates=0,model_files_unchanged=True,targets_read=False,
        identity_sha256=file_hash(root/'run/identity.json'),predictions_sha256=file_hash(root/'run/predictions.jsonl'),
        requests=2,predictions=2,length_limited=0)
    (root/'run/result.json').write_bytes(canonical(result));(root/'run/terminal.json').write_text('{"exit_code":0}')
    def inspect(cid):
        return dict(Id=cid,Image='image',State=dict(Status='exited',ExitCode=0),
            HostConfig=dict(Devices=[dict(PathOnHost='/dev/davinci6'),dict(PathOnHost='/dev/davinci7')]))
    monkeypatch.setattr(module,'inspect',inspect)
    return module,root,cfg,release


def test_independent_acceptance_never_needs_targets(tmp_path,monkeypatch):
    module,root,cfg,release=full_run(tmp_path,monkeypatch)
    (release/'test_id.targets.jsonl').unlink()
    assert len(module.accept(root)[1])==2
    result=json.loads((root/'run/result.json').read_text());result['requests']=1
    (root/'run/result.json').write_bytes(canonical(result))
    with pytest.raises(ValueError,match='coverage'):
        module.score(root,Path('/unused-scorer'),'unused')
    assert not (root/'score-input.jsonl').exists()


def test_complete_accepted_outputs_bridge_to_official_record_format(tmp_path,monkeypatch):
    module,root,cfg,release=full_run(tmp_path,monkeypatch)
    def evaluate(manifest,predictions,scorer,out,scorer_hash):
        # Verify scoring receives the original target-bearing rows only here.
        rows=list(map(json.loads,manifest.read_text().splitlines()))
        assert len(rows)==2 and all('supervision' in r for r in rows)
        assert rows[1]['input']['messages'][1]['content']=='official history'
        assert scorer_hash=='pinned'
        out.mkdir();report=dict(strict_metrics={},strict_conversation_metrics={})
        (out/'metrics.json').write_bytes(canonical(report));return report
    monkeypatch.setattr(module,'evaluate',evaluate)
    receipt=module.score(root,Path('/scorer'),'pinned')
    assert receipt['requests']==2 and receipt['source_groups']==1
    with pytest.raises(FileExistsError):module.score(root,Path('/scorer'),'pinned')


@pytest.mark.parametrize('change',['running','wrong_id','wrong_image','unreserved_cards'])
def test_acceptance_checks_actual_container_and_reserved_cards(tmp_path,monkeypatch,change):
    module,root,cfg,release=full_run(tmp_path,monkeypatch);original=module.inspect
    def inspect(cid):
        item=original(cid)
        if change=='running':item['State']['Status']='running'
        if change=='wrong_id':item['Id']='other'
        if change=='wrong_image':item['Image']='other'
        if change=='unreserved_cards':item['HostConfig']['Devices']=[dict(PathOnHost='/dev/davinci0')]
        return item
    monkeypatch.setattr(module,'inspect',inspect)
    with pytest.raises(ValueError):module.accept(root)


def test_launch_pins_reject_consistent_posthoc_source_rewrite(tmp_path,monkeypatch):
    module,root,_,_=full_run(tmp_path,monkeypatch)
    name=module.SOURCE_FILES[0];(root/'source'/name).write_text('changed')
    identity=json.loads((root/'run/identity.json').read_text())
    identity['source_files'][name]=file_hash(root/'source'/name)
    (root/'run/identity.json').write_bytes(canonical(identity))
    result=json.loads((root/'run/result.json').read_text())
    result['identity_sha256']=file_hash(root/'run/identity.json')
    (root/'run/result.json').write_bytes(canonical(result))
    with pytest.raises(ValueError,match='source identity'):module.accept(root)


def test_actual_runner_control_flow_preserves_audio_and_raw_failures(tmp_path,monkeypatch):
    import sys
    cfg,release=fixture(tmp_path,monkeypatch);rows=contract.select_inputs(cfg)
    module=load_tool('infer_sfc_omni_full');out=tmp_path/'run';out.mkdir()
    cfg.update(audio_model='/model',versions={k:'pinned' for k in
        ('torch','torch-npu','transformers','vllm','vllm-ascend')},native_probe_acceptance_sha256='a'*64)
    monkeypatch.setattr(module,'select_inputs',lambda _:rows)
    monkeypatch.setattr(module,'check_probe',lambda _:dict(model_files=['model']))
    verified=[]
    monkeypatch.setattr(module,'verify_assets',lambda _:verified.append(True) or dict(files=['model']))
    monkeypatch.setattr(module.importlib.metadata,'version',lambda _:'pinned')
    actual_hash=module.file_hash
    audio_hashes={p:h for r in rows for p,h in zip(r['input']['audios'],r['audio_sha256'])}
    monkeypatch.setattr(module,'file_hash',lambda p:audio_hashes[str(p)] if str(p) in audio_hashes else actual_hash(p))
    # A synthetic >30-second waveform is retained at full length, not excluded.
    waves={p:[0]*(640000 if p.endswith('_1.wav') else 320000) for p in audio_hashes}
    monkeypatch.setattr(module,'load_sfc_audio',lambda p:(waves[p],dict(synthetic=True)))
    class Tensor:
        def __init__(self,data):self.data=data
        def __getitem__(self,i):return Tensor(self.data[i])
        def tolist(self):return self.data
        def sum(self,axis):return self
    prompts=[]
    class Processor:
        feature_extractor=NS(hop_length=160)
        @classmethod
        def from_pretrained(cls,*a,**k):return cls()
        def apply_chat_template(self,messages,**kwargs):
            prompts.append(messages);return 'synthetic prompt'
        def __call__(self,**kwargs):
            lengths=[len(w) for w in kwargs['audio']]
            return dict(input_ids=Tensor([[1,2,3]]),input_features=object(),
                feature_attention_mask=Tensor(module.expected_mask_frames(lengths,160)))
    emitted=[];shutdown=[]
    class Engine:
        def __init__(self,**kwargs):
            assert kwargs['tensor_parallel_size']==2 and not kwargs['enable_prefix_caching']
            assert kwargs['limit_mm_per_prompt']==dict(audio=2,image=0,video=0)
            self.llm_engine=NS(engine_core=NS(shutdown=lambda **k:shutdown.append(True)))
        def generate(self,requests,params,**kwargs):
            emitted.append(requests[0]);return [NS(finished=True,prompt_token_ids=[1,2,3],
                outputs=[NS(text='INVALID RAW OUTPUT',token_ids=[42]*1024,finish_reason='length')])]
    monkeypatch.setitem(sys.modules,'transformers',NS(Qwen3OmniMoeProcessor=Processor))
    monkeypatch.setitem(sys.modules,'transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe',
        NS(Qwen3OmniMoeProcessorKwargs=NS(_defaults={'audio_kwargs':{'truncation':False}})))
    monkeypatch.setitem(sys.modules,'vllm',NS(LLM=Engine,SamplingParams=lambda **k:NS(**k)))
    monkeypatch.setitem(sys.modules,'torch',NS(isfinite=lambda _:NS(all=lambda:True)))
    module.run(cfg,out)
    ps=list(map(json.loads,(out/'predictions.jsonl').read_text().splitlines()))
    assert len(ps)==2 and all(p['raw_text']=='INVALID RAW OUTPUT' and p['length_limited'] for p in ps)
    assert len(emitted[1]['multi_modal_data']['audio'])==2
    assert emitted[1]['multi_modal_data']['audio'][0][0] is waves[rows[1]['input']['audios'][0]]
    assert prompts[1][2]['content']=='official history'
    assert len(verified)==2 and shutdown==[True]
    assert json.loads((out/'result.json').read_text())['targets_read'] is False
