import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_jev_runtime import INSTRUCTION
from multimodal_jev.sfc_native_full import RUNTIME
from multimodal_jev import sfc_jev_shard_runtime as contract
from multimodal_jev.sfc_jev_shards import make_plan,load_shard
from test_sfc_jev_shards import fixture as release_fixture
from test_sfc_jev import load_tool

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))


def fixture(tmp_path):
    release,_=release_fixture(tmp_path);plan=make_plan(release,'test_id',12)
    plan_path=tmp_path/'plan.json';plan_path.write_bytes(canonical(plan))
    rows=load_shard(plan,'shard-0000');audit=tmp_path/'audit';run=audit/'run';run.mkdir(parents=True)
    names=('src/multimodal_jev/sfc_jev_runtime.py','src/multimodal_jev/sfc_audio.py')
    for name in names:
        path=audit/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((Path(__file__).parents[1]/name).read_bytes())
    identity=dict(model='/model',release=str(release),manifest_sha256=plan['manifest_sha256'],
        targets_read=False,context_limit=4096,output_budget=1024,processor_assets={},
        source_files={n:file_hash(audit/'source'/n) for n in names})
    (run/'identity.json').write_bytes(canonical(identity));(run/'terminal.json').write_text('{"exit_code":0}')
    checks=[dict(id=r['id'],decision_id=d['id'],context_overflow=False,input_tokens=3,
        input_ids_sha256=contract.digest([1,2,3]),prompt_sha256=hashlib.sha256(b'prompt').hexdigest(),
        expected_frames=[4000]*len(r['audio_sha256']),observed_frames=[4000]*len(r['audio_sha256']))
        for r in rows for d in r['input']['decisions']]
    path=run/'test_id.context.jsonl';path.write_bytes(b''.join(map(canonical,checks)))
    result=dict(status='passed',all_inputs_fit=True,identity_sha256=file_hash(run/'identity.json'),
        splits={'test_id':dict(records_sha256=file_hash(path))})
    (run/'result.json').write_bytes(canonical(result))
    acc=tmp_path/'context-acceptance.json'
    acc.write_bytes(canonical(dict(result_sha256=file_hash(run/'result.json'),container_id='cpu-container',all_inputs_fit=True)))
    cfg=dict(RUNTIME,prompt_id='parent',instruction_sha256=hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        shard_plan=str(plan_path),shard_plan_sha256=file_hash(plan_path),shard_id='shard-0000',split='test_id',
        context_audit=str(audit),context_result_sha256=file_hash(run/'result.json'),context_acceptance=str(acc),
        context_acceptance_sha256=file_hash(acc),context_container_id='cpu-container',audio_model='/model')
    return cfg,rows,{(r['id'],r['decision_id']):r for r in checks}


def test_parent_shard_matches_every_context_decision_without_targets(tmp_path):
    cfg,rows,checks=fixture(tmp_path)
    actual,plan,records=contract.select_inputs(cfg)
    assert actual==rows and records==checks and plan['decisions']==24
    assert len(checks)==12 and not list(Path(plan['release']).glob('*.targets.jsonl'))


@pytest.mark.parametrize('change',['candidate','prompt','budget','split','plan_hash','context_hash','acceptance','source','candidate_context'])
def test_changed_runtime_or_evidence_is_rejected(tmp_path,change):
    cfg,_,_=fixture(tmp_path)
    if change=='candidate':cfg['prompt_id']='p1'
    if change=='prompt':cfg['instruction_sha256']='bad'
    if change=='budget':cfg['max_new_tokens']=128
    if change=='split':cfg['split']='test_ood'
    if change=='plan_hash':cfg['shard_plan_sha256']='bad'
    if change=='context_hash':cfg['context_result_sha256']='bad'
    if change=='acceptance':cfg['context_acceptance_sha256']='bad'
    if change=='source':(Path(cfg['context_audit'])/'source/src/multimodal_jev/sfc_audio.py').write_text('changed')
    if change=='candidate_context':
        p=Path(cfg['context_acceptance']);value=json.loads(p.read_text());value['prompt']={'candidate_id':'p1'}
        p.write_bytes(canonical(value));cfg['context_acceptance_sha256']=file_hash(p)
    with pytest.raises(ValueError):contract.select_inputs(cfg)


def test_missing_context_decision_rejected_even_with_updated_receipts(tmp_path):
    cfg,_,_=fixture(tmp_path);run=Path(cfg['context_audit'])/'run';path=run/'test_id.context.jsonl'
    path.write_bytes(b''.join(path.read_bytes().splitlines(keepends=True)[:-1]))
    result=json.loads((run/'result.json').read_text());result['splits']['test_id']['records_sha256']=file_hash(path)
    (run/'result.json').write_bytes(canonical(result));cfg['context_result_sha256']=file_hash(run/'result.json')
    acc=Path(cfg['context_acceptance']);data=json.loads(acc.read_text());data['result_sha256']=cfg['context_result_sha256']
    acc.write_bytes(canonical(data));cfg['context_acceptance_sha256']=file_hash(acc)
    with pytest.raises(ValueError,match='every shard decision'):contract.select_inputs(cfg)


def predictions(rows,checks):
    return [dict(example_id=r['id'],decision_id=d['id'],request_sha256=contract.digest(r),source_group=r['source_group'],
        split=r['split'],raw_text='false',token_ids=[4],finish_reason='stop',length_limited=False,
        audio_sha256=r['audio_sha256'],hf_vllm_prompt_ids_equal=True,
        **{k:checks[r['id'],d['id']][k] for k in ('input_tokens','input_ids_sha256','prompt_sha256','expected_frames','observed_frames')})
        for r in rows for d in r['input']['decisions']]


@pytest.mark.parametrize('change',['drop_calls','duplicate','order','binding','parity','frames','tokens','length'])
def test_predictions_require_all_decisions_and_exact_engine_evidence(tmp_path,change):
    _,rows,checks=fixture(tmp_path);ps=predictions(rows,checks);contract.check_predictions(rows,checks,ps)
    if change=='drop_calls':ps=[p for p in ps if not p['decision_id'].endswith('calls')]
    if change=='duplicate':ps[1]=ps[0]
    if change=='order':ps.reverse()
    if change=='binding':ps[0]['request_sha256']='bad'
    if change=='parity':ps[0]['hf_vllm_prompt_ids_equal']=False
    if change=='frames':ps[0]['observed_frames']=[10]
    if change=='tokens':ps[0]['token_ids']=[True]
    if change=='length':ps[0]['length_limited']=True
    with pytest.raises(ValueError):contract.check_predictions(rows,checks,ps)


@pytest.mark.parametrize('bad_engine_ids',[False,True])
def test_runner_generates_calls_even_after_false_noul_and_keeps_long_audio(tmp_path,monkeypatch,bad_engine_ids):
    cfg,rows,checks=fixture(tmp_path);module=load_tool('infer_sfc_jev_omni_shard');out=tmp_path/'out';out.mkdir()
    cfg['versions']={k:'pinned' for k in ('torch','torch-npu','transformers','vllm','vllm-ascend')}
    monkeypatch.setattr(module,'check_probe',lambda _:dict(model_files=['model']))
    verified=[];monkeypatch.setattr(module,'verify_assets',lambda _:verified.append(True) or dict(files=['model']))
    monkeypatch.setattr(module.importlib.metadata,'version',lambda _:'pinned')
    original_hash=module.file_hash;audios={p:h for r in rows for p,h in zip(r['input']['context']['audios'],r['audio_sha256'])}
    monkeypatch.setattr(module,'file_hash',lambda p:audios[str(p)] if str(p) in audios else original_hash(p))
    waves={p:[0]*640000 for p in audios};monkeypatch.setattr(module,'load_sfc_audio',lambda p:(waves[p],dict(synthetic=True)))
    class Tensor:
        def __init__(self,data):self.data=data
        def __getitem__(self,i):return Tensor(self.data[i])
        def tolist(self):return self.data
        def sum(self,axis):return self
    rendered=[]
    class Processor:
        feature_extractor=NS(hop_length=160)
        @classmethod
        def from_pretrained(cls,*a,**k):return cls()
        def apply_chat_template(self,messages,**kwargs):rendered.append(messages);return 'prompt'
        def __call__(self,**kwargs):return dict(input_ids=Tensor([[1,2,3]]),input_features=object(),
            feature_attention_mask=Tensor([4000]*len(kwargs['audio'])))
    emitted=[];shutdown=[]
    class Engine:
        def __init__(self,**kwargs):
            assert kwargs['tensor_parallel_size']==2 and not kwargs['enable_prefix_caching']
            self.llm_engine=NS(engine_core=NS(shutdown=lambda **k:shutdown.append(True)))
        def generate(self,requests,params,**kwargs):
            emitted.append(requests[0]);return [NS(finished=True,prompt_token_ids=[3,2,1] if bad_engine_ids else [1,2,3],
                outputs=[NS(text='false',token_ids=[4],finish_reason='stop')])]
    monkeypatch.setitem(sys.modules,'transformers',NS(Qwen3OmniMoeProcessor=Processor))
    monkeypatch.setitem(sys.modules,'transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe',
        NS(Qwen3OmniMoeProcessorKwargs=NS(_defaults={'audio_kwargs':{'truncation':False}})))
    monkeypatch.setitem(sys.modules,'vllm',NS(LLM=Engine,SamplingParams=lambda **k:NS(**k)))
    monkeypatch.setitem(sys.modules,'torch',NS(isfinite=lambda _:NS(all=lambda:True)))
    if bad_engine_ids:
        with pytest.raises(ValueError,match='IDs differ'):module.run(cfg,out)
        assert shutdown==[True] and not (out/'result.json').exists()
    else:
        module.run(cfg,out)
        ps=list(map(json.loads,(out/'predictions.jsonl').read_text().splitlines()))
        assert len(ps)==12 and len(emitted)==12 and all(p['raw_text']=='false' for p in ps)
        assert sum(p['decision_id'].endswith('.calls') for p in ps)==4
        assert len(emitted[-1]['multi_modal_data']['audio'])==2
        assert rendered[-1][2]['content']=='supplied prior assistant'
        assert len(verified)==2 and shutdown==[True]


def probe_fixture(tmp_path,cfg):
    root=tmp_path/'probe';run=root/'run';run.mkdir(parents=True);(root/'score').mkdir()
    prior=dict(RUNTIME,split='train_id',audio_model=cfg['audio_model'],versions={'vllm':'pinned'})
    cfg.update(versions=prior['versions'],jev_probe=str(root),jev_probe_container_id='probe-cid',image='image')
    names=('src/multimodal_jev/sfc_jev_runtime.py','src/multimodal_jev/sfc_audio.py')
    for name in names:
        p=root/'source'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((Path(__file__).parents[1]/name).read_bytes())
    (root/'config.json').write_bytes(canonical(prior));(run/'predictions.jsonl').write_text('synthetic')
    (root/'score/metrics.json').write_text('{}')
    for p in (run/'terminal.json',root/'evaluation-terminal.json'):p.write_text('{"exit_code":0}')
    source_files={n:file_hash(root/'source'/n) for n in names}
    (root/'launch.json').write_bytes(canonical(dict(container_id='probe-cid',image='image')))
    (root/'preparation.json').write_bytes(canonical(dict(image='image',config_sha256=file_hash(root/'config.json'),inference_files=source_files)))
    (root/'acceptance.json').write_bytes(canonical(dict(bundles=6,decisions=78,source_groups=4,weight_updates=0,
        metrics_sha256=file_hash(root/'score/metrics.json'))))
    cfg['jev_probe_acceptance_sha256']=file_hash(root/'acceptance.json')
    result=dict(status='passed',weight_updates=0,model_files_unchanged=True,bundles=6,decisions=78,predictions=78,
                predictions_sha256=file_hash(run/'predictions.jsonl'))
    (run/'result.json').write_bytes(canonical(result))
    identity=dict(config=prior,source_files=source_files,model_files=['synthetic'])
    (run/'identity.json').write_bytes(canonical(identity))
    return root,identity


@pytest.mark.parametrize('change',['acceptance','prediction','source','terminal','container','budget'])
def test_probe_gate_uses_complete_result_and_fixed_adapter(tmp_path,change):
    cfg,_,_=fixture(tmp_path);root,identity=probe_fixture(tmp_path,cfg)
    assert contract.check_probe(cfg)==identity
    if change=='acceptance':(root/'acceptance.json').write_text('{}')
    if change=='prediction':(root/'run/predictions.jsonl').write_text('changed')
    if change=='source':(root/'source/src/multimodal_jev/sfc_jev_runtime.py').write_text('changed')
    if change=='terminal':(root/'evaluation-terminal.json').write_text('{"exit_code":1}')
    if change=='container':cfg['jev_probe_container_id']='other'
    if change=='budget':
        prior=json.loads((root/'config.json').read_text());prior['max_new_tokens']=128
        (root/'config.json').write_bytes(canonical(prior))
    with pytest.raises(ValueError):contract.check_probe(cfg)


def completed_shard(tmp_path,monkeypatch):
    cfg,rows,checks=fixture(tmp_path);_,probe=probe_fixture(tmp_path,cfg)
    module=load_tool('verify_sfc_jev_shard');root=tmp_path/'shard';run=root/'run';run.mkdir(parents=True)
    for name in module.SOURCE_FILES:
        p=root/'source'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((Path(__file__).parents[1]/name).read_bytes())
    (root/'config.json').write_bytes(canonical(cfg));ps=predictions(rows,checks)
    files={n:file_hash(root/'source'/n) for n in module.SOURCE_FILES}
    identity=dict(config=cfg,model_files=probe['model_files'],versions=cfg['versions'],inputs_sha256=contract.digest(rows),
        context_result_sha256=cfg['context_result_sha256'],targets_read=False,source_files=files)
    (run/'identity.json').write_bytes(canonical(identity))
    (run/'predictions.jsonl').write_bytes(b''.join(map(canonical,ps)))
    (run/'selection.json').write_bytes(canonical(dict(ids=[r['id'] for r in rows],decisions=[[r['id'],d['id']] for r in rows for d in r['input']['decisions']])))
    (run/'terminal.json').write_text('{"exit_code":0}')
    result=dict(status='passed',weight_updates=0,model_files_unchanged=True,targets_read=False,
        identity_sha256=file_hash(run/'identity.json'),predictions_sha256=file_hash(run/'predictions.jsonl'),
        bundles=2,decisions=12,predictions=12,length_limited=0)
    (run/'result.json').write_bytes(canonical(result))
    (root/'launch.json').write_bytes(canonical(dict(container_id='shard-cid',image='image',
        config_sha256=file_hash(root/'config.json'),source_files=files)))
    calls=[]
    monkeypatch.setattr(module,'accept_probe',lambda p:calls.append('probe'))
    monkeypatch.setattr(module,'verify_context',lambda p:calls.append('context') or dict(
        container_id=cfg['context_container_id'],result_sha256=cfg['context_result_sha256'],all_inputs_fit=True))
    monkeypatch.setattr(module,'inspect',lambda cid:dict(Id=cid,Image='image',State=dict(Status='exited',ExitCode=0),
        HostConfig=dict(Devices=[dict(PathOnHost='/dev/davinci6'),dict(PathOnHost='/dev/davinci7')])) )
    return module,root,calls


def test_independent_shard_acceptance_revalidates_dependencies_and_keeps_raw_errors(tmp_path,monkeypatch):
    module,root,calls=completed_shard(tmp_path,monkeypatch)
    receipt=module.accept(root)
    assert calls==['probe','context'] and receipt['decisions']==12 and receipt['targets_read'] is False
    # All raw outputs are false, including calls. Engineering acceptance cannot
    # silently impose a task-quality threshold or repair those outputs.
    assert all(json.loads(line)['raw_text']=='false' for line in (root/'run/predictions.jsonl').read_text().splitlines())


@pytest.mark.parametrize('change',['container','chips','source','prediction','missing_output','summary','failed_probe'])
def test_independent_shard_acceptance_rejects_incomplete_or_mutated_run(tmp_path,monkeypatch,change):
    module,root,_=completed_shard(tmp_path,monkeypatch)
    if change in ('container','chips'):
        original=module.inspect
        def inspect(cid):
            item=original(cid)
            if change=='container':item['State']['Status']='running'
            else:item['HostConfig']['Devices']=[dict(PathOnHost='/dev/davinci0')]
            return item
        monkeypatch.setattr(module,'inspect',inspect)
    if change=='source':(root/'source'/module.SOURCE_FILES[0]).write_text('changed')
    if change=='prediction':(root/'run/predictions.jsonl').write_text('changed')
    if change=='missing_output':
        p=root/'run/predictions.jsonl';p.write_bytes(b''.join(p.read_bytes().splitlines(keepends=True)[:-1]))
        result=json.loads((root/'run/result.json').read_text());result['predictions_sha256']=file_hash(p)
        (root/'run/result.json').write_bytes(canonical(result))
    if change=='summary':
        result=json.loads((root/'run/result.json').read_text());result['decisions']=11
        (root/'run/result.json').write_bytes(canonical(result))
    if change=='failed_probe':
        def fail(_):raise ValueError('failed probe')
        monkeypatch.setattr(module,'accept_probe',fail)
    with pytest.raises(ValueError):module.accept(root)
