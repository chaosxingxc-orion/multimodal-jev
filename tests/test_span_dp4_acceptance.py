import importlib.util
import json
from pathlib import Path
import subprocess
import pytest


@pytest.fixture
def queue(monkeypatch):
    path=Path(__file__).resolve().parents[1]/'tools/jev/queue_span_dp4_acceptance.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec=importlib.util.spec_from_file_location('acceptance_queue',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.mark.parametrize('fault',[None,'running','queue_failure','wrong_root','wrong_id','image','failed','paused'])
def test_only_successful_exact_resume_allows_acceptance(queue,tmp_path,monkeypatch,fault):
    q=tmp_path/'queue';r=tmp_path/'resume'
    write(q/'run/terminal.json',dict(exit_code=int(fault=='queue_failure')))
    write(q/'run/dispatched.json',dict(root='wrong' if fault=='wrong_root' else str(r),container_id='parent'))
    write(r/'launch.json',dict(container_id='wrong' if fault=='wrong_id' else 'parent',launch_exit_code=0))
    monkeypatch.setattr(queue,'inspect',lambda ids:{'parent':dict(Image='wrong' if fault=='image' else 'image',
        State=dict(Status='running' if fault=='running' else 'paused' if fault=='paused' else 'exited',ExitCode=int(fault=='failed')))})
    config=dict(resume_queue=str(q),resume_root=str(r),image='image',pins={})
    if fault in (None,'running'):assert queue.predecessor(config)==('parent' if fault is None else None)
    else:
        with pytest.raises(ValueError):queue.predecessor(config)


@pytest.mark.parametrize('fault',[None,'not_ready','busy','freeze','create','start','timeout'])
def test_no_gate_start_before_verified_resume_and_receipt(queue,tmp_path,monkeypatch,fault):
    suite=tmp_path/'suite';suite.mkdir();output=tmp_path/'out';output.mkdir()
    write(suite/'prepared.json',dict(create_argv=['docker','create','gate'],image='image',physical_devices=[4]))
    (suite/'source-manifest.json').write_text('{}')
    monkeypatch.setattr(queue,'predecessor',lambda c:None if fault=='not_ready' else 'parent')
    def devices(image):
        if fault=='busy':raise ValueError('busy')
    monkeypatch.setattr(queue,'check_devices',devices)
    def freeze(c):
        if fault=='freeze':raise ValueError('incomplete rank')
        (suite/'config.json').write_text('{}')
    monkeypatch.setattr(queue,'freeze_inputs',freeze)
    child='b'*64;calls=[]
    def remote(argv):
        calls.append(argv)
        if argv[1]=='start':assert json.loads((suite/'launch.json').read_text())['container_id']==child
        if fault=='timeout':raise subprocess.TimeoutExpired(argv,60)
        return subprocess.CompletedProcess(argv,int(fault==argv[1]),child+'\n','')
    monkeypatch.setattr(queue,'remote',remote)
    monkeypatch.setattr(queue,'inspect',lambda ids:{child:dict(Image='image',State=dict(Status='running'))})
    c=dict(suite_root=str(suite),lock_directory=str(tmp_path),image='image')
    if fault:
        with pytest.raises((ValueError,subprocess.SubprocessError)):queue.dispatch(c,output)
        assert not (output/'dispatched.json').exists()
        if fault in ('not_ready','busy','freeze'):assert not calls
        if fault=='timeout':assert (suite/'create-attempt.json').exists()
    else:
        queue.dispatch(c,output);assert [a[1] for a in calls]==['create','start']
        assert (output/'dispatched.json').exists()


@pytest.mark.parametrize('fault',[None,'rank_failure','wrong_step','wrong_adapter','changed_asset','claimed'])
def test_freeze_binds_all_four_rank_outputs_before_publication(queue,tmp_path,monkeypatch,fault):
    from multimodal_jev.baseline import file_hash
    import multimodal_jev.policy_identity as identity
    root=tmp_path/'resume';suite=tmp_path/'suite';suite.mkdir()
    source=suite/'source';source.mkdir();(source/'code.py').write_text('pass\n')
    write(suite/'source-manifest.json',{'code.py':file_hash(source/'code.py')})
    audio_identity=tmp_path/'identity.json';audio_identity.write_text('{}')
    write(suite/'prepared.json',dict(identity=str(audio_identity)))
    asset=tmp_path/'asset';asset.write_text('fixed')
    write(root/'config.json',dict(model_path='model',runtime_profile='profile',pins={str(asset):file_hash(asset)}))
    write(root/'source-manifest.json',{})
    write(root/'launch.json',dict(config_sha256=file_hash(root/'config.json'),source_manifest_sha256=file_hash(root/'source-manifest.json')))
    adapter=root/'run/adapter';adapter.mkdir(parents=True);(adapter/'adapter_model.safetensors').write_text('test-only')
    monkeypatch.setattr(identity,'checkpoint_identity',lambda p:('adapter',{'adapter_model.safetensors':file_hash(adapter/'adapter_model.safetensors')}))
    for rank in range(4):
        out=root/'run'/f'rank-{rank}'
        write(out/'terminal.json',dict(exit_code=int(fault=='rank_failure' and rank==3)))
        write(out/'result.json',dict(status='passed',optimizer_updates=1 if fault=='wrong_step' and rank==3 else 2,
            rank=rank,world_size=4,adapter_id='bad' if fault=='wrong_adapter' and rank==3 else 'adapter'))
        for name in ('optimizer-local.pt','restored-local.pt','optimizer-resume.json','selection.json','after.json'):
            (out/name).write_text('test-only')
    if fault=='changed_asset':asset.write_text('changed')
    if fault=='claimed':(suite/'run').mkdir()
    c=dict(resume_root=str(root),suite_root=str(suite))
    if fault:
        with pytest.raises(ValueError):queue.freeze_inputs(c)
        assert not (suite/'config.json').exists()
    else:
        result=queue.freeze_inputs(c)
        assert result['adapter']==str(adapter)
        for rank in range(4):
            assert str(root/'run'/f'rank-{rank}'/'optimizer-local.pt') in result['pins']
        assert result==json.loads((suite/'config.json').read_text())
