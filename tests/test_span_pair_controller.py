import importlib.util
import json
from pathlib import Path
import subprocess
import pytest
from multimodal_jev.baseline import file_hash


@pytest.mark.parametrize('fault',[None,'training','hf','runtime','gate','snapshot','device'])
def test_controller_stops_before_next_branch_if_any_stage_fails(tmp_path,monkeypatch,fault):
    path=Path(__file__).resolve().parents[1]/'tools/jev/run_span_pair_pilot.py'
    monkeypatch.syspath_prepend(str(path.parent));spec=importlib.util.spec_from_file_location('pair_controller',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=tmp_path/'pilot';root.mkdir();output=root/'run';output.mkdir()
    def write(p,v):p.write_text(json.dumps(v))
    source=root/'source';source.mkdir();(source/'file.py').write_text('fixed')
    write(root/'source-manifest.json',dict(files={'file.py':file_hash(source/'file.py')}))
    write(root/'launch.json',dict(launch_exit_code=0,source_manifest_sha256=file_hash(root/'source-manifest.json'),image='fixed-image',physical_devices=[5]))
    transfer=root/'transfer.json';write(transfer,{})
    admission=root/'initial';admission.mkdir()
    write(admission/'config.json',dict(transfer_protocol=str(transfer),request_ids=['fixed'],pins={},adapter='old',adapter_id='old',hf_run='old'))
    for name,beta in [('pair0',0.),('pair05',.5)]:
        write(root/(name+'.json'),dict(protocol='protocol',pair_weight=beta,physical_devices=[4] if fault=='device' else [5],pins={}))
    if fault=='snapshot':(source/'file.py').write_text('changed')
    monkeypatch.setattr(module,'pilot_rows',lambda c:(dict(runtime_admission=str(admission)),{},[],None))
    monkeypatch.setattr(module,'checkpoint_identity',lambda p:('new-adapter',{}))
    calls=[]
    def child(argv,**kwargs):
        script=Path(argv[1]).name;dest=Path(argv[argv.index('--output')+1]);calls.append((script,dest))
        stage={'train_span_pair_pilot.py':'training','infer_span_rule_baseline.py':'hf','check_span_rule_runtime.py':'runtime'}[script]
        if fault==stage:raise subprocess.CalledProcessError(1,argv)
        dest.mkdir()
        if stage=='training':
            adapter=dest/'policy-1/adapter';adapter.mkdir(parents=True);(adapter/'weights').write_text('weights')
        else:
            write(dest/'result.json',{});write(dest/'terminal.json',dict(exit_code=0));(dest/'records.jsonl').write_text('')
    monkeypatch.setattr(module.subprocess,'run',child)
    audits=[]
    def verify(*args):
        audits.append(args)
        return dict(passed=True,accepted_for_reuse=fault!='gate',adapter_id='new-adapter')
    monkeypatch.setattr(module,'verify',verify)
    if fault:
        with pytest.raises((ValueError,subprocess.CalledProcessError)):module.run(root,output)
        assert not (output/'READY.json').exists()
        assert all('pair05' not in str(dest) for _,dest in calls)
    else:
        module.run(root,output)
        assert len(calls)==6 and len(audits)==4
        assert [x['branch'] for x in json.loads((output/'READY.json').read_text())['branches']]==['pair0','pair05']
        c=json.loads((output/'pair05-runtime/config.json').read_text())
        assert c['request_ids']==['fixed'] and c['engineering_pair_weight']==.5
        assert str(output/'pair05-hf/run/records.jsonl') in c['pins']
