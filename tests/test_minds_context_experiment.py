import json
from pathlib import Path
import pytest
from tools.reports import run_minds_context_experiment as module


@pytest.fixture
def spec(tmp_path,monkeypatch):
    def write(p,value):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
        return dict(path=str(p),sha256=module.sha(p))
    inputs=tmp_path/'inputs';artifacts={}
    for key,name in [(n.replace('.','_'),n) for n in ('MANIFEST.json','definitions.json','inputs.jsonl','conversion.jsonl')]:
        artifacts[key]=write(inputs/name,{})
    contract=write(tmp_path/'contract.json',dict(schema='minds-native-descriptor-scoring-v1',threshold=.5,
        target_tuning=False,artifacts=artifacts))
    processor=tmp_path/'processor';pins={}
    for name in ('MANIFEST.json','inputs.jsonl','audit.json'):
        a=write(processor/name,{});pins[a['path']]=a['sha256']
    source=tmp_path/'source';write(source/'PACKAGE.json',{})
    from tools.jev import prepare_fsc_context_source
    monkeypatch.setattr(prepare_fsc_context_source,'verify',lambda *a:{})
    return dict(inputs=str(inputs),processor_receipts_root=str(processor),input_config=str(processor/'audit.json'),
        source_package=str(source),scoring_contract=contract,pins=pins,
        remote_root=module.PREFIX+'minds-context-pair-fixture'),tmp_path/'output'


def test_default_plan_no_source_selection_or_remote(spec,monkeypatch):
    a,out=spec
    monkeypatch.setattr(module,'invoke',lambda *a:(_ for _ in ()).throw(AssertionError('no remote')))
    assert module.run(a,out)['status']=='validated_MINDS_plan_no_dispatch'
    assert not out.exists()


@pytest.mark.parametrize('failure',[None,'package','inference','collect','score'])
def test_independent_collection_before_scoring_and_no_retry(spec,monkeypatch,failure):
    a,out=spec;calls=[]
    def invoke(name,args,log):
        phase=log.stem;calls.append(phase)
        if phase==failure:raise ValueError('actual phase failed')
        if phase=='package':
            (out/'runtime-package').mkdir();(out/'runtime-package/PACKAGE.json').write_text('{}')
        elif phase=='score':module.write(out/'scores.json',dict(target_tuning=False,recordings=8168,decisions=14))
    monkeypatch.setattr(module,'invoke',invoke)
    if failure:
        with pytest.raises(ValueError):module.run(a,out,launch=True)
        assert module.read(out/'status.json')['phase']=='stopped_preserve_handles'
        assert calls==['package','inference','collect','score'][:['package','inference','collect','score'].index(failure)+1]
    else:
        assert module.run(a,out,launch=True)['phase']=='frozen_MINDS_collected_and_scored'
        assert calls==['package','inference','collect','score']
    before=list(calls)
    with pytest.raises(FileExistsError):module.run(a,out,launch=True)
    assert calls==before


@pytest.mark.parametrize('fault',['contract','inputs','processor','remote'])
def test_identity_drift_fails_before_dispatch(spec,fault):
    a,out=spec
    if fault=='contract':Path(a['scoring_contract']['path']).write_text('{}')
    elif fault=='inputs':(Path(a['inputs'])/'definitions.json').write_text('changed')
    elif fault=='processor':Path(a['input_config']).write_text('changed')
    else:a['remote_root']='/invalid'
    with pytest.raises(ValueError):module.run(a,out,launch=True)
    assert not out.exists()
