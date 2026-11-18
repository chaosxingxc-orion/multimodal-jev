import json
from pathlib import Path
import pytest
from tools.reports import context_acoustic_transfer_parity as a
from tools.reports import context_kv_training_operations as kv
from test_context_transfer_parity_operations import config
from test_prepare_context_transfer_parity import setup


@pytest.mark.parametrize('role',['preflight','client','service-0','service-1'])
def test_hf_reference_is_unchanged_and_services_use_acoustic(role):
    before=a.original.command(config(),role);after=a.command(config(),role)
    if role.startswith('service-'):
        assert len(before)==len(after)
        assert [(x,y) for x,y in zip(before,after) if x!=y]==[
            ('tools.reports.context_dual_runtime_candidate','tools.reports.context_acoustic_service_candidate')]
    else:assert before==after
    assert a._base_validate.__globals__['command'] is a.command
    assert a.original.validate.__globals__['command'] is a.original.command


@pytest.mark.parametrize('fault',[None,'kv','asset','extra','parent'])
def test_real_sealing_and_parent_delta_rejection(tmp_path,monkeypatch,fault):
    c,source=setup(tmp_path,monkeypatch);c['parent_kind']='kv16'
    parent_path=Path(c['parent_operations']);parent=json.loads(parent_path.read_text())
    receipt=a.preparation.validate_parent(parent)
    parent['manifest_sha256']=receipt['files_sha256'];parent_path.write_text(json.dumps(parent))
    monkeypatch.setattr(kv,'validate',lambda c:receipt)
    before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    # Bind after fixture installs policy/input substitutes; actual file sealing
    # and full service/source difference validators execute.
    monkeypatch.setattr(a,'_base_validate',a.bind(a.original.validate,command=a.command))
    result=a.prepare(c,{})
    package=Path(c['output'])/'package';ops=json.loads((package.parent/'operations.json').read_text())
    assert result['launched'] is False and ops['finite_inference_dispatch_allowed'] is False
    assert before=={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    if fault=='parent':ops['efficiency_parent']['manifest_sha256']='other'
    elif fault=='asset':(package/'model_code.py').write_text('changed')
    elif fault=='extra':(package/'extra.py').write_text('unexpected')
    elif fault=='kv':
        path=package/'service-0.json';s=json.loads(path.read_text());s['policy']['kv_cache_memory_bytes']=1
        path.write_text(json.dumps(s))
    # Re-seal tampered contents to ensure rejection is not only FILES checking.
    if fault in ('asset','extra','kv'):
        a.write(package/'FILES.json',{str(p.relative_to(package)):a.file_hash(p) for p in package.rglob('*') if p.is_file() and p.name!='FILES.json'})
        r=json.loads((package.parent/'PREPARED.json').read_text());r['files_sha256']=a.preparation.verify(package)
        ops['manifest_sha256']=r['files_sha256'];a.write(package.parent/'PREPARED.json',r)
    if fault:
        with pytest.raises(ValueError):a.validate(ops)
    else:assert a.validate(ops)['dataset']=='MINDS_native14'
