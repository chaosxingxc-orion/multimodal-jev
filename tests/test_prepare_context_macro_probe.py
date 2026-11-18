import json
from pathlib import Path

import pytest

from tools.reports import prepare_context_macro_probe as m


def fixture(tmp_path,monkeypatch):
    parent=tmp_path/'parent';package=parent/'package';package.mkdir(parents=True)
    (package/'config.json').write_text(json.dumps(dict(policy=dict(policy_id='p',base_id='base',
        contract={},max_batch_queries=4,max_batch_tokens=16384),input_sha256='input',audio_token_id=1)))
    for name in ('inputs.pt','INPUTS.json'):(package/name).write_text('fixture')
    (package/'FILES.json').write_text('{}')
    (parent/'PREPARED.json').write_text(json.dumps(dict(files_sha256='parent',
        remote_root='/local/old',image='sha256:'+'a'*64)))
    (package/'old.py').write_text('x=1')
    reference=tmp_path/'reference.json';reference.write_text('{}')
    transport=tmp_path/'transport.json';transport.write_text(json.dumps(dict(files={
        'evaluation/serial/measurement.json':m.file_hash(reference)})))
    audit=tmp_path/'audit.json';audit.write_text(json.dumps(dict(
        status='finite_fixed_policy_inference_audited',files_sha256='parent',
        exact_actual_logits=True,exact_probabilities_and_G16=True,transport_sha256=m.file_hash(transport))))
    c=dict(parent=str(parent),source=str(Path.cwd()),output=str(tmp_path/'prepared'),
        parent_manifest_sha256='parent',parent_audit=str(audit),parent_audit_sha256=m.file_hash(audit),
        parent_transport=str(transport),serial_reference=str(reference),
        serial_reference_sha256=m.file_hash(reference),remote_root='/local/new',ipc_root='/jev-ipc',model='/models/asr')
    monkeypatch.setattr(m,'verify',lambda p:'parent' if p==package else 'new')
    monkeypatch.setattr(m,'processor_cache_identity',lambda p:'processor')
    monkeypatch.setattr(m.torch,'load',lambda *a,**kw:['fixture'])
    monkeypatch.setattr(m,'check_reference',lambda *a:None)
    return c


def test_service_pair_and_measurement_share_pinned_runtime_and_policy(tmp_path,monkeypatch):
    c=fixture(tmp_path,monkeypatch);result=m.prepare(c,{'fixture':True})
    p=Path(c['output'])/'package'
    services=[json.loads((p/f'service-{i}.json').read_text()) for i in range(2)]
    measurement=json.loads((p/'measurement.json').read_text())
    for i,s in enumerate(services):
        assert s['policy']['physical_devices']==[6+i]
        assert s['dual_runtime_identity']==measurement['macro_client']['runtime_identity']
        assert s['socket_path']==measurement['macro_client']['replicas'][i]['socket']
        assert s['exchange']==measurement['macro_client']['replicas'][i]['exchange']
        assert s['policy']['snapshot']==measurement['policy']['snapshot']
        for name in ('context_dual_runtime_candidate.py','serve_context_dual_candidate.py'):
            assert c['remote_root']+'/package/tools/reports/'+name in s['pins']
    assert result['training_dispatch_allowed'] is False
    sources=json.loads((p/'RUNTIME-SOURCES.json').read_text())
    assert 'src/multimodal_jev/local_config.py' in sources
    assert set(sources)=={str(f.relative_to(p)) for f in p.rglob('*.py')}


@pytest.mark.parametrize('fault',['audit','transport','reference','parent','ipc'])
def test_missing_actual_parent_evidence_rejects_before_copy(tmp_path,monkeypatch,fault):
    c=fixture(tmp_path,monkeypatch)
    if fault=='audit':c['parent_audit_sha256']='wrong'
    if fault=='transport':Path(c['parent_transport']).write_text('{}')
    if fault=='reference':Path(c['serial_reference']).write_text('{"changed":true}')
    if fault=='parent':c['parent_manifest_sha256']='wrong'
    if fault=='ipc':c['ipc_root']='/'+'x'*110
    with pytest.raises(ValueError):m.prepare(c,{})
    assert not Path(c['output']).exists()
