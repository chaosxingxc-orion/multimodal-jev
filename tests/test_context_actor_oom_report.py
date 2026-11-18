import json

import pytest

from tools.reports import report_context_actor_oom as m


def fixture(tmp_path,monkeypatch):
    def write(name,value):(tmp_path/name).write_text(json.dumps(value))
    write('terminal.json',dict(exit_code=1))
    for role in ('preflight','client','service-0','service-1'):
        write(role+'-launch.json',dict(container_id=role))
        write(role+'-inspect.json',dict(Id=role,State=dict(Running=False,Status='exited',
            ExitCode=int(role=='client'),OOMKilled=False)))
    log='\n'.join(f'[rank{r}]: torch.OutOfMemoryError: NPU out of memory. Tried to allocate 178.00 MiB (NPU {r}; 60.96 GiB total capacity; 57.69 GiB already allocated; 57.69 GiB current active; 11.32 MiB free; 59.52 GiB reserved' for r in range(2))
    (tmp_path/'client.log').write_text(log)
    def inspect(observed,handle):
        if observed['Id']!=handle['container_id']:raise ValueError('identity differs')
    monkeypatch.setattr(m,'inspect_runtime',inspect)
    return dict(operations_root=str(tmp_path))


def test_device_oom_is_not_docker_host_oom(tmp_path,monkeypatch):
    c=fixture(tmp_path,monkeypatch);r=m.report(c)
    assert r['classification']=='actor_device_allocation_failed'
    assert r['containers']['client']['state']['OOMKilled'] is False
    assert r['rank_failures'][1]['allocated_gib']==57.69
    assert len(r['evidence_sha256'])==10 and r['retry'] is False


@pytest.mark.parametrize('fault',['running','wrong_container','success','missing_rank','generic_error'])
def test_no_false_terminal_or_oom_claim(tmp_path,monkeypatch,fault):
    c=fixture(tmp_path,monkeypatch)
    if fault in ('running','wrong_container','success'):
        p=tmp_path/'client-inspect.json';v=json.loads(p.read_text())
        if fault=='running':v['State'].update(Running=True,Status='running')
        if fault=='wrong_container':v['Id']='different'
        if fault=='success':v['State']['ExitCode']=0
        p.write_text(json.dumps(v))
    else:
        p=tmp_path/'client.log';p.write_text(p.read_text().splitlines()[0] if fault=='missing_rank' else 'failed')
    with pytest.raises(ValueError):m.report(c)
