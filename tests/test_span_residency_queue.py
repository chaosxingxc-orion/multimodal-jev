import os
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import queue_span_residency as module
from multimodal_jev.baseline import atomic_json,file_hash


def fixture(tmp_path):
    root=tmp_path/'predecessor';root.mkdir();(root/'run').mkdir()
    atomic_json(root/'config.json',{})
    atomic_json(root/'launch.json',dict(pid=os.getpid(),argv=Path(f'/proc/{os.getpid()}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')))
    return dict(predecessor_controller=str(root),predecessor_suite=str(tmp_path/'suite'),pins={str(root/n):file_hash(root/n) for n in ('launch.json','config.json')})


def test_ten_minute_wait_stops_on_failure(tmp_path,monkeypatch):
    c=fixture(tmp_path);waits=[]
    def sleep(seconds):
        waits.append(seconds);atomic_json(Path(c['predecessor_controller'])/'run/terminal.json',dict(exit_code=1))
    monkeypatch.setattr(module.time,'sleep',sleep)
    with pytest.raises(ValueError,match='predecessor failed'):module.await_predecessor(c,tmp_path)
    assert waits==[600]


def test_success_file_does_not_substitute_complete_update(tmp_path):
    c=fixture(tmp_path);root=Path(c['predecessor_controller'])
    atomic_json(root/'run/terminal.json',dict(exit_code=0));atomic_json(root/'run/progress.json',dict(completed_updates=0))
    with pytest.raises(ValueError,match='complete first update'):module.await_predecessor(c,tmp_path)


@pytest.mark.parametrize('missing',['rank','residency','bf16','checkpoint'])
def test_probe_acceptance_rejects_missing_or_wrong_evidence(tmp_path,missing):
    for rank in range(4):
        out=tmp_path/'run'/f'rank-{rank}';out.mkdir(parents=True);(out/'local-checkpoint.pt').write_bytes(b'evidence')
        r=dict(status='passed',rank=rank,world_size=4,updates=3,restored_at_update=2,backend='hccl',
            exact_resume_output=True,exact_resume_masters=True,exact_resume_adam=True,frozen_base_unchanged=True,
            cross_rank_export_exact=True,execution_dtypes=[['torch.bfloat16','torch.bfloat16']],adapter_export_hashes={'p':'h'},
            checkpoint_sha256=file_hash(out/'local-checkpoint.pt'),residency_roundtrip=dict(exact_parameters_and_adam=True,
                original_tensor_devices_preserved=True,parameter_references_preserved=True))
        if rank==3:
            if missing=='rank':r['rank']=2
            elif missing=='residency':r['residency_roundtrip']['exact_parameters_and_adam']=False
            elif missing=='bf16':r['execution_dtypes']=[['torch.float32','torch.float32']]
            else:r['checkpoint_sha256']='different'
        atomic_json(out/'result.json',r);atomic_json(out/'terminal.json',dict(exit_code=0))
    with pytest.raises(ValueError):module.verify_results(tmp_path)
