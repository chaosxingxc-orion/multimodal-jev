import importlib.util
import json
from pathlib import Path
import pytest
from test_compare_policies import fixture
from multimodal_jev.baseline import file_hash


def module():
    path=Path(__file__).parents[1]/'tools/jev/finalize_policy_comparisons.py'
    spec=importlib.util.spec_from_file_location('finalizer',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def job(tmp_path):
    manifest,base,candidate=fixture(tmp_path)
    return {'manifest':str(manifest),'manifest_sha256':file_hash(manifest),'draws':100,
            'base':{'run':str(base),'host':'npu','container':'base','container_id':'base-id'},
            'candidate':{'run':str(candidate),'host':'npu','container':'candidate','container_id':'candidate-id'}}


def state(name,status='exited',exit_code=0):
    return {'id':name+'-id','state':{'Status':status,'Running':status=='running','ExitCode':exit_code}}


def test_live_container_never_scores_even_with_old_terminal_files(tmp_path):
    m=module();j=job(tmp_path)
    result=m.finalize_job(j,tmp_path/'out',inspector=lambda host,name:state(name,'running'))
    assert result['phase']=='waiting_for_evaluation'
    assert not (tmp_path/'out/comparison.json').exists()


def test_observation_error_is_retry_not_terminal(tmp_path):
    m=module();j=job(tmp_path)
    def unavailable(*args):raise m.ObservationUnavailable('timeout')
    result=m.finalize_job(j,tmp_path/'out',inspector=unavailable)
    assert result['phase']=='observation_retry'
    assert not (tmp_path/'out/comparison.json').exists()


def test_failed_or_replaced_container_is_rejected(tmp_path):
    m=module();j=job(tmp_path)
    with pytest.raises(RuntimeError,match='exited 1'):
        m.finalize_job(j,tmp_path/'out',inspector=lambda h,n:state(n,exit_code=1))
    with pytest.raises(ValueError,match='identity changed'):
        m.finalize_job(j,tmp_path/'out',inspector=lambda h,n:state('replacement'))


def test_completed_comparison_rescores_without_touching_source_metrics(tmp_path):
    m=module();j=job(tmp_path);before=file_hash(Path(j['candidate']['run'])/'metrics.json')
    inspector=lambda h,n:state(n)
    result=m.finalize_job(j,tmp_path/'out',inspector=inspector)
    assert result['phase']=='completed' and result['candidate_all']['correct']==3
    assert file_hash(Path(j['candidate']['run'])/'metrics.json')==before
    assert m.finalize_job(j,tmp_path/'out',inspector=inspector)==result
    p=tmp_path/'out/comparison.json';d=json.loads(p.read_text());d['draws']=1;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='comparison changed'):
        m.finalize_job(j,tmp_path/'out',inspector=inspector)


def test_exit_zero_without_prediction_receipt_does_not_finalize(tmp_path):
    m=module();j=job(tmp_path)
    (Path(j['candidate']['run'])/'terminal.json').unlink()
    with pytest.raises(FileNotFoundError):
        m.finalize_job(j,tmp_path/'out',inspector=lambda h,n:state(n))
    assert not (tmp_path/'out/comparison.json').exists()
