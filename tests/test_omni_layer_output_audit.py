import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

from multimodal_jev.omni_layer_output_audit import (
    difference, paired_rows, validate_accounting, validate_output,
)


def tensors():
    hidden = torch.arange(24,dtype=torch.bfloat16).reshape(2,3,4)
    mask = torch.tensor([[0,1,1],[1,1,1]])
    return dict(output=hidden.clone(),token_mask=mask.clone()),dict(hidden_states=hidden,token_mask=mask)


@pytest.mark.parametrize('change',['shape','dtype','nan','mask','mask_dtype','extra'])
def test_outputs_reject_corruption(change):
    saved, original = tensors()
    validate_output(saved,original)
    if change=='shape':saved['output']=saved['output'][:1]
    if change=='dtype':saved['output']=saved['output'].float()
    if change=='nan':saved['output'][0,0,0]=float('nan')
    if change=='mask':saved['token_mask'][0,0]=1
    if change=='mask_dtype':saved['token_mask']=saved['token_mask'].bool()
    if change=='extra':saved['unexpected']=True
    with pytest.raises(ValueError):validate_output(saved,original)


def test_padded_differences_do_not_contaminate_valid_statistics_or_disappear():
    batch,_ = tensors()
    single = {k:v[:1].clone() for k,v in batch.items()}
    batch['output'][0,0,0] += 2
    observed = paired_rows(single,batch,0)
    assert observed['valid_tokens']==2
    assert observed['valid']['equal'] and observed['valid']['elements']==8
    assert observed['full_padded']['differing_elements']==1
    assert observed['full_padded']['max_abs_error']==2
    assert observed['full_padded']['mean_abs_error']==pytest.approx(2/12)
    assert observed['full_padded']['rms_error']==pytest.approx((4/12)**.5)
    batch['output'][0,1,0] += 1
    assert paired_rows(single,batch,0)['valid']['differing_elements']==1
    batch['token_mask'][0,0]=1
    with pytest.raises(ValueError,match='masks'):paired_rows(single,batch,0)


@pytest.mark.parametrize('value',[torch.empty(0),torch.tensor([float('nan')]),torch.tensor([float('inf')])])
def test_no_empty_or_nonfinite_comparison(value):
    with pytest.raises(ValueError):difference(value,value)


def records():
    return [dict(size=s,offset=o,peak_allocated_bytes=10,peak_reserved_bytes=20,seconds=1.)
            for s in (1,2) for o in range(0,6,s)]


@pytest.mark.parametrize('change',['missing','duplicate','reordered','peak_decrease','peak_inverted',
    'bool_peak','nan_seconds','negative_seconds'])
def test_all_original_calls_and_cumulative_peaks_required(change):
    rows=records()
    assert validate_accounting(rows)['forward_and_save_seconds']==9
    if change=='missing':rows.pop()
    if change=='duplicate':rows[-1]=copy.deepcopy(rows[0])
    if change=='reordered':rows.reverse()
    if change=='peak_decrease':rows[-1]['peak_allocated_bytes']=9
    if change=='peak_inverted':rows[-1]['peak_allocated_bytes']=21
    if change=='bool_peak':rows[0]['peak_allocated_bytes']=True
    if change=='nan_seconds':rows[0]['seconds']=float('nan')
    if change=='negative_seconds':rows[0]['seconds']=-1
    with pytest.raises(ValueError):validate_accounting(rows)


def test_auditor_refuses_running_or_failed_job_before_inspection(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
    import audit_omni_npu_layer as auditor
    monkeypatch.setattr(auditor,'check_pins',lambda *args:pytest.fail('must stop before artifact audit'))
    with pytest.raises(FileNotFoundError):auditor.audit(tmp_path,tmp_path/'prefix',tmp_path/'audit.json')
    (tmp_path/'evaluation-terminal.json').write_text('{"exit_code":1}')
    with pytest.raises(ValueError,match='successful original'):auditor.audit(tmp_path,tmp_path/'prefix',tmp_path/'audit.json')
    assert not (tmp_path/'audit.json').exists()


def test_complete_saved_output_audit_without_forward(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
    import audit_omni_npu_layer as auditor
    torch.set_num_threads(2)
    root=tmp_path/'probe';run=root/'run';run.mkdir(parents=True)
    prefix=tmp_path/'prefix';prefix.mkdir()
    preflight=tmp_path/'preflight';(preflight/'run').mkdir(parents=True)
    write=lambda p,v:p.write_text(json.dumps(v))
    write(prefix/'identity.json',{'order':[2,0,5,1,3,4]})
    calls=[];outputs=[];originals=[]
    for size,offset in [(s,o) for s in (1,2) for o in range(0,6,s)]:
        name=f'micro-{size}-offset-{offset}.pt'
        mask=torch.ones(size,1230,dtype=torch.int64);mask[:,0]=0
        hidden=torch.zeros(size,1230,2048,dtype=torch.bfloat16)
        torch.save(dict(hidden_states=hidden,token_mask=mask),prefix/name)
        call=dict(size=size,offset=offset,input_file=name,input_sha256=auditor.file_hash(prefix/name),
            input_shape=[size,1230,2048],valid_lengths=[1229]*size)
        calls.append(call)
        if size==2:hidden[0,0,0]=1 # Padded-only perturbations remain visible.
        torch.save(dict(output=hidden,token_mask=mask),run/name)
        record=dict(**call,output_file=name,output_sha256=auditor.file_hash(run/name),seconds=1.,
            peak_allocated_bytes=100,peak_reserved_bytes=200)
        outputs.append(record);write((run/name).with_suffix('.json'),record)
        originals.append(dict(micro_batch_size=size,offset=offset,tensor_file=name,tensor_sha256=call['input_sha256']))
    write(prefix/'result.json',{'records':originals})
    prefix_sha=auditor.file_hash(prefix/'result.json');monkeypatch.setattr(auditor,'PREFIX_SHA',prefix_sha)
    identity=dict(calls=calls,order=[2,0,5,1,3,4],preflight_only=False,
        prefix_result_sha256=prefix_sha,prefix_identity_sha256=auditor.file_hash(prefix/'identity.json'))
    write(run/'identity.json',identity)
    write(preflight/'run/identity.json',dict(identity,preflight_only=True))
    write(preflight/'run/result.json',dict(identity_sha256=auditor.file_hash(preflight/'run/identity.json')))
    prep=dict(source_commit='commit',source_files={'runner.py':hashlib.sha256(b'code').hexdigest()},
        preflight_root=str(preflight),image='image')
    write(root/'intent.json',prep)
    write(root/'launch.json',dict(container_id='cid',intent_sha256=auditor.file_hash(root/'intent.json'),
        source_commit='commit',physical_devices=[4]))
    write(run/'result.json',dict(outputs=outputs,identity_sha256=auditor.file_hash(run/'identity.json'),
        status='diagnostic_complete',preflight_only=False,calls=9,npu_forward_calls=9,weight_updates=0))
    write(root/'execution-complete.json',dict(container_id='cid',result_sha256=auditor.file_hash(run/'result.json')))
    for p in (root/'evaluation-terminal.json',run/'terminal.json'):write(p,{'exit_code':0})
    monkeypatch.setattr(auditor,'check_pins',lambda *args:({},prep))
    monkeypatch.setattr(auditor,'inspect',lambda *args:dict(RestartCount=0,State=dict(Error='',Dead=False)))
    monkeypatch.setattr(auditor,'successful',lambda *args:None)
    monkeypatch.setattr(auditor,'check_container',lambda *args:None)
    monkeypatch.setattr(auditor.subprocess,'check_output',lambda *args,**kw:b'code')
    report=auditor.audit(root,prefix,tmp_path/'audit.json')
    assert len(report['calls'])==9 and len(report['micro1_micro2_comparisons'])==6
    assert all(p['valid']['equal'] for p in report['micro1_micro2_comparisons'])
    assert sum(p['full_padded']['differing_elements'] for p in report['micro1_micro2_comparisons'])==3
    with pytest.raises(FileExistsError):auditor.audit(root,prefix,tmp_path/'audit.json')
    (run/'micro-1-offset-0.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='tensor or per-call'):auditor.audit(root,prefix,tmp_path/'corrupt.json')
    assert not (tmp_path/'corrupt.json').exists()
