import sys
from pathlib import Path
import pytest
import json
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from test_sfc_jev import load_tool
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))


def test_context_audio_segments_frame_coverage_and_exact_budget_boundary():
    module=load_tool('audit_sfc_jev_context')
    ids=[1,9,9,2,9,9,9,3]
    result=module.check_coverage({'audio_sha256':['a','b']},ids,[101,200],[16001,32001],160,9)
    assert result['audio_token_lengths']==[2,3] and not result['context_overflow']
    assert module.check_coverage({'audio_sha256':['a']},[9]+[1]*3071,[3001],[480160],160,9)['context_overflow'] is False
    assert module.check_coverage({'audio_sha256':['a']},[9]+[1]*3072,[3001],[480160],160,9)['context_overflow'] is True


def test_long_audio_is_checked_without_thirty_second_exclusion():
    module=load_tool('audit_sfc_jev_context')
    result=module.check_coverage({'audio_sha256':['a']},[1,9,9,2],[3965],[634412],160,9)
    assert result['observed_frames']==[3965]
    with pytest.raises(ValueError,match='coverage'):
        module.check_coverage({'audio_sha256':['a']},[1,9,9,2],[3000],[634412],160,9)


def test_missing_or_merged_audio_segments_rejected():
    module=load_tool('audit_sfc_jev_context')
    for ids in ([1,2],[1,9,9,9,2]):
        with pytest.raises(ValueError,match='count'):
            module.check_coverage({'audio_sha256':['a','b']},ids,[10,20],[1600,3200],160,9)


def test_independent_verifier_detects_missing_decisions_even_with_updated_file_hash(tmp_path,monkeypatch):
    module=load_tool('verify_sfc_jev_context');run=tmp_path/'run';run.mkdir();release=tmp_path/'release';release.mkdir()
    pins={};reports={}
    for split in ('train_id','test_id','test_ood'):
        row=dict(id=split,split=split,source_group=split,audio_sha256=['audio'],input={'decisions':[{'id':'a'},{'id':'b'}]})
        path=release/f'{split}.requests.jsonl';path.write_bytes(canonical(row))
        pins[split]=dict(bundles=1,decisions=2,source_groups=1,requests_sha256=file_hash(path))
        records=[dict(id=split,decision_id=k,audio_token_lengths=[10],expected_frames=[20],observed_frames=[20],
            input_tokens=100,context_overflow=False,prompt_sha256='a'*64,input_ids_sha256='b'*64) for k in ('a','b')]
        path=run/f'{split}.context.jsonl';path.write_bytes(b''.join(map(canonical,records)))
        reports[split]=dict(bundles=1,decisions=2,source_groups=1,context_overflow=0,min_input_tokens=100,max_input_tokens=100,records_sha256=file_hash(path))
    (release/'MANIFEST.json').write_bytes(canonical(dict(splits=pins)))
    identity=dict(targets_read=False,context_limit=4096,output_budget=1024,source_files={},release=str(release),manifest_sha256=file_hash(release/'MANIFEST.json'))
    (run/'identity.json').write_bytes(canonical(identity));(run/'terminal.json').write_text('{"exit_code":0}')
    result=dict(status='passed',identity_sha256=file_hash(run/'identity.json'),splits=reports,all_inputs_fit=True)
    (run/'result.json').write_bytes(canonical(result));(tmp_path/'launch.json').write_text('{"container_id":"cid","image":"image"}')
    monkeypatch.setattr(module,'inspect',lambda cid:dict(Id='cid',Image='image',State={'Status':'exited','ExitCode':0},HostConfig={'Devices':[]}))
    assert module.verify(tmp_path)['all_inputs_fit']
    path=run/'test_id.context.jsonl';path.write_bytes(path.read_bytes().splitlines(keepends=True)[0])
    reports['test_id']['records_sha256']=file_hash(path)
    (run/'result.json').write_bytes(canonical(result))
    with pytest.raises(StopIteration):module.verify(tmp_path)
