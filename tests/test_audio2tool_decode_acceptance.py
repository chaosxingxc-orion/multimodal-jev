from pathlib import Path
import json

import pytest

from multimodal_jev.audio2tool_decode_acceptance import audit_records
from multimodal_jev.audio2tool_decode_profile import summarize
from multimodal_jev.baseline import file_hash


def fixture(tmp):
    run=tmp/'run'; before=tmp/'preflight'; run.mkdir(); before.mkdir()
    def write(p,v): p.write_text(json.dumps(v))
    names=['case-'+str(i) for i in range(16)]
    common=dict(versions={},engine={},model='fixture',asset_audit_sha256='fixture',model_files=[],sources={},
                cases=names,sampling=dict(temperature=0.,max_tokens=1024,n=1,seed=17),input_pins={},
                protocol='fixture',targets_read=False,weight_updates=0)
    write(before/'identity.json',dict(**common,preflight_only=True,capacity=None))
    write(run/'identity.json',dict(**common,preflight_only=False,capacity={'fixture':True}))
    rows=[]; priors=[]
    for i,name in enumerate(names):
        a=dict(name=name,family='native' if i<4 else 'jev',id=str(i),decision_id=None,
               prompt_sha256='fixture',audio_receipts=[],input_tokens=10,input_ids_sha256='fixture',preparation_seconds=.1)
        prior=dict(**a,model_loaded=False); priors.append(prior); write(before/(name+'.json'),prior)
        row=dict(**a,model_loaded=True,engine_seconds=1.,response=dict(raw_text='invalid JSON',error=None,
                 token_ids=[7],finish_reason='stop',length_limited=False))
        rows.append(row); write(run/(name+'.attempt.json'),a); write(run/(name+'.json'),row)
    write(before/'result.json',dict(status='preflight_passed',model_loaded=False,records=priors,targets_read=False,
          weight_updates=0,identity_sha256=file_hash(before/'identity.json')))
    write(run/'result.json',dict(status='profile_complete',model_loaded=True,records=rows,targets_read=False,
          weight_updates=0,identity_sha256=file_hash(run/'identity.json'),summary=summarize(rows,names),
          elapsed_seconds=30.,engine_load_seconds=2.))
    for r in (run,before): write(r/'terminal.json',dict(exit_code=0))
    write(run/'effective-cache.json',dict(configuration_exposed=False,fields=dict(mm_processor_cache_gb=None,
          mm_processor_cache_type=None,mm_shm_cache_max_object_size_mb=None)))
    return run,before


@pytest.mark.parametrize('fault',[None,'missing','extra','order','input','attempt','tokens','truncation',
                                 'decode','summary','duration','cache','preflight','capacity','targets'])
def test_complete_artifact_audit(tmp_path,fault):
    run,before=fixture(tmp_path)
    def edit(p,fn):
        v=json.loads(p.read_text());fn(v);p.write_text(json.dumps(v))
    def rowchange(fn):
        edit(run/'case-0.json',fn)
        edit(run/'result.json',lambda v:fn(v['records'][0]))
    if fault=='missing': (run/'case-0.json').unlink()
    if fault=='extra': (run/'extra.json').write_text('{}')
    if fault=='order': edit(run/'result.json',lambda v:v['records'].reverse())
    if fault=='input':
        rowchange(lambda v:v.update(input_tokens=11));edit(run/'case-0.attempt.json',lambda v:v.update(input_tokens=11))
    if fault=='attempt': edit(run/'case-0.attempt.json',lambda v:v.update(id='wrong'))
    if fault=='tokens': rowchange(lambda v:v['response'].update(token_ids=[True]))
    if fault=='truncation': rowchange(lambda v:v['response'].update(finish_reason='length',length_limited=True))
    if fault=='summary': edit(run/'result.json',lambda v:v['summary'].update(population_throughput_estimate=100))
    if fault=='duration': edit(run/'result.json',lambda v:v.update(elapsed_seconds=1))
    if fault=='cache': edit(run/'effective-cache.json',lambda v:v['fields'].pop('mm_processor_cache_gb'))
    if fault=='preflight': edit(before/'result.json',lambda v:v.update(model_loaded=True))
    if fault in ('capacity','targets'):
        edit(run/'identity.json',lambda v:v.update(**({'capacity':None} if fault=='capacity' else {'targets_read':True})))
        edit(run/'result.json',lambda v:v.update(identity_sha256=file_hash(run/'identity.json')))
    decode=lambda tokens:'wrong' if fault=='decode' else 'invalid JSON'
    if fault is None:
        result=audit_records(run,before,decode)
        assert result['records']==16 and result['recomputed_summary']['population_throughput_estimate'] is None
    else:
        with pytest.raises((ValueError,FileNotFoundError)): audit_records(run,before,decode)


def test_natural_eos_and_length_limited_invalid_json_both_retained(tmp_path):
    run,before=fixture(tmp_path)
    path=run/'case-0.json'; row=json.loads(path.read_text())
    row['response'].update(token_ids=[7]*1024,finish_reason='length',length_limited=True)
    path.write_text(json.dumps(row))
    path=run/'result.json'; result=json.loads(path.read_text());result['records'][0]=row
    result['summary']=summarize(result['records'],[r['name'] for r in result['records']])
    path.write_text(json.dumps(result))
    accepted=audit_records(run,before,lambda _: 'invalid JSON')
    summary=accepted['recomputed_summary']['boundary_observations']
    assert summary['native']['length_limited']==1 and summary['native']['output_tokens']==1027
    assert summary['jev']['length_limited']==0
    assert json.loads((run/'case-0.json').read_text())['response']==row['response']
