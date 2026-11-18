import copy
import json

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.encoded_audio_cache import key_for
from multimodal_jev.span_request_identity import tensor_identity
from multimodal_jev.slurp_training import content_hash
from test_cached_context_preparation import setup
from tools.reports.context_selection_audited_cache import bind_evidence,encoder


def evidence():
    population=[dict(id=str(i),audio_sha256='a',request_sha256='r') for i in range(9226)]
    tensors={k:{} for k in ('input_ids','attention_mask','input_features_mask')}
    tensors.update(positions=dict(shape=[1,60],dtype='torch.int64'),input_features=dict(dtype='torch.bfloat16'))
    values=[dict(p,tensors=tensors) for p in population]
    manifest=dict(schema='context-selection-cache-v1',role='selection',recordings=9226,
        release_sha256='old',processor_sha256='processor',request_version='jev-context-exclusive_event-v1')
    proof=dict(state=dict(Status='exited',ExitCode=0,OOMKilled=False),manifest=manifest)
    release=dict(schema='slurp-official-clean-intent-v2',migration=dict(source_manifest_sha256='old',
        unchanged_payload_sha256={k:k for k in ('selection.jsonl','selection.audio.jsonl','tasks.json')}))
    release['files']=dict(release['migration']['unchanged_payload_sha256'])
    return population,values,proof,release


def stream(values,manifest):
    return '\n'.join(json.dumps(v) for v in [dict(kind='terminal',value=dict(exit_code=0,recordings=9226)),
        *[dict(kind='input',value=r) for r in values],dict(kind='manifest',value=manifest)])


def test_complete_old_evidence_matches_explicit_portable_migration():
    population,values,proof,release=evidence()
    got,report=bind_evidence(proof,stream(values,proof['manifest']),population,release_sha256='new',
        release_manifest=release,processor_sha256='processor')
    assert len(got)==report['recordings']==9226


@pytest.mark.parametrize('fault',['migration','payload','duplicate','order','request','audio','processor','failed','width'])
def test_wrong_evidence_rejected(fault):
    population,values,proof,release=evidence();values=copy.deepcopy(values);processor='processor'
    if fault=='migration':release['migration']['source_manifest_sha256']='wrong'
    if fault=='payload':release['files']['tasks.json']='wrong'
    if fault=='duplicate':values[1]=values[0]
    if fault=='order':values.reverse()
    if fault=='request':values[0]['request_sha256']='wrong'
    if fault=='audio':values[0]['audio_sha256']='wrong'
    if fault=='processor':processor='wrong'
    if fault=='failed':proof['state']['ExitCode']=1
    if fault=='width':values[0]['tensors']['positions']['shape']=[1,31]
    with pytest.raises(ValueError):bind_evidence(proof,stream(values,proof['manifest']),population,
        release_sha256='new',release_manifest=release,processor_sha256=processor)


@pytest.mark.parametrize('fault',[None,'missing','bytes','tensor','request','processor'])
def test_consumption_reads_only_audited_cache_no_audio_or_processor_rebuild(tmp_path,monkeypatch,fault):
    request,digest,factory,calls,audio=setup(tmp_path,monkeypatch)
    prepare=factory(limit=4096);value=prepare((request,digest))
    key=key_for(prepare.identity(request,digest));path=prepare.cache.root/(key+'.pt')
    asset=dict(id='audio',audio_sha256=digest,request_sha256=content_hash(request.model_input()))
    receipt=dict(asset,cache_key=key,cache_sha256=file_hash(path),
        tensors={k:tensor_identity(v) for k,v in dict(value.inputs,positions=value.positions).items()})
    audio.unlink()  # Audited features are the input; source audio is not reopened.
    processor='a'*64
    if fault=='missing':path.unlink()
    if fault=='bytes':path.write_bytes(b'changed')
    if fault=='tensor':receipt['tensors']['input_features']['sha256']='wrong'
    if fault=='request':asset['request_sha256']='wrong'
    if fault=='processor':processor='b'*64
    encode=encoder(prepare.cache.root,{'audio':request},['q'],[receipt],
        processor_sha256=processor,decision_token_id=99,audio_token_id=98)
    if fault:
        with pytest.raises((ValueError,FileNotFoundError)):encode(asset)
    else:
        row,bound=encode(asset)
        assert bound['positions']==[2] and row['inputs']['input_features'].dtype==value.inputs['input_features'].dtype
    assert len(calls)==1
