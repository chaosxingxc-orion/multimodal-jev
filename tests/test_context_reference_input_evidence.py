import copy
import json

import pytest
import torch

from multimodal_jev.context_reference_dataset import digest
from multimodal_jev.span_request_identity import tensor_identity
from tools.reports.context_reference_input_evidence import bindings_from_records,load,bind


def fixture():
    positions=list(range(1,61))
    inputs=dict(input_ids=tensor_identity(torch.arange(64)[None]),
        attention_mask=tensor_identity(torch.ones(1,64,dtype=torch.long)),
        input_features=tensor_identity(torch.ones(1,128,100,dtype=torch.bfloat16)),
        input_features_mask=tensor_identity(torch.ones(1,100,dtype=torch.int32)))
    binding=dict(policy_id='a'*64,positions=positions,inputs=inputs);key=digest(binding)
    records={key:dict(request_id=key,policy_id='a'*64,request_identity=binding)}
    evidence={s:dict(tensors=dict(inputs,positions=tensor_identity(torch.tensor([positions],dtype=torch.long))))
              for s in ('sample0','sample1')}
    return copy.deepcopy(evidence),records,{s:key for s in evidence}


@pytest.mark.parametrize('fault',[None,'position','input','missing','policy','duplicate_marker'])
def test_original_tensor_hashes_authenticate_concrete_positions_and_duplicate_samples(fault):
    evidence,records,population=fixture();key=population['sample0'];record=records[key]
    if fault=='position':record['request_identity']['positions'][0]=0
    if fault=='input':record['request_identity']['inputs']['input_ids']['sha256']='f'*64
    if fault=='policy':record['request_identity']['policy_id']='b'*64
    if fault=='duplicate_marker':record['request_identity']['positions'][1]=1
    if fault=='missing':evidence.pop('sample1')
    if fault in ('position','input','policy','duplicate_marker'):
        newkey=digest(record['request_identity']);record['request_id']=newkey
        records={newkey:record};population={s:newkey for s in population}
    if fault:
        with pytest.raises(ValueError):bindings_from_records(evidence,records,population,policy_id='a'*64)
    else:
        result=bindings_from_records(evidence,records,population,policy_id='a'*64)
        assert set(result)=={'sample0','sample1'} and result['sample0']==result['sample1']


def test_stream_hash_must_be_bound_by_independent_proof(tmp_path):
    from multimodal_jev.baseline import file_hash
    proof=tmp_path/'proof.json';proof.write_text(json.dumps(dict(independent_receipts_sha256='wrong')))
    stream=tmp_path/'stream.jsonl';stream.write_text('{}')
    c=dict(input_proof=str(proof),input_proof_sha256=file_hash(proof),
        input_receipts=str(stream),input_receipts_sha256=file_hash(stream))
    with pytest.raises(ValueError,match='authenticate'):
        load(c,rows=[],task={},release_sha256='r',processor_sha256='p')


def test_selection_or_partial_audit_cannot_supply_train_evidence():
    manifest=dict(schema='context-selection-cache-v1',recordings=1)
    stream='\n'.join(json.dumps(v) for v in [dict(kind='terminal',value={'exit_code':0}),
        dict(kind='input',value={}),dict(kind='manifest',value=manifest)])
    proof=dict(state=dict(Status='exited',ExitCode=0,OOMKilled=False),manifest=manifest)
    with pytest.raises(ValueError,match='TRAIN'):
        bind(proof,stream,rows=[{}],task={},release_sha256='r',processor_sha256='p')


def test_reference_build_requires_explicit_local_audio_mirror():
    from tools.reports.build_context_reference_dataset import audio_mirror
    with pytest.raises(ValueError,match='local readonly'):audio_mirror({})


@pytest.mark.parametrize('fault',[None,'tensor','unknown','conflict'])
def test_inputs_are_audited_before_entering_inference_macro(fault):
    from types import SimpleNamespace
    from multimodal_jev.context_candidate_task import compile_candidates
    from multimodal_jev.slurp_training import content_hash
    from tools.reports.context_reference_input_evidence import AuditedPreparer
    request=compile_candidates('/audio.wav',[dict(id='a',definition='first'),dict(id='b',definition='second')],[],semantics='exclusive_event')
    value=SimpleNamespace(inputs=dict(input_ids=torch.tensor([[0,9,9]])),positions=torch.tensor([[1,2]]))
    row=dict(request_sha256=content_hash(request.model_input()),audio_sha256='a'*64,
        tensors={k:tensor_identity(v) for k,v in dict(value.inputs,positions=value.positions).items()})
    evidence={'sample':row};calls=[]
    def prepare(job):calls.append('prepared');return value
    if fault=='conflict':
        evidence['duplicate']=copy.deepcopy(row);evidence['duplicate']['tensors']['input_ids']['sha256']='bad'
        with pytest.raises(ValueError,match='conflicting'):AuditedPreparer(prepare,evidence)
        assert calls==[];return
    wrapped=AuditedPreparer(prepare,evidence)
    if fault=='tensor':value.inputs['input_ids'][0,0]=1
    audio='b'*64 if fault=='unknown' else 'a'*64
    if fault:
        with pytest.raises(ValueError):wrapped((request,audio))
        assert calls==([] if fault=='unknown' else ['prepared'])
    else:assert wrapped((request,audio)) is value
