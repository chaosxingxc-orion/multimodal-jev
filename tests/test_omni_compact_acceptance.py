import json
from pathlib import Path

import pytest
import torch

from multimodal_jev import omni_compact_acceptance as gate
from multimodal_jev.baseline import file_hash


def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))


def fixture(tmp,monkeypatch,delta=False):
    # Synthetic six-row success envelope; no model forward or actual container.
    root=tmp/'candidate';run=root/'run';run.mkdir(parents=True);prior=tmp/'precision/run';prior.mkdir(parents=True)
    source=root/'source';pins={}
    for n in ('tools/jev/diagnose_omni_compact_keys.py','src/multimodal_jev/omni_compact_attention.py'):
        p=source/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture');pins[n]=file_hash(p)
    original=dict(keys=[[str(i),'present'] for i in range(6)],response_ids=[1,3],model_files=['fixture'],prompt_width=1228)
    put(prior/'identity.json',original);monkeypatch.setattr(gate,'PRECISION_IDENTITY',file_hash(prior/'identity.json'))
    argv=['docker','create',gate.IMAGE,'python','fixture','--precision-root',str(prior.parent)]
    plan=dict(argv=argv,image=gate.IMAGE,source_commit='fixture',source_root=str(source),source_files=pins)
    put(root/'intent.json',plan);monkeypatch.setattr(gate,'INTENT',file_hash(root/'intent.json'))
    put(root/'launch.json',dict(container_id=gate.CONTAINER,image=gate.IMAGE,intent_sha256=gate.INTENT,source_commit='fixture'))
    state=dict(Id=gate.CONTAINER,Image=gate.IMAGE,State=dict(Status='exited',ExitCode=0,OOMKilled=False),
        Config=dict(Cmd=argv[3:],Env=[]),HostConfig=dict(NanoCpus=16*10**9,Memory=160*1024**3,NetworkMode='none',Devices=[],Privileged=False,Binds=[]))
    inspection=tmp/'inspection.json';put(inspection,state)
    identity=dict(**original,padding_counts=[154,252,0,251,519,402],precision_identity_sha256=gate.PRECISION_IDENTITY,
        response_logp_max_error_threshold=.1,preflight_only=False,targets_read=False,weight_updates=0,
        runner_sha256=pins['tools/jev/diagnose_omni_compact_keys.py'],adapter_sha256=pins['src/multimodal_jev/omni_compact_attention.py'])
    put(run/'identity.json',identity);put(run/'terminal.json',dict(exit_code=0));reports=[]
    labels=torch.tensor(original['response_ids']).unsqueeze(-1)
    for row,padding in enumerate(identity['padding_counts']):
        ref=torch.zeros((2,4),dtype=torch.bfloat16);candidate=ref.clone()
        if delta and row==4:candidate[0,1]=3
        p=run/f'row-{row:02d}-logits.pt';torch.save(candidate,p)
        logp=candidate.float().log_softmax(-1).gather(-1,labels).squeeze(-1)
        refp=ref.float().log_softmax(-1).gather(-1,labels).squeeze(-1)
        comparisons={}
        for cell in ('audio_padding','text_padding'):
            path=prior/f'row-{row:02d}-{cell}-logits.pt';torch.save(ref,path);pin=file_hash(path)
            put(prior/f'row-{row:02d}-{cell}.json',dict(logits_sha256=pin))
            comparisons[cell]=dict(reference_sha256=pin,exact_logits=bool(torch.equal(ref,candidate)),
                max_logits_error=float((ref.float()-candidate.float()).abs().max()),max_response_logp_error=float((refp-logp).abs().max()))
        events=[]
        for layer in range(48):
            e=dict(layer=layer,mode='compact_masked_keys' if padding else 'unchanged_no_mask',query_length=1230,key_lengths=[1230-padding])
            if padding:e['original_key_length']=1230
            events.append(e)
        r=dict(row=row,key=original['keys'][row],padding=padding,valid_tokens=1230-padding,attention_records=events,
            response_gate_passed=True,logits_sha256=file_hash(p),fp32_log_probs=logp.tolist(),comparisons=comparisons)
        put(run/f'row-{row:02d}.json',r);reports.append(r)
    put(run/'result.json',dict(status='diagnostic_complete',all_six_response_gates_passed=True,threshold=.1,weight_updates=0,reports=reports))
    return root,inspection


@pytest.mark.parametrize('fault',[None,'running','failed','oom','resource','command','source','partial','threshold','updates','layer','logits','recomputed_failure'])
def test_candidate_requires_complete_success_and_recomputed_probabilities(tmp_path,monkeypatch,fault):
    root,inspection=fixture(tmp_path,monkeypatch,delta=fault=='recomputed_failure')
    def edit(path,fn):
        v=json.loads(path.read_text());fn(v);put(path,v)
    if fault=='running':edit(inspection,lambda x:x['State'].update(Status='running'))
    if fault=='failed':edit(inspection,lambda x:x['State'].update(ExitCode=1))
    if fault=='oom':edit(inspection,lambda x:x['State'].update(OOMKilled=True))
    if fault=='resource':edit(inspection,lambda x:x['HostConfig'].update(Memory=1))
    if fault=='command':edit(inspection,lambda x:x['Config']['Cmd'].append('--preflight-only'))
    if fault=='source':(root/'source/src/multimodal_jev/omni_compact_attention.py').write_text('changed')
    if fault=='partial':edit(root/'run/result.json',lambda x:x['reports'].pop())
    if fault=='threshold':edit(root/'run/result.json',lambda x:x.update(threshold=1.))
    if fault=='updates':edit(root/'run/result.json',lambda x:x.update(weight_updates=1))
    if fault=='layer':
        edit(root/'run/row-00.json',lambda x:x['attention_records'].pop())
        edit(root/'run/result.json',lambda x:x['reports'][0]['attention_records'].pop())
    if fault=='logits':torch.save(torch.ones(2,4),root/'run/row-00-logits.pt')
    if fault:
        with pytest.raises(ValueError):gate.require_candidate(root,inspection,file_hash(inspection))
    else:
        r=gate.require_candidate(root,inspection,file_hash(inspection))
        assert r['rows']==6 and r['recomputed_reference_errors']==[0.]*6
