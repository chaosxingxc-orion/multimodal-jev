"""Adapt authenticated held-out inputs to the shared fixed-panel macro API.

No gold, reference transcript, weight update, target-driven prompt selection or
inference. Native membership only; this does not admit semantic-presence labels.
"""
import argparse
import json
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_decision_contract import ApplicabilityQuestion,ContextDecisionRequest,render_context_request
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.span_request_identity import tensor_identity
from tools.reports.context_selection_audited_cache import encoder


def normalize(rows,receipts,requests,candidate_ids,*,expected_count,decisions):
    if (len(rows)!=expected_count or len(receipts)!=expected_count or len(requests)!=expected_count
            or len({r['id'] for r in rows})!=expected_count or len(candidate_ids)!=decisions
            or len(set(candidate_ids))!=decisions):
        raise ValueError('complete unique fixed target population and menu required')
    population=[];independent=[];common=None
    for row,prior,request in zip(rows,receipts,requests,strict=True):
        asset={k:row[k] for k in ('id','audio_sha256','request_sha256')}
        if (request.version!='jev-context-exclusive_event-v1'
                or content_hash(request.model_input())!=asset['request_sha256']
                or any(prior[k]!=v for k,v in asset.items())
                or [q.id for q in request.questions]!=candidate_ids):
            raise ValueError('target request/audio/candidate identity differs')
        system,text=render_context_request(request)
        if content_hash(dict(system=system,text=text))!=row['rendered_sha256']:
            raise ValueError('rendered target input differs')
        # This track has one fixed panel; a per-query menu requires a separate task.
        panel=dict(questions=request.model_input()['questions'],requirements=request.model_input()['requirements'],
                   history=request.model_input()['history'],version=request.version)
        if common is None:common=panel
        if panel!=common:raise ValueError('fixed target panel may not depend on the query')
        if (len(prior['positions'])!=1 or len(prior['positions'][0])!=decisions
                or any(type(x) is not int for x in prior['positions'][0])
                or set(prior['tensors'])!={'input_ids','attention_mask','input_features','input_features_mask','positions'}
                or tensor_identity(torch.tensor(prior['positions'],dtype=torch.long))!=prior['tensors']['positions']):
            raise ValueError('actual complete marker positions must match independent tensor evidence')
        item=dict(asset,candidate_ids=candidate_ids,positions=prior['positions'][0],
                  inputs={k:v for k,v in prior['tensors'].items() if k!='positions'})
        if 'audio_receipt' in prior:item['audio_receipt']=prior['audio_receipt']
        population.append(asset);independent.append(item)
    return population,independent


def load(config):
    dataset=config['dataset'];c=config['inputs']
    if dataset=='FSC_native31':
        from tools.jev.infer_fsc_context_transfer import load_inputs,read_bound
        rows,receipts,proof=load_inputs(c)
        candidates=read_bound(c['input_manifest'])['candidate_ids']
        requests=[]
        for row in rows:
            raw=row['request']
            requests.append(ContextDecisionRequest(raw['audio'],tuple(ApplicabilityQuestion(**q) for q in raw['questions']),
                tuple(raw['requirements']),tuple(raw['history']),version=raw['version']))
        processor=proof['manifest']['processor_sha256'];count=3793;decisions=31
    elif dataset=='MINDS_native14':
        from tools.reports.minds_context_runtime import load_inputs
        from tools.reports.prepare_minds_context_inputs import reconstruct
        rows,receipts,definitions,manifest=load_inputs(c)
        candidates=[v['id'] for v in definitions['candidates']]
        requests=[reconstruct(row,definitions) for row in rows]
        processor=manifest['processor_sha256'];count=8168;decisions=14
    elif dataset=='SLURP_clean_TEST60':
        from tools.reports.context_slurp_test_macro_inputs import load_inputs
        rows,receipts,requests,candidates,processor=load_inputs(config)
        count=12991;decisions=60
    else:raise ValueError('explicit frozen native FSC, MINDS or clean SLURP TEST task required')
    population,independent=normalize(rows,receipts,requests,candidates,expected_count=count,decisions=decisions)
    return dict(population=population,candidate_ids=candidates,input_receipts=independent,
        requests={r['id']:request for r,request in zip(rows,requests,strict=True)},cache_receipts=receipts,
        processor_sha256=processor,cache_root=c['cache_root'],dataset=dataset,source_rows=rows)


def audited_encoder(config,admitted,*,snapshot,decision_token_id,audio_token_id):
    from multimodal_jev.cached_context_preparation import processor_cache_identity
    processor=processor_cache_identity(snapshot)
    if processor!=admitted['processor_sha256']:raise ValueError('source processor differs from target input audit')
    encoded=encoder(admitted['cache_root'],admitted['requests'],admitted['candidate_ids'],admitted['cache_receipts'],
        processor_sha256=processor,decision_token_id=decision_token_id,audio_token_id=audio_token_id)
    expected={r['id']:r for r in admitted['input_receipts']}
    def checked(asset):
        row,receipt=encoded(asset);prior=expected[asset['id']]
        if any(receipt[k]!=v for k,v in prior.items()):raise ValueError('actual transfer inputs differ from independent target receipt')
        # FSC's old input audit has no audio_receipt payload; do not invent an
        # independent waveform receipt from the current inference run.
        return row,dict(prior)
    return checked


def prepare(config,provenance):
    admitted=load(config)
    root=Path(config['output']);check_paths({'transfer_inputs':root});root.mkdir(parents=True,exist_ok=False)
    for key,name in [('population','population.json'),('candidate_ids','candidate-ids.json'),('input_receipts','input-receipts.json')]:
        commit_json(root/name,admitted[key])
    report=dict(dataset=config['dataset'],recordings=len(admitted['population']),decisions=len(admitted['candidate_ids']),
        processor_sha256=admitted['processor_sha256'],provenance=provenance,
        artifacts={p.name:file_hash(p) for p in root.iterdir() if p.is_file()},
        training_dispatch_allowed=False,evaluation_dispatch_allowed=False,
        scope='Full existing input/processor evidence adapted to fixed-panel macro interface; no actual cache reread, model inference, target score or policy selection')
    if config['dataset']=='SLURP_clean_TEST60':
        report.update(source_release=config['source_release'],evaluation_kind='source_domain_held_out')
    commit_json(root/'MANIFEST.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    print(json.dumps(prepare(config,provenance),indent=2))
