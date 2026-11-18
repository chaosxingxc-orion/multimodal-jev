"""Compact native-attribute inputs and late supervision/rollout binding.

Candidate only; frozen preparer and online assembly remain unchanged. Cached
inputs are shared by Q-input/Q-credit, while actor contracts remain distinct.
"""
import copy
import hashlib
from pathlib import Path
import torch
from multimodal_jev.audio_policy import load_audio
from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_preparation import CachedContextPreparer, PreparedContextInput
from multimodal_jev.encoded_audio_cache import canonical
from tools.jev.context_online_batches import OnlineContextBatches
from tools.reports.context_attribute_compact_candidate import encode
from tools.reports.build_context_attribute_candidate import request, digest
from tools.reports.context_attribute_actor_candidate import generate
from tools.reports.context_attribute_reward_candidate import validate_truth

COMPILER_SHA='d875d0638fa44f5ba083d6a160b45c48ebb38ee9e8ad5c2e1b8ffe4e9a29509f'
PREPARER_SHA='26744b067a0c0600e375e326b0175ae3ed43996d171413a4ea6e57f6e14033f0'
ONLINE_SHA='2419cb61f2ecab37c0a06082c1d9bfa59eae5a18df5f67c769b8261985c9f1f4'


class AttributeInputs(CachedContextPreparer):
    def __init__(self, menu, arm, *, source_root, **kwargs):
        root=Path(source_root)
        for name,h in [('src/multimodal_jev/cached_context_preparation.py',PREPARER_SHA),
                       ('tools/jev/context_online_batches.py',ONLINE_SHA),
                       ('tools/reports/context_attribute_compact_candidate.py',COMPILER_SHA)]:
            if file_hash(root/name)!=h:raise ValueError('Input runtime source changed: '+name)
        _,self.contract=generate(menu,arm,source_root=root)
        self.menu=copy.deepcopy(menu);self.arm=arm
        super().__init__(**kwargs)
        if self.encoding!=dict(decision_token_id=151706,audio_token_id=151676,max_tokens=4096) or self.preprocessing is not None:
            raise ValueError('Frozen attribute encoding and unaugmented audio required')

    def identity(self, req, expected_audio_sha256):
        expected=request(req.audio,self.menu,arm=self.arm)
        if req.model_input()!=expected.model_input():raise ValueError('Request differs from fixed attribute template')
        value=super().identity(req,expected_audio_sha256)
        value['context_encoding'].update(version='attribute-shared-question-indexed-membership-cache-v1',
            compiler_sha256=COMPILER_SHA,catalog_sha256=self.contract['catalog_sha256'])
        # Reward arm/policy/targets deliberately absent: Q-input and Q-credit
        # have the same complete encoded request and can share only inputs.
        return value

    def __call__(self, job):
        req,audio_sha=job
        identity=self.identity(req,audio_sha)
        path=req.audio if self.audio_mirror is None else self.audio_mirror.resolve(audio_sha)
        if self.audio_mirror is None and file_hash(path)!=audio_sha:raise ValueError('Source audio changed')
        value=self.cache.read(identity);hit=value is not None
        if value is None:
            if not hasattr(self.local,'processor'):self.local.processor=self.processor_factory()
            wave,receipt=load_audio(path,None,return_receipt=True)
            inputs,positions=encode(self.local.processor,req,wave,**self.encoding)
            inputs={k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in inputs.items()}
            self.cache.write(identity,inputs,positions,receipt)
        else:inputs,positions,receipt=value
        self.validate_encoded(req,inputs,positions)
        return PreparedContextInput(canonical(req.model_input()),inputs,positions,receipt,hit)

    def validate_encoded(self, req, inputs, positions):
        self.cache._validate(inputs,positions)
        ids=inputs['input_ids'];markers=(ids[0]==151706).nonzero().flatten()
        audio=(ids[0]==151676).nonzero().flatten()
        if (ids.shape[-1]>4096 or positions.shape!=(1,len(req.questions))
                or not torch.equal(positions[0],markers) or not len(audio)
                or audio[-1]>=positions[0,0] or (ids==151705).any()
                or inputs['input_features'].dtype!=torch.bfloat16):
            raise ValueError('Complete bounded attribute encoding required')

    def actor_item(self, asset, target, prepared, *, policy_id, slot_id, padding=False):
        """Attach separately authenticated targets after label-free preparation.

        Caller validates complete source manifests and the full data schedule;
        this function checks row bindings and projection consistency.
        """
        if (set(asset)!={'id','audio','audio_sha256','catalog_sha256'}
                or asset['catalog_sha256']!=self.contract['catalog_sha256']
                or target['id']!=asset['id'] or target['input_sha256']!=digest(asset)
                or type(padding) is not bool or not isinstance(policy_id,str) or not policy_id
                or not isinstance(slot_id,str) or not slot_id):
            raise ValueError('Authenticated input/target and explicit scheduled policy slot required')
        req=request(asset['audio'],self.menu,arm=self.arm)
        if prepared.request!=canonical(req.model_input()):raise ValueError('Prepared request differs')
        self.validate_encoded(req,prepared.inputs,prepared.positions)
        ids=[n['id'] for n in self.menu['nodes']]
        if (target['candidate_ids']!=ids or len(target['labels'])!=len(ids)
                or len(target['supervised'])!=len(ids)
                or any(type(v) is not bool for v in target['labels']+target['supervised'])):
            raise ValueError('Exact complete Boolean target catalog required')
        labels=torch.tensor([target['labels']],dtype=torch.bool)
        observed=torch.tensor([target['supervised']],dtype=torch.bool)
        validate_truth(labels,observed,self.menu,'q_input')
        indices=[ids.index(k) for k in self.contract['decision_ids']]
        return dict(request_id=slot_id,policy_id=policy_id,inputs=prepared.inputs,positions=prepared.positions,
            labels=labels[:,indices],observed=observed[:,indices],real_request=not padding,
            request_weight=0. if padding else 1.,attribute_contract_id=self.contract['contract_id'],
            decision_ids=list(self.contract['decision_ids']))


class AttributeOnlineBatches(OnlineContextBatches):
    def __init__(self, input_adapter, **kwargs):
        self.adapter=input_adapter
        super().__init__(**kwargs)
        c=input_adapter.contract
        if (self.group_size!=16 or self.total_updates!=634 or self.audio_token_id!=151676
                or self.reference_policy_id!=c['reference_policy_id']):
            raise ValueError('Frozen attribute rollout budget and reference required')

    def prepare(self, items, current_engine, *, policy_id, update):
        try:
            for row in items:
                c=self.adapter.contract
                if row.get('attribute_contract_id')!=c['contract_id'] or row.get('decision_ids')!=c['decision_ids']:
                    raise ValueError('Rollout attribute contract or node order changed')
                validate_truth(row['labels'],row['observed'],self.adapter.menu,self.adapter.arm)
            prepared,options,receipts=super().prepare(items,current_engine,policy_id=policy_id,update=update)
            options['attribute_contract_id']=self.adapter.contract['contract_id']
            for receipt in receipts:
                receipt['attribute_contract_id']=self.adapter.contract['contract_id']
                receipt['decision_ids']=list(self.adapter.contract['decision_ids'])
            return prepared,options,receipts
        except BaseException:
            self.fail_update()
            raise
