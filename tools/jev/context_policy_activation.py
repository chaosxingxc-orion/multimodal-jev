"""Candidate head/marker transaction for a quiescent vLLM context worker.

The serving owner must block requests, activate the matching native LoRA and
verify the complete policy before reopening. This helper alone does not certify
LoRA activation or continuous RL admission. No frozen Stage1 code is modified.
"""
import json
from pathlib import Path

import torch

from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter


def prepare_switch(current, *, snapshot, policy_id, adapter, base_id):
    """Authenticate everything on CPU before changing any live worker tensor."""
    next_bundle=read_execution_bundle(snapshot,expected_policy_id=policy_id,
        expected_contract=current['contract'],expected_base_id=base_id)
    verify_native_adapter(adapter,next_bundle)
    if not torch.equal(current['tensors']['decision_token_row'],next_bundle['tensors']['decision_token_row']):
        raise ValueError('frozen marker cannot change across activation')
    def processor_assets(bundle):
        path=Path(bundle['processor']).parent/'MANIFEST.json'
        files=json.loads(path.read_text())['files']
        return {k:v for k,v in files.items() if k.startswith('processor/')}
    if processor_assets(current)!=processor_assets(next_bundle):
        raise ValueError('resident tokenizer/processor cannot change')
    return next_bundle


class HeadActivation:
    """Two-phase, failed-closed head replacement; owner serializes all requests.

    readout.policy_id becomes an invalid serving identity before tensor mutation.
    Activation completes only after the owner supplies a matching native adapter
    receipt. Receipt authentication is the worker/owner's responsibility, never
    a client-supplied policy ID. Any exception requires closing this worker.
    """
    def __init__(self,readout,embedding,current):
        self.readout=readout;self.embedding=embedding;self.current=current
        self.phase='ready';self.pending=None;self.generation=0
        self._check_live(current)

    def _check_live(self,bundle):
        token=bundle['contract']['token']['id']
        if self.readout.decision_token_id!=token or self.readout.policy_id!=bundle['policy_id']:
            raise ValueError('live readout identity differs')
        expected={k.removeprefix('decision_head.'):v for k,v in bundle['tensors'].items() if k.startswith('decision_head.')}
        actual=self.readout.head.state_dict()
        if actual.keys()!=expected.keys() or any(not torch.equal(v.detach().cpu(),expected[k]) for k,v in actual.items()):
            raise ValueError('live head differs from current policy')
        if not torch.equal(self.embedding[token].detach().cpu(),bundle['tensors']['decision_token_row']):
            raise ValueError('live marker differs from fixed policy')

    def stage(self,bundle,*,expected_policy_id,slot):
        if self.phase!='ready':raise ValueError('ready quiescent transaction required')
        self.phase='validating'
        try:
            if (expected_policy_id!=self.current['policy_id'] or type(slot) is not int or slot<=0 or
                    bundle['contract']!=self.current['contract'] or
                    not torch.equal(bundle['tensors']['decision_token_row'],self.current['tensors']['decision_token_row'])):
                raise ValueError('current identity, compatible contract and frozen marker required')
            self._check_live(self.current)
            head={k.removeprefix('decision_head.'):v for k,v in bundle['tensors'].items() if k.startswith('decision_head.')}
            live=self.readout.head.state_dict()
            if head.keys()!=live.keys() or any(v.shape!=live[k].shape or v.dtype!=torch.bfloat16 or
                                                not torch.isfinite(v).all() for k,v in head.items()):
                raise ValueError('complete finite BF16 head required')
            self.readout.policy_id='activation-in-progress'
            self.readout.head.load_state_dict(head,strict=True)
            if any(not torch.equal(v.detach().cpu(),head[k]) for k,v in self.readout.head.state_dict().items()):
                raise ValueError('head readback differs')
            self.pending=(bundle,slot);self.phase='staged'
            return dict(previous_policy_id=expected_policy_id,next_policy_id=bundle['policy_id'],slot=slot,
                        head_exact=True,serving_allowed=False)
        except BaseException:
            self.phase='failed';self.readout.policy_id='activation-failed';raise

    def commit(self,receipt):
        if self.phase!='staged':raise ValueError('staged head required')
        try:
            bundle,slot=self.pending
            if receipt!=dict(policy_id=bundle['policy_id'],slot=slot,adapter_verified=True):
                raise ValueError('matching worker native-adapter verification required')
            self.readout.policy_id=bundle['policy_id']
            self._check_live(bundle)
            self.current=bundle;self.pending=None;self.generation+=1;self.phase='ready'
            return dict(policy_id=bundle['policy_id'],generation=self.generation,slot=slot,head_marker_exact=True,
                        scope='Head/marker transaction only; end-to-end probabilities must be verified separately')
        except BaseException:
            self.phase='failed';self.readout.policy_id='activation-failed';raise
