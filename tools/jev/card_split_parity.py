"""Independent inference replay of the synthetic actor's pre-update logits."""
import hashlib
import json
from pathlib import Path

import torch


def digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


class Witness:
    def __init__(self, spec, rows, c, base, processor, model, case):
        from multimodal_jev.qwen3_asr_provider import Qwen3ASR
        self.c, self.spec = c, spec
        self.ids = spec['witness_indices']
        if len(set(self.ids)) != len(self.ids) or not all(0 <= i < len(rows) for i in self.ids):
            raise ValueError('witness indices outside probe')
        self.root = Path(spec['scratch'])
        self.root.mkdir(exist_ok=False)
        # torch.save preserves shared recording storage but not repeated Python object identity
        # across sliced tensors; the uid explicitly identifies siblings on replay.
        torch.save([rows[i] for i in self.ids], self.root/'rows.pt')
        self.data = dict(case=case, collation=Qwen3ASR.collation(base, processor, model),
                         snapshots={}, outputs={}, steps=[])

    def snapshot(self, i, wrapped, model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from multimodal_jev.context_fsdp_export import canonical_execution
        from multimodal_jev.context_checkpoint_policy import export_checkpoint_policy
        c = self.c
        if i == 0:
            self.data['snapshots'][i] = [c['source_policy'],c['source_policy_id']]
            return
        path = self.root/f'policy-{i}'
        with FSDP.summon_full_params(wrapped,writeback=False):
            tensors = canonical_execution({k:p.detach().cpu().bfloat16().clone() for k,p in
                                           model.named_parameters() if p.requires_grad})
        receipt = export_checkpoint_policy(c['source_policy'],path,tensors,policy_id=c['source_policy_id'],
                                           contract=c['source_contract'],base_id=c['base_id'])
        self.data['snapshots'][i] = [str(path),receipt['policy_id']]

    def observe(self,i,logits,loss,norm,wrapped):
        self.data['steps'].append(dict(logits=digest(logits), loss=loss.item(), norm=norm.item(),
            updated_parameters={n:digest(p) for n,p in wrapped.named_parameters() if p.requires_grad}))
        if i in self.spec['check_updates']:
            self.data['outputs'][i] = logits.detach().float().flatten().cpu()[self.ids].tolist()

    def finish(self):
        (self.root/'actor.json').write_text(json.dumps(self.data))
        print('ACTOR_WITNESS '+json.dumps(self.data),flush=True)


def replay(config, prior):
    import torch_npu
    from multimodal_jev.qwen3_asr_provider import Qwen3ASR
    from tools.reports.context_transfer_macro_parity import load_native_reference
    from tools.reports.evaluate_single_cards import SingleCardScorer
    from multimodal_jev.context_policy_snapshot import load_policy
    from multimodal_jev.baseline import file_hash
    spec=config['paired_check']
    root=Path(spec['scratch'])
    d=json.loads((root/'actor.json').read_text())
    c=json.loads(Path(prior).read_text())
    rows=torch.load(root/'rows.pt',map_location='cpu',weights_only=True)
    torch.set_num_threads(config['threads'])
    torch.npu.set_device(0)
    torch.use_deterministic_algorithms(True)
    provider=Qwen3ASR()
    provider.initialize_actor_runtime()
    device=torch.device('npu',0)
    source=dict(c,policy_id=c['source_policy_id'],snapshot=c['source_policy'],contract=c['source_contract'])
    model,_,_,handle=load_native_reference(source,source,root,device)
    settings=json.loads(Path(spec['inference_profile']).read_bytes())['macros']['CARD_PAIR_MODES']
    cards=json.loads(Path(c['single_card']['path']).read_bytes())
    out=[]
    for i,(path,pid) in d['snapshots'].items():
        load_policy(path,model,expected_policy_id=pid,expected_base_id=c['base_id'],expected_contract=c['source_contract'])
        for row in rows:
            row['policy_id']=pid
        expected=torch.tensor(d['outputs'][i])
        for panel in spec['inference_panels']:
            mode=settings[panel]
            opts=dict(mode=mode) if isinstance(mode,str) else mode
            scorer=SingleCardScorer(dict(cards=c['single_card'],encoding={'max_tokens':max(config['audio_tokens'],key=int)},
                         pair_batch=len(rows),**opts),None,cards['labels'],d['collation'])
            with torch.inference_mode():
                fn=scorer.evaluation_forward(model,device,provider,pid)
                for order in spec['replay_groups']:
                    actual=fn([rows[k] for k in order]).float().flatten().cpu()
                    delta=actual-expected[order]
                    prob=actual.sigmoid()-expected[order].sigmoid()
                    out.append(dict(step=int(i),panel=panel,order=order,logit_mismatches=delta.count_nonzero().item(),
                        probability_mismatches=prob.count_nonzero().item(),max_logit=delta.abs().max().item(),
                        max_probability=prob.abs().max().item()))
    print('INFERENCE_WITNESS '+json.dumps(dict(rows=out,profile_sha256=file_hash(spec['inference_profile']))),flush=True)
    return all(r['max_logit']==0 and r['max_probability']==0 for r in out)


if __name__=='__main__':
    import sys
    sys.exit(0 if replay(json.loads(Path(sys.argv[1]).read_bytes()),sys.argv[2]) else 2)
