"""Finite dual-credit integration into the exact existing checkpoint trainer.

Preserves its NPU/FSDP, optimizer, checkpoint, export, activation and recovery
loop. Not a formal release or launcher; original source files stay immutable.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

SOURCE_SHA='7bb1b2259bd799907856455cf75475280dd0bbb3998f6a928c03595141e73288'


def generate(source_root):
    p=Path(source_root)/'tools/jev/train_context_online_candidate.py';raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('original trainer source changed')
    source=raw.decode()
    replacements=[
        ("        named=[(n,p) for n,p in wrapped.named_parameters() if p.requires_grad]",
         "        named=[(n,p) for n,p in wrapped.named_parameters() if p.requires_grad]\n"
         "        from multimodal_jev.context_lora_checkpoint_scope import validate_trainable_scope\n"
         "        atomic_json(out/'checkpoint-scope.json',validate_trainable_scope(named))"),
        ('    from multimodal_jev.context_native_actor import MODE,build_actor,make_batch',
         "    from tools.reports.context_dual_actor_candidate import load as load_dual_actor\n"
         "    dual_actor=load_dual_actor(c['source_root'])\n"
         "    MODE,build_actor,make_batch=dual_actor.MODE,dual_actor.build_actor,dual_actor.make_batch"),
        ("        contract.update(optimizer=dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.),\n            objective=dict(group_size=16,reward='complete_candidate_set_f1',kl=.01,auxiliary='cosine_0.3_to_0.1_first_80pct'),grad_clip=1.,world_size=4)",
         "        from tools.reports.context_dual_runtime_candidate import verify_runtime\n"
         "        from tools.reports.context_dual_actor_candidate import options_check\n"
         "        runtime_identity=verify_runtime(c)\n"
         "        options=dict(algorithm=c['algorithm'],arm=c['dual_arm'],ranking_coefficient=.1,probability_coefficient=.3,kl_coefficient=.01,reference_policy_id=c['source_policy_id'])\n"
         "        options_check(options)\n"
         "        contract.update(dual_objective=options,dual_runtime_identity=runtime_identity,\n"
         "            optimizer=dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.),\n"
         "            objective=dict(group_size=16,reward='component_binary_correctness',kl=.01,auxiliary='constant_pair0.1_nll0.3'),grad_clip=1.,world_size=4)"),
        ("        options=dict(stage='task',ranking_coefficient=.3,probability_coefficient=.3,algorithm=c['algorithm'],\n            kl_coefficient=.01,reference_policy_id=c['source_policy_id'])",''),
        ('        from context_rollout_rpc import ContextEngineClient\n        from context_rollout_session import ContextRolloutSession\n        from context_reference_cache import ReferenceCache',
         '        from tools.reports.context_dual_runtime_candidate import BoundDualClient\n'
         '        from tools.reports.context_dual_session_candidate import DualContextRolloutSession,RuntimeReferenceCache,bind_candidates'),
        ("                cache=ReferenceCache(c['reference_cache'],policy_id=c['source_policy_id'],\n                    base_id=c['base_id'],contract=c['source_contract'])\n                client=ContextEngineClient(c['socket_path'],c['exchange'],bundle=current)",
         "                cache=RuntimeReferenceCache(c['reference_cache'],policy_id=c['source_policy_id'],\n"
         "                    base_id=c['base_id'],contract=c['source_contract'],runtime_identity=runtime_identity)\n"
         "                client=BoundDualClient(c['socket_path'],c['exchange'],bundle=current,runtime_identity=runtime_identity)"),
        ("                session=ContextRolloutSession(client,cache,reference=reference,initial_policy=initial,\n                    audio_token_id=base.config.audio_token_id,algorithm=c['algorithm'],start_update=start_update)",
         "                candidate_ids=[v['id'] for v in task['schema']['candidates']]\n"
         "                session=DualContextRolloutSession(client,cache,reference=reference,initial_policy=initial,\n"
         "                    runtime_identity=runtime_identity,candidate_ids=candidate_ids,arm=c['dual_arm'],\n"
         "                    audio_token_id=base.config.audio_token_id,algorithm=c['algorithm'],start_update=start_update)"),
        ("                        items,metadata=stream.consume(index,cfg.jev_policy_id)\n                        items,options,rollout_receipt=session.prepare(items,update=index)",
         "                        items,metadata=stream.consume(index,cfg.jev_policy_id)\n"
         "                        items=bind_candidates(items,metadata,candidate_ids)\n"
         "                        items,options,rollout_receipt=session.prepare(items,update=index)"),
        ("('request_id','policy_id','labels','observed','behavior','actions','old_joint','advantages','rewards','reference_policy','request_weight','real_request')",
         "('request_id','policy_id','labels','observed','candidate_ids','dual_credit','dual_identity','dual_options','reference_probability','reference_policy_id','request_weight','real_request','global_rows','global_real_requests')")]
    for before,after in replacements:
        if source.count(before)!=1:raise ValueError('trainer integration anchor changed: '+before[:80])
        source=source.replace(before,after)
    ast.parse(source)
    return source


def run(config,output):
    source=generate(config['source_root'])
    namespace={'__name__':__name__+'.derived'}
    exec(compile(source,'<dual-checkpoint-trainer>','exec'),namespace)
    return namespace['run'](config,Path(output))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);run(json.loads(a.config.read_text()),a.output)
