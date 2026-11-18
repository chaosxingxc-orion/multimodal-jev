"""Formal source candidate over the admitted replica trainer; no dispatch entry.

The evaluation window is derived from the declared topology and schedule, not
hard-coded per width: stop is the authenticated total update count and the
recovery/evaluation curve is the declared interval over that window. Every
optimizer update exports a policy before the dual-replica activation ack;
checkpoint states are saved only on the pre-registered recovery steps.
"""
import ast
from collections import Counter
from pathlib import Path

from multimodal_jev.context_topology_batch_plan import derive
from tools.reports.context_replica_trainer_candidate import generate as replica_source


def window(plan, steps):
    """Pre-registered evaluation/recovery curve for a declared topology.

    The curve is experiment data, not a per-width constant: it must start at 0,
    end at the authenticated update total and increase strictly."""
    total = plan['total_updates']
    if (not isinstance(steps, list) or len(steps) < 2
            or any(type(s) is not int for s in steps)
            or steps[0] != 0 or steps[-1] != total
            or any(b <= a for a, b in zip(steps, steps[1:]))):
        raise ValueError('increasing pre-registered curve from 0 to the declared endpoint required')
    return list(steps)


def contract(config):
    plan = derive(config['batch_config'])
    steps = window(plan, config.get('evaluation_steps'))
    if (config.get('experiment_kind') != 'formal_source'
            or type(config.get('start_update')) is not int or config.get('start_update') != 0
            or type(config.get('stop_update')) is not int
            or config.get('stop_update') != plan['total_updates']
            or config.get('stage') != 'task' or config.get('resume_root')
            or config.get('engineering_prefix') is not False
            or config.get('training_dispatch_allowed') is not True
            or any(type(s) is not int for s in config['evaluation_steps'])
            or config.get('seed') != 317 or type(config.get('seed')) is not int
            or config.get('algorithm') != 'dual_credit_grpo_floor_clipped_v1'
            or config.get('dual_arm') != 'joint'):
        raise ValueError('fresh formal source-only joint GRPO and declared curve required')
    return dict(experiment_kind='formal_source', updates=plan['total_updates'],
        world_size=plan['dp_replicas'], global_batch=plan['global_batch'],
        updates_per_epoch=plan['updates_per_epoch'],
        tail_real_queries=plan['tail_real_queries'],
        tail_padding_slots=plan['tail_padding_slots'], seed=317,
        evaluation_steps=list(steps),
        recovery_checkpoint_steps=steps[1:],
        behavior_export='every_update_before_dual_replica_ack',
        query_weights='uniform_real_zero_padding', source_selection_only=True)


def validate_schedule(batches, population, plan):
    """Validate the complete declared-epoch schedule for any DP width: real
    samples appear exactly once per epoch, each epoch tail keeps the real
    denominator, and padding slots carry zero weight only."""
    total = plan['total_updates']; per_epoch = plan['updates_per_epoch']
    batch = plan['global_batch']; queries = plan['configuration']['source_queries']
    epochs = plan['configuration']['epochs']
    if len(population) != queries or len(batches) != total:
        raise ValueError('complete source-recording multi-epoch schedule required')
    counts = []
    for epoch in range(epochs):
        seen = Counter()
        for index, entry in enumerate(batches[epoch*per_epoch:(epoch+1)*per_epoch]):
            if (entry['epoch'] != epoch or entry['index'] != index
                    or len(entry['slots']) != batch):
                raise ValueError('fixed epoch and batch cursor required')
            real = 0
            for slot in entry['slots']:
                if (slot['id'] not in population or type(slot['padding']) is not bool
                        or type(slot['weight']) not in (float, int)
                        or slot['weight'] != float(not slot['padding'])):
                    raise ValueError('known samples with uniform real and zero padding weights required')
                if not slot['padding']:
                    real += 1
                    seen[slot['id']] += 1
            expected = plan['tail_real_queries'] if index == per_epoch-1 else batch
            if real != expected:
                raise ValueError('epoch tail real/zero-weight denominator differs')
            counts.append(real)
        if seen != Counter(population):
            raise ValueError('each real source sample must appear exactly once per epoch')
    return counts


def generate(source_root):
    source = replica_source(source_root)
    replacements = [
        ("        from tools.reports.context_macro_trainer_candidate import experiment_window\n"
         "        experiment_window(c)",
         "        from tools.reports.context_replica_formal_candidate import contract as formal_contract,validate_schedule\n"
         "        formal=formal_contract(c)"),
        ("        progress('load')", "        contract['formal_experiment']=formal\n        progress('load')"),
        ("                session=DualContextRolloutSession(client,cache,reference=reference,initial_policy=initial,",
         "                validate_schedule(batches,population.keys(),formal_plan(c))\n"
         "                session=DualContextRolloutSession(client,cache,reference=reference,initial_policy=initial,"),
        ("    from tools.reports.context_replica_trainer_candidate import topology\n"
         "    from multimodal_jev.context_replica_schedule import rank_indices",
         "    from tools.reports.context_replica_trainer_candidate import topology\n"
         "    from multimodal_jev.context_replica_schedule import rank_indices\n"
         "    from tools.reports.context_replica_formal_candidate import plan as formal_plan"),
        ("type(c['checkpoint_interval']) is not int or c['checkpoint_interval']<1",
         "'checkpoint_interval' in c"),
        ("        if c['engineering_prefix'] is not True or c['training_dispatch_allowed'] is not False:\n"
         "            raise ValueError('continuous candidate is engineering-only, not released')",
         "        if c['engineering_prefix'] is not False or c['training_dispatch_allowed'] is not True:\n"
         "            raise ValueError('released formal interval required')"),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('formal replica integration anchor changed: '+before[:90])
        source = source.replace(before, after)
    begin = '                save_now=True\n'
    end = '                activation=[None]\n'
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError('unique export/checkpoint section required')
    left, rest = source.split(begin)
    _, right = rest.split(end)
    source = left + CHECKPOINT_BLOCK + end + right
    ast.parse(source)
    return source


def plan(config):
    """Batch plan from the declaration; shared by the generated trainer."""
    return derive(config['batch_config'])


CHECKPOINT_BLOCK = '''                save_now=(index+1) in formal['recovery_checkpoint_steps']
                if frozen!={n:tensor_identity(p.detach().cpu()) for n,p in wrapped.named_parameters() if not p.requires_grad}:
                    raise ValueError('frozen backbone changed')
                if save_now:
                    state=capture(named,opt,rank=rank,world_size=world,completed_updates=index+1,contract=contract)
                    commit_file(out/f'checkpoint-{index+1:04d}.pt',lambda f:torch.save(state,f))
                # Every optimizer update exports only trainable LoRA/head tensors.
                with FSDP.summon_full_params(wrapped,writeback=False,rank0_only=False):
                    full=canonical_execution({n:p.detach().cpu().to(torch.bfloat16).clone()
                        for n,p in model.named_parameters() if p.requires_grad})
                    hashes={n:tensor_identity(v) for n,v in full.items()};peers=[None]*world
                    dist.all_gather_object(peers,hashes)
                    if any(v!=hashes for v in peers):raise ValueError('assembled execution differs across ranks')
                    if rank==0:
                        commit_file(root/f'execution-{index+1:04d}.pt',lambda f:torch.save(full,f))
                        joint=export_checkpoint_policy(c['source_policy'],root/f'policy-{index+1:04d}',full,
                            policy_id=c['source_policy_id'],contract=c['source_contract'],base_id=c['base_id'])
                dist.barrier()
                if rank==0:
                    published=dict(updates=index+1,contract=contract,
                        execution_file=f'execution-{index+1:04d}.pt',
                        joint_policy=joint,joint_policy_directory=f'policy-{index+1:04d}',
                        joint_policy_manifest_sha256=file_hash(root/f'policy-{index+1:04d}/MANIFEST.json'),
                        execution_sha256=file_hash(root/f'execution-{index+1:04d}.pt'))
                    commit_json(root/f'policy-{index+1:04d}-committed.json',published)
                    if save_now:
                        commit_json(root/f'checkpoint-{index+1:04d}-committed.json',dict(published,
                            rank_shards={f'rank-{r}/checkpoint-{index+1:04d}.pt':
                                file_hash(root/f'rank-{r}/checkpoint-{index+1:04d}.pt') for r in range(world)}))
                dist.barrier()
'''


def run(config, output):
    contract(config)
    namespace = {'__name__': __name__+'.derived'}
    exec(compile(generate(config['source_root']), '<formal-replica-trainer>', 'exec'), namespace)
    return namespace['run'](config, Path(output))


if __name__ == '__main__':
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    run(json.loads(args.config.read_text()),args.output)
