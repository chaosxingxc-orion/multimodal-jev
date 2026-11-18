"""Finite DP2/dual-TP1 GRPO integration over the existing verl trainer.

Historical DP4 sources/schedules remain unchanged. Only rank allocation within
each authenticated optimizer batch changes; population and update order do not.
This entrypoint is engineering-only until actual NPU update/replay acceptance.
"""
import argparse
import ast
import json
from pathlib import Path

from multimodal_jev.span_length_groups import length_group_indices
from tools.reports.context_dual_trainer_candidate import generate as dual_source

ALGORITHM = 'dual_credit_grpo_floor_clipped_v1'


def experiment_window(config):
    """Keep short admission separate from explicitly bounded efficiency runs."""
    start, stop = config['start_update'], config['stop_update']
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop:
        raise ValueError('integer optimizer window required')
    if config.get('experiment_kind', 'engineering_prefix') == 'efficiency':
        interval = config.get('checkpoint_interval')
        if (start != 0 or not 3 <= stop <= 100 or config.get('resume_root')
                or type(interval) is not int or not 1 <= interval <= stop
                or config.get('engineering_prefix') is not True
                or config.get('training_dispatch_allowed') is not False):
            raise ValueError('efficiency requires fresh 3..100 updates and a bounded integer checkpoint cadence')
        return 'efficiency'
    if config.get('experiment_kind', 'engineering_prefix') != 'engineering_prefix' or stop != 2 or start not in (0, 1):
        raise ValueError('short admission requires initial two steps or step2 replay')
    # A step2 replay restores from the step1 shard, so short admission keeps
    # every-update recovery state.
    if config.get('checkpoint_interval', 1) != 1:
        raise ValueError('short admission requires every-update recovery state')
    return 'engineering_prefix'


def topology(config, world, mounted):
    micro = config['microbatch']
    if (world != 2 or config['physical_devices'] != [4, 5]
            or sorted(mounted) != ['davinci4', 'davinci5']
            or config['global_batch'] != 128 or type(micro) is not int or micro not in (8, 16, 32)
            or config['algorithm'] != ALGORITHM or config['dual_arm'] != 'joint'):
        raise ValueError('isolated DP2 cards4/5 global128 micro8/16/32 joint GRPO required')
    client = config['macro_client']
    if (client['max_queries'] != 128 or client['rollout_queries'] != 32
            or client['runtime_identity'] != config['dual_runtime_identity']):
        raise ValueError('matching macro128 rollout32 runtime required')
    return dict(world_size=world, local_batch=128//world, microbatch=micro,
                accumulation=128//world//micro, rollout_queries=32, replicas=2)


def repartition(lengths, frozen_ranks, *, microbatch, world_size):
    # Verify actual encoded lengths against the original authenticated schedule
    # first. Repartition only within this fixed optimizer batch, never resample.
    if (len(lengths) != 128 or world_size != 2 or microbatch not in (8, 16, 32)
            or frozen_ranks != length_group_indices(lengths, microbatch=8,
                world_size=4, order='length_grouped')):
        raise ValueError('original complete DP4 schedule/processor binding differs')
    return length_group_indices(lengths, microbatch=microbatch, world_size=world_size,
                                order='length_grouped')


def integrated(source_root):
    """DP2 integration edits with the frozen unconditional export section intact."""
    source = dual_source(source_root)
    edits = [
        ("        if c['engineering_prefix'] is True and stop_update-start_update>2:\n"
         "            raise ValueError('engineering prefix is at most two updates')",
         "        from tools.reports.context_macro_trainer_candidate import experiment_window\n"
         "        experiment_window(c)"),
        ('    from multimodal_jev.slurp_official_training import load',
         '    from multimodal_jev.slurp_portable_release import load'),
        ("release_sha256=c['release_sha256'],population=population)",
         "release_sha256=c.get('schedule_release_sha256',c['release_sha256']),population=population)"),
        ("publication['release_sha256']!=c['release_sha256']",
         "publication['release_sha256']!=c.get('schedule_release_sha256',c['release_sha256'])"),
        ("    if (world!=4 or c['global_batch']!=128 or c['microbatch']!=8 or\n"
         "            c['runtime_profile']!=PROFILE or c['physical_devices']!=[4,5,6,7] or\n"
         "            sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))!=['davinci4','davinci5','davinci6','davinci7']):\n"
         "        raise ValueError('isolated DP4 global128 micro8 required')",
         "    from tools.reports.context_macro_trainer_candidate import topology,repartition\n"
         "    layout=topology(c,world,[p.name for p in Path('/dev').glob('davinci[0-9]*')])\n"
         "    if c['runtime_profile']!=PROFILE:raise ValueError('fixed actor runtime required')"),
        ('grad_clip=1.,world_size=4)', "grad_clip=1.,world_size=world,macro_layout=layout,schedule_release_sha256=c.get('schedule_release_sha256',c['release_sha256']))"),
        ('ppo_mini_batch_size=32,', "ppo_mini_batch_size=layout['local_batch'],"),
        ('ppo_micro_batch_size_per_gpu=8,', "ppo_micro_batch_size_per_gpu=c['microbatch'],"),
        ("len(commit['rank_shards'])!=4", "len(commit['rank_shards'])!=world"),
        ('rank=rank,world_size=4,', 'rank=rank,world_size=world,'),
        ('[None]*4', '[None]*world'),
        ('for r in range(4)', 'for r in range(world)'),
        ('        from tools.reports.context_dual_runtime_candidate import BoundDualClient',
         '        from tools.reports.context_macro_client import connect_pool'),
        ("                client=BoundDualClient(c['socket_path'],c['exchange'],bundle=current,runtime_identity=runtime_identity)",
         "                client=connect_pool(c['macro_client'],bundle=current)"),
        ("                        planned=batches[index]['rank_indices']\n"
         "                        ids=[i for group in planned for i in group]\n"
         "                        # Verify actual processor shapes reproduce the frozen grouping.\n"
         "                        from multimodal_jev.span_length_groups import length_group_indices\n"
         "                        lengths=[(p['inputs']['input_ids'].shape[-1],p['inputs']['input_features'].shape[-1]) for p in items]\n"
         "                        if planned!=length_group_indices(lengths,microbatch=8,world_size=4,order='length_grouped'):\n"
         "                            raise ValueError('actual processor lengths differ from frozen grouping')",
         "                        lengths=[(p['inputs']['input_ids'].shape[-1],p['inputs']['input_features'].shape[-1]) for p in items]\n"
         "                        planned=repartition(lengths,batches[index]['rank_indices'],microbatch=c['microbatch'],world_size=world)\n"
         "                        ids=[i for group in planned for i in group]"),
        ('parts=list(complete.split(32))', "parts=list(complete.split(layout['local_batch']))"),
    ]
    counts = {'rank=rank,world_size=4,': 2, '[None]*4': 3}
    for before, after in edits:
        if source.count(before) != counts.get(before, 1):
            raise ValueError('macro trainer integration anchor changed: '+before[:90])
        source = source.replace(before, after)
    ast.parse(source)
    return source


def generate(source_root):
    """Apply the declared optimizer-shard cadence; every update still exports a policy.

    Only recovery state (FP32 masters, Adam, rank shards) follows
    ``checkpoint_interval``. Execution export, policy publication, adapter
    activation and the per-update commit receipt stay unconditional so the audit
    chain remains complete at every update.
    """
    source = integrated(source_root)
    begin = '                save_now=True\n'
    end = '                activation=[None]\n'
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError('unique export/checkpoint section required')
    left, rest = source.split(begin)
    _, right = rest.split(end)
    source = left + CHECKPOINT_BLOCK + end + right
    ast.parse(source)
    return source


CHECKPOINT_BLOCK = '''                ckpt_interval=c['checkpoint_interval']
                ckpt_completed=index+1-start_update
                save_now=ckpt_completed%ckpt_interval==0 or index+1==stop_update
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
                    ckpt_published=dict(updates=index+1,contract=contract,
                        execution_file=f'execution-{index+1:04d}.pt',
                        joint_policy=joint,joint_policy_directory=f'policy-{index+1:04d}',
                        joint_policy_manifest_sha256=file_hash(root/f'policy-{index+1:04d}/MANIFEST.json'),
                        execution_sha256=file_hash(root/f'execution-{index+1:04d}.pt'))
                    if save_now:
                        ckpt_published['rank_shards']={f'rank-{r}/checkpoint-{index+1:04d}.pt':
                            file_hash(root/f'rank-{r}/checkpoint-{index+1:04d}.pt') for r in range(world)}
                    commit_json(root/f'checkpoint-{index+1:04d}-committed.json',ckpt_published)
                dist.barrier()
'''


def run(config, output):
    namespace = {'__name__': __name__+'.derived'}
    exec(compile(generate(config['source_root']), '<macro-dp2-trainer>', 'exec'), namespace)
    return namespace['run'](config, Path(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    run(json.loads(args.config.read_text()), args.output)
