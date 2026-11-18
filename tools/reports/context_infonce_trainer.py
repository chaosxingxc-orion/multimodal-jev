"""Declared-DP supervised InfoNCE candidate over the existing verl/FSDP trainer.

Frozen trainers are left intact. This adapter adds topology, objective and
cursor-derived warmup; publication still requires actual NPU acceptance.
"""
import ast
import hashlib
from pathlib import Path

from multimodal_jev.context_topology_batch_plan import derive
from multimodal_jev.context_warmup import learning_rate

SOURCE_SHA = '247fa78a7d479b1cade062622812320d0c6f64550bcf37474a4fd72ee70a5a73'


def topology(c, world, mounted):
    plan = derive(c['batch_config'])
    devices = c['physical_devices']
    objective = c['instruction_objective']
    if (type(world) is not int or not 1 <= world <= 8 or plan['dp_replicas'] != world
            or c['batch_config']['tensor_parallel'] != 1
            or c['batch_config']['pipeline_parallel'] != 1
            or c['batch_config']['inference_instances'] != 0
            or len(devices) != world or len(set(devices)) != world
            or any(type(d) is not int or d < 0 for d in devices)
            or sorted(mounted) != sorted('davinci'+str(d) for d in devices)
            or c['global_batch'] != plan['global_batch']
            or c['microbatch'] != c['batch_config']['microbatch']
            or c['stage'] != 'instruction'
            or set(objective) != {'kind', 'temperature', 'probability_coefficient'}
            or objective['kind'] != 'candidate_infonce'
            or objective['probability_coefficient'] != 0.):
        raise ValueError('declared-DP supervised InfoNCE with no inference service required')
    import math
    t = objective['temperature']
    if type(t) not in (int, float) or not math.isfinite(t) or t <= 0:
        raise ValueError('positive InfoNCE temperature required')
    learning_rate(c['lr_schedule'], 0, plan['total_updates'])
    return dict(world_size=world, local_batch=64, total_updates=plan['total_updates'],
                batch_config=c['batch_config'])


def generate(source_root):
    path = Path(source_root) / 'tools/jev/train_context_instruction_npu.py'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('pinned instruction trainer changed')
    source = raw.decode()
    edits = [
        ('from multimodal_jev.encoded_audio_cache import EncodedAudioCache',
         'from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache as EncodedAudioCache'),
        ("                require_training_storage(c['encoded_audio_cache'])",
         "                # Prebuilt inputs are read-only; only runtime output needs writable storage.\n"
         "                require_training_storage(root)"),
        ('    from multimodal_jev.context_native_actor import MODE,build_actor,make_batch',
         '    from tools.reports.context_infonce_actor import load as load_actor\n'
         "    actor_module=load_actor(c['source_root'],temperature=c['instruction_objective']['temperature'])\n"
         '    MODE,build_actor,make_batch=actor_module.MODE,actor_module.build_actor,actor_module.make_batch'),
        ('    from multimodal_jev.slurp_official_training import load',
         '    from multimodal_jev.slurp_portable_release import load'),
        ("    if (world!=4 or c['global_batch']!=128 or c['microbatch']!=8 or\n"
         "            c['runtime_profile']!=PROFILE or c['physical_devices']!=[4,5,6,7] or\n"
         "            sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))!=['davinci4','davinci5','davinci6','davinci7']):\n"
         "        raise ValueError('isolated DP4 global128 micro8 required')",
         '    from tools.reports.context_infonce_trainer import topology\n'
         '    from multimodal_jev.context_warmup import learning_rate,set_learning_rate\n'
         "    layout=topology(c,world,[p.name for p in Path('/dev').glob('davinci[0-9]*')])\n"
         "    if c['runtime_profile']!=PROFILE:raise ValueError('fixed actor runtime required')"),
        ('from multimodal_jev.context_training_schedule import load_schedule',
         'from multimodal_jev.context_replica_schedule import load_schedule,rank_indices'),
        ('stop_update<=634', "stop_update<=layout['total_updates']"),
        ('optimizer=dict(lr=1e-4,betas=', "optimizer=dict(lr=c['lr_schedule']['peak_lr'],betas="),
        ('objective=dict(ranking=1.,binary_nll=1.),grad_clip=1.,world_size=4)',
         "objective=c['instruction_objective'],lr_schedule=c['lr_schedule'],grad_clip=1.,world_size=world,\n"
         "            batch_config=c['batch_config'],parent_schedule_sha256=c['parent_schedule_sha256'],\n"
         "            schedule_release_sha256=c['schedule_release_sha256'])"),
        ('torch.optim.AdamW(parameters,lr=1e-4,',
         "torch.optim.AdamW(parameters,lr=learning_rate(c['lr_schedule'],max(0,start_update-1),layout['total_updates']),"),
        ('ppo_mini_batch_size=32,', "ppo_mini_batch_size=layout['local_batch'],"),
        ('ppo_micro_batch_size_per_gpu=8,', "ppo_micro_batch_size_per_gpu=c['microbatch'],"),
        ("options=dict(stage='instruction',ranking_coefficient=1.,probability_coefficient=1.,",
         "options=dict(stage='instruction',ranking_coefficient=1.,probability_coefficient=0.,"),
        ("release_sha256=c['release_sha256'],population=population)",
         "release_sha256=c['release_sha256'],population=population,batch_config=c['batch_config'],\n"
         "                    parent_root=c['parent_schedule'],parent_manifest_sha256=c['parent_schedule_sha256'],\n"
         "                    parent_release_sha256=c['schedule_release_sha256'])"),
        ('if len(population)!=40513 or len(batches)!=634:',
         "if len(population)!=c['batch_config']['source_queries'] or len(batches)!=layout['total_updates']:"),
        ("publication['release_sha256']!=c['release_sha256']",
         "publication['release_sha256']!=c['schedule_release_sha256']"),
        ('stop_update=stop_update,global_batch=128)', "stop_update=stop_update,global_batch=c['global_batch'])"),
        ("len(commit['rank_shards'])!=4", "len(commit['rank_shards'])!=world"),
        ('rank=rank,world_size=4,', 'rank=rank,world_size=world,'),
        ('[None]*4', '[None]*world'),
        ('for r in range(4)', 'for r in range(world)'),
        ("                        planned=batches[index]['rank_indices']\n"
         "                        ids=[i for group in planned for i in group]\n"
         "                        # Verify actual processor shapes reproduce the frozen grouping.\n"
         "                        from multimodal_jev.span_length_groups import length_group_indices\n"
         "                        lengths=[(p['inputs']['input_ids'].shape[-1],p['inputs']['input_features'].shape[-1]) for p in items]\n"
         "                        if planned!=length_group_indices(lengths,microbatch=8,world_size=4,order='length_grouped'):\n"
         "                            raise ValueError('actual processor lengths differ from frozen grouping')",
         "                        lengths=[(p['inputs']['input_ids'].shape[-1],p['inputs']['input_features'].shape[-1]) for p in items]\n"
         "                        planned=rank_indices(batches[index],lengths,c['batch_config'])\n"
         "                        ids=[i for group in planned for i in group]"),
        ('**options).split(32)', "**options).split(layout['local_batch'])"),
        ("                progress('native_update',update=index)",
         "                current_lr=set_learning_rate(opt,c['lr_schedule'],index,layout['total_updates'])\n"
         "                progress('native_update',update=index,learning_rate=current_lr)"),
        ('record=dict(update=index+1,preparation_seconds=',
         'record=dict(update=index+1,learning_rate=current_lr,preparation_seconds='),
        ('schedule_updates=634,learning_population=40513',
         "schedule_updates=layout['total_updates'],learning_population=c['batch_config']['source_queries']"),
        ("        named=[(n,p) for n,p in wrapped.named_parameters() if p.requires_grad]",
         "        named=[(n,p) for n,p in wrapped.named_parameters() if p.requires_grad]\n"
         "        from multimodal_jev.context_lora_checkpoint_scope import validate_trainable_scope\n"
         "        atomic_json(out/'checkpoint-scope.json',validate_trainable_scope(named))"),
    ]
    counts = {'rank=rank,world_size=4,': 2, '[None]*4': 2}
    for before, after in edits:
        if source.count(before) != counts.get(before, 1):
            raise ValueError('instruction integration anchor changed: '+before[:100])
        source = source.replace(before, after)
    ast.parse(source)
    return source


def run(config, output):
    namespace = {'__name__': __name__+'.derived'}
    exec(compile(generate(config['source_root']), '<infonce-trainer>', 'exec'), namespace)
    return namespace['run'](config, Path(output))


if __name__ == '__main__':
    import argparse
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(json.loads(args.config.read_text()), args.output)
