"""Finite native verl candidate for any local64 DP width and declared macro batch.

Reuses the frozen macro trainer; actual NPU execution and independent update
audit are still required. No model/optimizer changes or training launch here.
"""
import ast
import re
from pathlib import Path

from multimodal_jev.context_topology_batch_plan import derive
from tools.reports.context_macro_trainer_candidate import generate as macro_source, ALGORITHM

MODULE = re.compile(r'multimodal_jev\.([A-Za-z0-9_]+)')


def referenced_modules(source_root):
    """Every multimodal_jev module the generated trainer imports must exist in
    the assembled source. Packages inherit most of them from the parent; the
    ones the transform or its macro base first reference have to be shipped."""
    return sorted(set(MODULE.findall(generate(source_root))))


def topology(c,world,mounted):
    plan=derive(c['batch_config'])
    devices=c['physical_devices'];client=c['macro_client']
    if (plan['configuration']['tensor_parallel'] != 1
            or plan['configuration']['pipeline_parallel'] != 1
            or not 1 <= world <= 8 or world != plan['dp_replicas']
            or len(devices) != world or any(type(d) is not int or d<0 for d in devices)
            or len(set(devices)) != world
            or not set(devices) <= {int(n[len('davinci'):]) for n in mounted
                                   if n.startswith('davinci') and n[len('davinci'):].isdigit()}
            or c['global_batch'] != plan['global_batch']
            or c['microbatch'] != plan['configuration']['microbatch']
            or c['algorithm'] != ALGORITHM or c['dual_arm'] != 'joint'
            or client['schema'] != 'jev-replica-macro-client-v1'
            or client['max_queries'] != plan['macro_queries']
            or type(client['rollout_queries']) is not int
            or not 1 <= client['rollout_queries'] <= plan['macro_queries']
            or len(client['replicas']) != plan['configuration']['inference_instances']
            or client['runtime_identity'] != c['dual_runtime_identity']):
        raise ValueError('declared local64 DP topology and matching macro runtime required')
    return dict(world_size=world,local_batch=64,microbatch=c['microbatch'],
        accumulation=plan['accumulation'],rollout_queries=client['rollout_queries'],
        replicas=len(client['replicas']),global_batch=plan['global_batch'],
        total_updates=plan['total_updates'],batch_config=c['batch_config'])


def generate(source_root):
    source=macro_source(source_root)
    edits=[
        ('from tools.reports.context_macro_trainer_candidate import topology,repartition',
         'from tools.reports.context_replica_trainer_candidate import topology\n'
         '    from multimodal_jev.context_replica_schedule import rank_indices'),
        ('from multimodal_jev.context_training_schedule import load_schedule',
         'from multimodal_jev.context_replica_schedule import load_schedule'),
        ('from tools.reports.context_macro_client import connect_pool',
         'from tools.reports.context_replica_macro_client import connect_pool'),
        ('stop_update<=634',"stop_update<=layout['total_updates']"),
        ("release_sha256=c.get('schedule_release_sha256',c['release_sha256']),population=population)",
         "release_sha256=c['release_sha256'],population=population,batch_config=c['batch_config'],\n"
         "                    parent_root=c['parent_schedule'],parent_manifest_sha256=c['parent_schedule_sha256'],\n"
         "                    parent_release_sha256=c['schedule_release_sha256'])"),
        ("if len(population)!=40513 or len(batches)!=634:",
         "if len(population)!=c['batch_config']['source_queries'] or len(batches)!=layout['total_updates']:"),
        ('stop_update=stop_update,global_batch=128)',"stop_update=stop_update,global_batch=c['global_batch'])"),
        ("planned=repartition(lengths,batches[index]['rank_indices'],microbatch=c['microbatch'],world_size=world)",
         "planned=rank_indices(batches[index],lengths,c['batch_config'])"),
        ('schedule_updates=634,learning_population=40513',
         "schedule_updates=layout['total_updates'],learning_population=c['batch_config']['source_queries']"),
    ]
    for before,after in edits:
        if source.count(before)!=1:raise ValueError('replica integration anchor changed: '+before)
        source=source.replace(before,after)
    ast.parse(source)
    return source


def run(config,output):
    namespace={'__name__':__name__+'.derived'}
    exec(compile(generate(config['source_root']),'<replica-trainer>','exec'),namespace)
    return namespace['run'](config,Path(output))


def prepare_output(path):
    """Every torchrun rank enters the same entrypoint; the shared root must be
    idempotent. Freshness is enforced by the generated run's per-rank dirs."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare_output(a.output)
    run(json.loads(a.config.read_text()),a.output)
