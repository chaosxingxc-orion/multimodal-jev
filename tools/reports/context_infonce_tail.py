"""Two diagnostic updates: actual declared-DP epoch tail then first next-epoch batch.

Optimizer cursors are diagnostic 0/1, not fabricated formal steps 80/81.
The complete authenticated source schedule remains intact on disk.
"""
import ast
from pathlib import Path

from multimodal_jev.context_topology_batch_plan import derive
from tools.reports.context_infonce_trainer import generate as base_generate


def contract(c):
    plan=derive(c['batch_config'])
    if (c.get('experiment_kind')!='infonce_tail_admission' or c.get('start_update')!=0
            or c.get('stop_update')!=2 or c.get('resume_root') is not None
            or c.get('engineering_prefix') is not True or c.get('training_dispatch_allowed') is not False
            or c.get('checkpoint_interval')!=1 or not 1<=plan['dp_replicas']<=8
            or c['stage']!='instruction' or plan['configuration']['epochs']<2):
        raise ValueError('fresh diagnostic declared-DP tail/next-epoch two-update window required')
    return dict(schedule_indices=[plan['updates_per_epoch']-1,plan['updates_per_epoch']],
        real_queries=[plan['tail_real_queries'],plan['global_batch']],diagnostic_updates=2,
        formal_history=False,optimizer_cursor='fresh_0_1',full_schedule_updates=plan['total_updates'])


def select(batches,c,population):
    from tools.reports.context_replica_formal_candidate import validate_schedule
    plan=derive(c['batch_config']);validate_schedule(batches,population,plan)
    spec=contract(c);selected=[batches[i] for i in spec['schedule_indices']]
    if [b['real_queries'] for b in selected]!=spec['real_queries']:
        raise ValueError('authenticated real epoch tail required')
    return selected


def generate(source_root):
    source=base_generate(source_root)
    edits=[
        ("        progress('load')",
         "        from tools.reports.context_infonce_tail import contract as tail_contract,select as select_tail\n"
         "        contract['tail_admission']=tail_contract(c)\n        progress('load')"),
        ("                if len(population)!=c['batch_config']['source_queries'] or len(batches)!=layout['total_updates']:raise ValueError('full learning population required')",
         "                if len(population)!=c['batch_config']['source_queries'] or len(batches)!=layout['total_updates']:raise ValueError('full learning population required')\n"
         "                batches=select_tail(batches,c,population.keys())"),
    ]
    for before,after in edits:
        if source.count(before)!=1:raise ValueError('unique tail integration anchor required')
        source=source.replace(before,after)
    ast.parse(source);return source


def run(c,output):
    contract(c);namespace={'__name__':__name__+'.derived'}
    exec(compile(generate(c['source_root']),'<infonce-tail>','exec'),namespace)
    return namespace['run'](c,Path(output))


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    run(json.loads(a.config.read_text()),a.output)
