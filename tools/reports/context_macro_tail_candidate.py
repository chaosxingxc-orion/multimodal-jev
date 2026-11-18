"""Two fresh diagnostic updates on the actual epoch tail and next epoch batch.

Retains the complete authenticated source schedule, then explicitly selects
indices316/317 for a bounded engineering run. This is not a formal prefix or
source efficacy result; optimizer/rollout cursors are diagnostic0/1.
"""
import ast
from pathlib import Path

from tools.reports.context_macro_formal_candidate import validate_schedule
from tools.reports.context_macro_trainer_candidate import generate as base_source

INDICES=(316,317)


def contract(c):
    if (c.get('experiment_kind')!='tail_admission' or c.get('start_update')!=0
            or type(c.get('start_update')) is not int or c.get('stop_update')!=2
            or type(c.get('stop_update')) is not int or c.get('microbatch')!=8
            or c.get('global_batch')!=128 or c.get('resume_root')
            or c.get('engineering_prefix') is not True
            or c.get('training_dispatch_allowed') is not False
            or c.get('checkpoint_interval')!=1 or c.get('stage')!='task'):
        raise ValueError('fresh DP2 micro8/global128 two-update tail admission required')
    return dict(schema='jev-macro-tail-admission-v1',schedule_indices=list(INDICES),
        real_queries=[65,128],diagnostic_updates=2,formal_history=False,
        initial_weights='common_Stage1',fresh_reference_cache=True)


def select(batches,population):
    counts=validate_schedule(batches,population)
    if [counts[i] for i in INDICES]!=[65,128]:raise ValueError('actual tail/next-epoch counts differ')
    return [batches[i] for i in INDICES]


def generate(source_root):
    source=base_source(source_root)
    edits=[
        ('        from tools.reports.context_macro_trainer_candidate import experiment_window\n'
         '        experiment_window(c)',
         '        from tools.reports.context_macro_tail_candidate import contract as tail_contract,select as select_tail\n'
         '        tail=tail_contract(c)'),
        ("        progress('load')","        contract['tail_admission']=tail\n        progress('load')"),
        ("                if len(population)!=40513 or len(batches)!=634:raise ValueError('full learning population required')",
         "                if len(population)!=40513 or len(batches)!=634:raise ValueError('full learning population required')\n"
         "                batches=select_tail(batches,population.keys())"),
        ('stop_update=stop_update,schedule_updates=634,learning_population=40513,',
         'stop_update=stop_update,schedule_updates=2,learning_population=40513,tail_admission=tail,')]
    for before,after in edits:
        if source.count(before)!=1:raise ValueError('unique tail integration anchor required: '+before[:80])
        source=source.replace(before,after)
    ast.parse(source)
    return source


def run(config,output):
    contract(config)
    ns={'__name__':__name__+'.derived'}
    exec(compile(generate(config['source_root']),'<macro-tail-admission>','exec'),ns)
    return ns['run'](config,Path(output))


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    run(json.loads(a.config.read_text()),a.output)
