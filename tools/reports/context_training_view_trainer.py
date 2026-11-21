"""Stage2 opt-in view adapter over the shared editable-objective trainer.

Full reference and mirror coverage remain authenticated. Only the scheduled
training population is reduced. Frozen packages never import this implicitly.
"""
import ast
from pathlib import Path
from tools.reports.context_plugin_trainer import generate as base_generate


def generate(source_root, *, formal=False):
    source=base_generate(source_root,formal=formal)
    begin="                population={r['id']:r for r in rows}\n"
    end="                if len(population)!=c['batch_config']['source_queries']"
    a=source.index(begin);b=source.index(end,a)
    if source.count(begin)!=1 or source.count(end)!=1:raise ValueError('unique population load required')
    source=source[:a]+("                from multimodal_jev.stage2_training_view import training_inputs\n"
        "                full_population={r['id']:r for r in rows}\n"
        "                population,batches,manifest=training_inputs(c,rows,task)\n")+source[b:]
    before="                if set(c['reference_dataset']['population'])!=set(population):"
    if source.count(before)!=1:raise ValueError('reference coverage anchor changed')
    source=source.replace(before,"                if set(c['reference_dataset']['population'])!=set(full_population):")
    # Mirror still covers all source recordings, preserving original validation.
    a=source.index("                if c.get('audio_mirror_root') is not None:")
    b=source.index('                preparer=CachedContextPreparer',a)
    section=source[a:b].replace('len(population)','len(full_population)').replace('set(population)','set(full_population)').replace("source=population[r['id']]","source=full_population[r['id']]")
    source=source[:a]+section+source[b:]
    anchor="        contract['stage2_objective']=objective_contract"
    if source.count(anchor)!=1:raise ValueError('checkpoint objective anchor changed')
    source=source.replace(anchor,anchor+"\n        contract['training_view']=dict(manifest_sha256=c['training_view']['manifest_sha256'],mode='fixed_stage1_train_difficulty')")
    ast.parse(source);return source


def run(c,output):
    if c.get('stage')!='task' or not c.get('training_view'):raise ValueError('explicit Stage2 training view required')
    scope={'__name__':__name__+'.derived'}
    exec(compile(generate(c['source_root'],formal=c['experiment_kind']=='formal_source'),'<training-view-trainer>','exec'),scope)
    return scope['run'](c,Path(output))


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output)
