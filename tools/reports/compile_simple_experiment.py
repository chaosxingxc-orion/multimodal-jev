"""Compile one portable recipe into a new immutable experiment directory."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

from multimodal_jev.context_stage_lora import transition
from multimodal_jev.context_topology_batch_plan import derive
from multimodal_jev.source_snapshot import regular
from multimodal_jev.stage_experiment_contract import reference_batch, source_curve

ROOT = Path(__file__).resolve().parents[2]


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()


def pin(name, raw):
    return dict(path=str(name), sha256=hashlib.sha256(raw).hexdigest())


def load(ref):
    if set(ref) != {'path','sha256'}:
        raise ValueError('versioned project input required')
    raw = regular(ROOT,ref['path']).read_bytes()
    if pin(ref['path'],raw) != ref:
        raise ValueError('recipe dependency changed: '+ref['path'])
    return json.loads(raw)


def compile_recipe(recipe, destination):
    fields = ('base objective dataset_template manifest_template total_devices rollout_instances '
              'replica_batch microbatch epochs rollout_batch stage1_lora stage2_lora '
              'reuse_stage1_adapter warmup_fraction selection_interval').split()
    version=recipe['schema']
    if version=='jev-simple-recipe-v2':
        fields+=['parity_minimum_queries','serving','advantage_std_floor']
    if set(recipe) != {'schema',*fields} or version not in ('jev-simple-recipe-v1','jev-simple-recipe-v2'):
        raise ValueError('explicit complete simple experiment recipe required')
    total, rollout = recipe['total_devices'],recipe['rollout_instances']
    if type(total) is not int or type(rollout) is not int or not 0 < rollout < total:
        raise ValueError('training devices and dedicated rollout services required')
    fraction, interval = recipe['warmup_fraction'],recipe['selection_interval']
    if type(fraction) not in (float,int) or not 0 < fraction <= 1 or type(interval) is not int or interval<1:
        raise ValueError('invalid warmup or evaluation cadence')
    design, dataset, manifest = (load(recipe[key]) for key in ('base','dataset_template','manifest_template'))
    objective = load(recipe['objective'])
    execution=load(dict(path=objective['execution']['source'],sha256=objective['execution']['sha256']))
    file=regular(ROOT,objective['objective_source'])
    if hashlib.sha256(file.read_bytes()).hexdigest()!=objective['objective_sha256']:
        raise ValueError('objective source changed')
    source, task = design['stage1'],design['stage2']
    for name,stage,train,engines in [('stage1',source,total,0),('stage2',task,total-rollout,rollout)]:
        adapter=recipe[name+'_lora']
        if set(adapter)!={'rank','alpha'}:
            raise ValueError('explicit adapter rank and alpha required')
        stage['lora'].update(adapter)
        stage['batch_config'].update(training_devices=train,inference_instances=engines,
            per_dp_replica_batch=recipe['replica_batch'],microbatch=recipe['microbatch'],
            epochs=recipe['epochs'],inference_batch_queries=recipe['rollout_batch'])
        batch=dict(stage['batch_config'],source_queries=design['population']['learning'])
        plan=derive(batch)
        if plan['accumulation']<1 or batch['per_dp_replica_batch']%batch['microbatch']:
            raise ValueError('microbatch must divide effective replica batch')
        stage['global_batch']=plan['global_batch']
        if name=='stage1':
            stage['batch_config']=batch
            stage.update(total_updates=plan['total_updates'],updates_per_epoch=plan['updates_per_epoch'],
                tail_real=plan['tail_real_queries'],tail_padding=plan['tail_padding_slots'],
                evaluation_steps=[*range(0,plan['total_updates'],interval),plan['total_updates']],
                reference_endpoint=plan['total_updates'],initialization='fresh_pre_stage1_joint_snapshot')
            stage['lr_schedule']['warmup_updates']=math.ceil(plan['total_updates']*fraction)
        else:
            stage['batch_config']['source_queries']=None
            stage.update(macro_queries=plan['global_batch'],rollout_queries=batch['inference_batch_queries'],
                population_plan='derive schedule from frozen difficulty view',objective_config=recipe['objective'],
                reuse_stage1_adapter=recipe['reuse_stage1_adapter'])
            dataset['batch_config']={k:v for k,v in batch.items() if k!='source_queries'}
    transition(source['lora'],task['lora'],reuse_stage1_adapter=task['reuse_stage1_adapter'])
    design.update(total_device_budget=total,status='candidate_pending_admission',
                  implementation_defaults='see recipe',model=design['model'].split(' + ')[0])
    source_curve(design)
    reference_batch(design)
    runtime_files={}
    if version=='jev-simple-recipe-v2':
        from multimodal_jev.stage2_view_runtime import validate_execution
        from multimodal_jev.grpo_numerics import validate as validate_numerics
        validate_execution(execution)
        floor=recipe['advantage_std_floor']
        validate_numerics(dict(advantage_std_floor=floor,policy_clip=execution['policy_clip']))
        task.update(group_size=execution['objective_group_size'],learning_rate=execution['optimizer']['lr'],
                    clip_ratio=execution['policy_clip']['clip_ratio'],advantage_std_floor=floor)
        serving=load(recipe['serving'])
        serving['max_batch_queries']=task['rollout_queries']
        if (any(type(v) is not int or v<1 for v in serving.values())
                or serving['max_batch_tokens']<serving['max_tokens'] or serving['max_tokens']!=execution['max_input_tokens']):
            raise ValueError('incompatible service and actor input budgets')
        minimum=recipe['parity_minimum_queries']
        if type(minimum) is not int or minimum<256:
            raise ValueError('cannot shrink numerical admission below the existing panel')
        panel=max(minimum,task['macro_queries'])
        if panel>design['population']['learning']:
            raise ValueError('insufficient TRAIN population for complete parity macro')
        runtime_files['stage2-runtime.json']=encoded(dict(serving=serving,
            panel_config=dict(minimum_queries=panel,max_tokens=serving['max_tokens'])))
        design['batch_rule']='global_batch = per_dp_replica_batch * training_devices / (TP * PP)'
    files={**runtime_files,'two-stage.json':encoded(design),'dataset.json':encoded(dataset),'recipe.json':encoded(recipe)}
    manifest.update(status=design['status'],stage2=recipe['objective'],
                    stage1=pin(destination/'two-stage.json',files['two-stage.json']),
                    dataset=pin(destination/'dataset.json',files['dataset.json']))
    if runtime_files:
        manifest['runtime']=pin(destination/'stage2-runtime.json',runtime_files['stage2-runtime.json'])
    files['experiment.json']=encoded(manifest)
    return files


def publish(recipe, destination):
    destination=Path(destination).resolve().relative_to(ROOT)
    if not destination.is_relative_to('configs/experiments') or destination==Path('configs/experiments'):
        raise ValueError('new project experiment directory required')
    files=compile_recipe(recipe,destination)
    target=ROOT/destination
    if target.exists():
        if (not target.is_symlink() and target.is_dir() and set(p.name for p in target.iterdir())==set(files)
                and all(regular(target,k).read_bytes()==raw for k,raw in files.items())):
            return target/'experiment.json'
        raise FileExistsError(target)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        ready=Path(tmp)/'sealed'
        ready.mkdir()
        for name,raw in files.items():
            (ready/name).write_bytes(raw)
        ready.rename(target)
    return target/'experiment.json'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(publish(json.loads(args.recipe.read_bytes()),args.output))
