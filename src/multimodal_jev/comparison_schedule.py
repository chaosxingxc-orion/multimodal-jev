"""Freeze complete TRAIN recording groups before algorithm outcomes exist."""
from collections import defaultdict
from copy import deepcopy
import random

VIEWS=('native','permutation','rebind','missing_candidate','noul.True','noul.False')


def source_batches(examples,selected_ids,seed,epochs):
    if type(seed) is not int or seed<0 or type(epochs) is not int or epochs<1:
        raise ValueError('nonnegative seed and positive epochs required')
    if not selected_ids or len(selected_ids)!=len(set(selected_ids)):
        raise ValueError('unique nonempty initializer example IDs required')
    wanted=set(selected_ids);selected={};groups=defaultdict(dict)
    for ex in examples:
        if ex.example_id not in wanted:continue
        if ex.example_id in selected:raise ValueError('duplicate manifest example')
        if ex.split!='train':raise ValueError('comparison admits TRAIN only')
        group=ex.metadata.get('source_group');view=ex.metadata.get('variant')
        if not group or view not in VIEWS or view in groups[group]:
            raise ValueError('invalid or duplicate source-group view')
        selected[ex.example_id]=ex;groups[group][view]=ex
    if set(selected)!=wanted:raise ValueError('initializer selection missing from manifest')
    for views in groups.values():
        if set(views)!=set(VIEWS):raise ValueError('complete six-view source groups required')
        if len({ex.audio for ex in views.values()})!=1:
            raise ValueError('source group crosses audio recordings')
    rng=random.Random(seed);batches=[]
    for _ in range(epochs):
        order=sorted(groups);rng.shuffle(order)
        batches.extend([[groups[group][view].example_id for view in VIEWS] for group in order])
    return batches


def comparison_configs(rl_template,sft_template,batches,seed,export_root):
    """Same data/update boundaries; realized tokens and updates are not assumed equal."""
    if not batches:raise ValueError('nonempty comparison batches required')
    if (sft_template['model_path']!=rl_template['actor']['model_path'] or
        any(sft_template.get(k)!=rl_template.get(k) for k in ('initial_adapter','initial_checkpoint')) or
        sft_template['train_manifest']!=rl_template['manifest']):
        raise ValueError('comparison initializer/base/manifest must agree')
    for key in ('lr','lora_rank','lora_alpha','device','dtype','cpu_threads'):
        if sft_template[key]!=rl_template['actor'][key]:
            raise ValueError('comparison training setting differs: '+key)
    sft=deepcopy(sft_template)
    for key in ('accumulate','groups_per_class'):sft.pop(key,None)
    sft.update(seed=seed,batches=deepcopy(batches))
    configs={'sft':sft}
    for reward in ('grpo','pair-grpo'):
        for normalized in (True,False):
            name=reward+('-standard' if normalized else '-center')
            config=deepcopy(rl_template)
            config.update(seed=seed,batches=deepcopy(batches),reward_mode=reward,
                          export_root=str(export_root)+'/'+name)
            config['sampling'].update(n=4,seed=seed)
            config['engine']['seed']=seed
            config['actor']['norm_adv_by_std']=normalized
            configs[name]=config
    return configs
