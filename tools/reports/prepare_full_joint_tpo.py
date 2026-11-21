"""Publish one full clean TRAIN schedule and authenticate step zero."""
from collections import Counter
import json
import math
from pathlib import Path
import random
import shutil

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_preparation import processor_cache_identity
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from multimodal_jev.context_reference_dataset import digest
from multimodal_jev.local_checkpoint import commit_file, commit_json
from tools.reports.build_tpo_view import read, verified


def schedule(ids, plan):
    """Labels and prior scores are deliberately absent from this interface."""
    names = sorted(ids)
    widths = plan['candidate_lengths']
    size, micro = plan['global_batch'], plan['microbatch']
    steps = math.ceil(len(names) / size)
    if (not names or len(set(names)) != len(names) or size % micro
            or widths != sorted(set(widths)) or not 0 < widths[0] < widths[-1]
            or widths[-1] != len(plan['candidate_ids'])):
        raise ValueError('invalid full TRAIN schedule geometry')
    for epoch in range(plan['epochs']):
        rng = random.Random(f"{plan['seed']}:joint-schedule:{epoch}")
        order = rng.sample(names, len(names))
        widths_order = rng.sample(widths,len(widths))
        for index in range(steps):
            members = order[index*size:(index+1)*size]
            slots = []
            for pos in range(size):
                pad = pos >= len(members)
                uid = members[pos % len(members)]
                width = widths_order[(index*(size//micro)+pos//micro) % len(widths)]
                rng_slate = random.Random(f"{plan['seed']}:joint-slate:{epoch}:{uid}")
                slate = rng_slate.sample(range(widths[-1]), width)
                slots.append(dict(id=uid,padding=pad,weight=int(not pad),slate=slate))
            yield dict(epoch=epoch,index=index,mode='joint',slots=slots)


def exposure(rows, batches, plan):
    import itertools
    seen = Counter()
    outcomes = Counter()
    size, micro = plan['global_batch'], plan['microbatch']
    steps = math.ceil(len(rows)/size)
    for step, batch in enumerate(batches):
        epoch, index = divmod(step,steps)
        slots = batch['slots']
        real = min(size, len(rows)-index*size)
        if (batch['epoch'],batch['index'],batch['mode'],len(slots)) != (epoch,index,'joint',size):
            raise ValueError('joint batch cursor differs')
        for pos, slot in enumerate(slots):
            uid, slate = slot['id'],slot['slate']
            row = rows[uid]
            if (slot['padding'] is not (pos >= real) or slot['weight'] != int(pos < real)
                    or len(slate) not in plan['candidate_lengths']
                    or len(slate) != len(set(slate))
                    or len(slate) != len(slots[pos-pos%micro]['slate'])
                    or any(type(i) is not int or not 0 <= i < len(plan['candidate_ids']) for i in slate)):
                raise ValueError('invalid joint slate, tail or weight')
            if pos < real:
                seen[epoch,uid] += 1
                outcomes[epoch,len(slate),row['gold'],row['gold'] in slate] += 1
    if (len(batches) != steps * plan['epochs']
            or seen != Counter(itertools.product(range(plan['epochs']), rows))):
        raise ValueError('each real TRAIN recording must occur once per epoch')
    return [dict(epoch=e,width=w,candidate=plan['candidate_ids'][g],hit=h,n=n)
            for (e,w,g,h), n in sorted(outcomes.items())]


def training_cache_keys(root, manifest, template, plan, receipt, train, ids):
    from multimodal_jev.model_provider import cache_namespace
    from multimodal_jev.span_request_identity import tensor_identity

    if (manifest.get('schema') != 'context-full-learning-cache-v1'
            or manifest.get('model_provider_sha256') != cache_namespace(template)
            or manifest['base_id'] != receipt['base_id'] or manifest['candidate_ids'] != ids
            or manifest['release_sha256'] != template['release_sha256']
            or manifest['prompt_layout'] != template['prompt_layout']
            or manifest['processor_sha256'] != receipt['processor_sha256']
            or manifest['recordings'] != len(train) or len(train) != plan['recordings']):
        raise ValueError('TRAIN input cache belongs to a different model or release')
    keys = {}
    with verified(root/'inputs.jsonl', manifest['inputs_sha256']).open() as stream:
        for line in stream:
            row = json.loads(line)
            uid, positions = row['id'], row['positions']
            inputs = dict(row['tensors'])
            if (uid in keys or row['audio_sha256'] != train[uid]['audio_sha256']
                    or len(positions) != len(ids) or positions != sorted(set(positions))
                    or inputs.pop('positions') != tensor_identity(torch.tensor([positions], dtype=torch.long))):
                raise ValueError('duplicate or inconsistent TRAIN input receipt')
            keys[uid] = digest(dict(policy_id=plan['source_policy'],positions=positions,inputs=inputs))
    if keys.keys() != train.keys():
        raise ValueError('incomplete TRAIN input population')
    return keys


def publish(c, template, entry, repo):
    from multimodal_jev.model_provider import require_same_provider
    spec = c['full_joint']
    design = repo/spec['design']
    plan = read(verified(design, spec['design_sha256']) if 'design_sha256' in spec else design)
    terminal = read(verified(Path(spec['initial'])/'terminal.json',spec['initial_terminal_sha256']))
    receipt = read(verified(Path(spec['initial'])/'result.json',spec['initial_result_sha256']))
    init_config = read(verified(Path(spec['initial']).parent/'init.json',spec['initial_config_sha256']))
    require_same_provider(template,init_config)
    if 'model_provider' in c:
        require_same_provider(c,template)
    init = Path(spec['initial'])/'policy'
    if 'model_provider' in template:
        if 'design_sha256' not in spec or {'source_policy', 'source_manifest_sha256'} & plan.keys():
            raise ValueError('explicit providers require a pinned model-independent training design')
        plan.update(source_policy=receipt['policy_id'], source_manifest_sha256=file_hash(init/'MANIFEST.json'))
    manifest = read(verified(init/'MANIFEST.json',plan['source_manifest_sha256']))
    if (terminal['exit_code'] != 0 or receipt['status'] != 'stage1_fresh_cpu_policy_verified'
            or receipt['npu_executed'] is not False or receipt['formal_training_admitted'] is not False
            or receipt['policy_id'] != plan['source_policy'] or receipt['base_id'] != template['base_id']
            or init_config['seed'] != plan['seed'] or init_config['model'] != template['model']):
        raise ValueError('unbound CPU step-zero policy')
    torch.set_num_threads(spec.get('cpu_threads', 2))
    contract = manifest['contract']
    bundle = read_execution_bundle(init,expected_policy_id=plan['source_policy'],
                                   expected_contract=contract,expected_base_id=template['base_id'])
    adapter = init_config['lora'] if 'model_provider' in template else dict(rank=64, alpha=128)
    if (contract['token']['registration_status'] != 'admitted'
            or {(x['rank'],x['alpha']) for x in contract['lora'].values()} != {
                (adapter['rank'], adapter['alpha'])}
            or any(t.count_nonzero() for n,t in bundle['tensors'].items() if '.lora_B.' in n)
            or processor_cache_identity(init) != receipt['processor_sha256']):
        raise ValueError('policy is not the configured fresh step-zero execution')
    release = Path(template['release'])
    release_m = read(verified(release/'MANIFEST.json',template['release_sha256']))
    files = release_m['files']
    task = read(verified(release/'tasks.json',files['tasks.json']))['intent']
    ids = [x['id'] for x in task['schema']['candidates']]
    train = {}
    with verified(release/'learning.jsonl',files['learning.jsonl']).open() as f:
        for line in f:
            row = json.loads(line)
            if (row['id'] in train or row['role'] != 'learning' or row['official_split'] != 'train'
                    or row['known_pending_source'] is not False or row['intent'] not in ids):
                raise ValueError('unexpected row in official clean TRAIN')
            train[row['id']] = row
    old = Path(spec['input_identity_publication'])
    ref = read(verified(old/'MANIFEST.json',spec['input_identity_sha256']))
    if 'model_provider' in template:
        keys = training_cache_keys(old,ref,template,plan,receipt,train,ids)
    else:
        identity = ref['identity']
        pop = read(verified(old/'population.json',ref['files']['population.json']))
        if (identity['candidate_ids'] != ids or identity['base_id'] != receipt['base_id']
                or identity['release_sha256'] != template['release_sha256']
                or identity['runtime_identity']['processor_sha256'] != receipt['processor_sha256']
                or ref['recordings'] != len(train) or len(train) != plan['recordings']
                or digest(pop) != identity['population_sha256'] or set(pop) != set(train)):
            raise ValueError('old input identity population does not match clean TRAIN')
        # Keep only authenticated tensor identities; never access historical scores.
        records = read(verified(old/'records.json',ref['files']['records.json']))
        if set(pop.values()) != set(records):
            raise ValueError('incomplete input identity publication')
        new_keys = {}
        for key, record in records.items():
            req = record['request_identity']
            if (set(req) != {'policy_id','positions','inputs'} or req['policy_id'] != identity['policy_id']
                    or len(req['positions']) != len(ids) or digest(req) != key):
                raise ValueError('native request tensor identity differs')
            new_keys[key] = digest(dict(req,policy_id=plan['source_policy']))
        keys = {uid:new_keys[key] for uid,key in pop.items()}
    plan['candidate_ids'] = ids
    batches = list(schedule(train,plan))
    gs = {name:i for i,name in enumerate(ids)}
    rows = {uid:dict(request_key=keys[uid],audio_sha256=r['audio_sha256'],gold=gs[r['intent']])
            for uid,r in train.items()}
    counts = exposure(rows,batches,plan)
    devices, infer = spec['training_devices'],spec['inference_devices']
    batch = template['batch_config']
    scope = devices + infer
    if (len(devices) != batch['training_devices'] or len(infer) != batch['inference_instances']
            or not devices or not infer or len(set(scope)) != len(scope)
            or any(type(card) is not int or card < 0 for card in scope)
            or len(devices)*batch['per_dp_replica_batch'] != plan['global_batch']
            or plan['microbatch'] != batch['microbatch']):
        raise ValueError('runtime device, global batch or microbatch geometry differs')
    dest = entry/'targets'
    dest.mkdir()
    commit_json(dest/'rows.json',rows)
    # Avoid a large indent-heavy wire format in the frozen release.
    commit_file(dest/'batches.json',lambda f:f.write(json.dumps(batches,separators=(',', ':')).encode()))
    sched = {k:plan[k] for k in ('global_batch','microbatch','seed','epochs')}
    updates = len(batches)
    commit_json(dest/'MANIFEST.json',dict(schema='jev-tpo-joint-view-v1',source_policy=plan['source_policy'],
        release_sha256=template['release_sha256'],candidate_ids=ids,updates=updates,schedule=sched,
        input_identity_sha256=spec['input_identity_sha256'],real_exposures=len(rows)*plan['epochs'],
        exposures_by_epoch_width_class=counts,files={p.name:file_hash(p) for p in sorted(dest.iterdir())}))
    target_sha = file_hash(dest/'MANIFEST.json')
    parent = {k:template[k] for k in ('model','base_id','release','release_sha256','encoded_audio_cache',
                                    'prompt_layout','view_execution','behavior_bounds')}
    if 'model_provider' in template:
        parent['model_provider'] = template['model_provider']
    parent.update(source_policy='/jev-entry/initial/policy',source_policy_id=plan['source_policy'],
                  source_contract=contract,physical_devices=devices,engineering_prefix=False,
                  batch_config=dict(batch,source_queries=len(rows)),training_dataset={'manifest_sha256':target_sha})
    parent['view_execution']['actor_seed']=plan['seed']
    shutil.copytree(init,entry/'initial/policy')
    for name in ('terminal.json','result.json'):
        shutil.copyfile(Path(spec['initial'])/name,entry/'initial'/name)
    commit_json(entry/'initial/config.json',init_config)
    parent_path = entry/'parent.json'
    commit_json(parent_path,parent)
    return parent, dict(plan, updates=updates, target_sha=target_sha, parent_sha=file_hash(parent_path),
        training_devices=devices,inference_devices=infer,intermediate_export_updates=[updates//plan['epochs']],
        initialization_receipt_sha256=spec['initial_result_sha256'])
