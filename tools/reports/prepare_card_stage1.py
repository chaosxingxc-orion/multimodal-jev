"""Build the knowledge-card Stage1 package for the experiment ledger."""
import json
from pathlib import Path
import shutil
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

from multimodal_jev.baseline import file_hash
from multimodal_jev.card_experiment_config import bind_stage1
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.source_snapshot import freeze
from tools.reports.freeze_card_assignments import compile_plan, read


def load(path, sha):
    return json.loads(read(Path(path),sha))


def run(c):
    from multimodal_jev.card_contrast import KIND, Directions
    window = c['window']
    if 'continuation' in window:
        from multimodal_jev.card_experiment_config import DEFAULT_PROFILE
        origin = window['continuation']
        legacy = json.loads(DEFAULT_PROFILE.with_name('card-legacy-stage1-runtime-v1.json').read_bytes())
        if 'legacy_runtime_sha256' in origin:
            raise ValueError('legacy runtime binding must come from the registered source identity')
        if origin['formal_sha256'] in legacy:
            window['continuation'] = dict(origin, legacy_runtime_sha256=legacy[origin['formal_sha256']])
    if 'training_acoustic_engine' in window:
        admission=c['comparison_admission']
        verdict=load(admission['path'],admission['sha256'])
        if not (verdict['passed'] is True and verdict['identical_training_state'] is True and
                0 <= verdict['max_abs_logit_error'] <= verdict.get('logit_atol', 0) and 'continuation' in window):
            raise ValueError('acoustic engine lacks an accepted paired training comparison')
    objective = window.get('triplet_objective', {})
    if objective.get('kind') == KIND:
        Directions.from_objective(objective)
        if ('class_logit_ema' not in window or 'single_card' not in window or not c.get('mixed')
                or not c.get('bind_design_budget') or 'epoch_mining' in window or 'bootstrap_input' in window):
            raise ValueError('card contrast requires the budget-bound single-card online EMA pipeline')
    root=Path(c['output'])
    entry=root/'entry'
    entry.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2]
    design=load(c['design'],c['design_sha256'])
    for stage in ('stage1','stage2'):
        design[stage]['lora'].update(c['lora'])
    design.update(status='registered',model=c['model_description'])
    extension = None
    if c.get('mixed'):
        from multimodal_jev.context_topology_batch_plan import derive
        m = c['mixed']
        data = Path(m['data'])
        proof = load(data/'admission.json',m['admission_sha256'])
        synth = load(data/'synthetic.json',proof['roster_sha256'])
        for name in ('admission.json','synthetic.json','annotation-errors.json'):
            shutil.copyfile(data/name,entry/name)
        definitions = json.loads(Path(m['protocol']).read_bytes())['extra_cards']
        extension = synth, definitions, m['admission_sha256']
        d = design['stage1']
        d['batch_config']['source_queries'] = design['population']['learning'] = proof['population']
        if c.get('fresh_policy'):
            d['batch_config'].update(per_dp_replica_batch=c['window']['effective_anchors_per_rank'],
                                     microbatch=c['window']['execution_microbatch'])
        plan = derive(d['batch_config'])
        d.update(initialization='fresh_pre_stage1_joint_snapshot' if c.get('fresh_policy') else f"fresh_rank{c['lora']['rank']}_pre_stage1_joint_snapshot",total_updates=plan['total_updates'],
            updates_per_epoch=plan['updates_per_epoch'],reference_endpoint=plan['total_updates'],
            tail_real=plan['tail_real_queries'],tail_padding=plan['tail_padding_slots'],
            evaluation_steps=[i*plan['updates_per_epoch'] for i in range(1+d['batch_config']['epochs'])])
        if c.get('bind_design_budget'):
            # Reproduce the historical cache key with only the schedule metadata substituted.
            from hashlib import sha256
            def publication_hash(value):
                return sha256((json.dumps(value, indent=2)+'\n').encode()).hexdigest()
            cache_design = publication_hash(design)
            w = c['window']
            d['batch_config'].update(training_devices=w['training_npus'], batch_rule=w['batch_rule'],
                                     microbatch=w['execution_microbatch'], per_dp_replica_batch=w['effective_anchors_per_rank'])
            plan = derive(d['batch_config'])
            d.update(global_batch=plan['global_batch'], instruction_objective=w['triplet_objective'],total_updates=plan['total_updates'], updates_per_epoch=plan['updates_per_epoch'],
                     reference_endpoint=plan['total_updates'], tail_real=plan['tail_real_queries'],
                     tail_padding=plan['tail_padding_slots'],
                     evaluation_steps=list(range(0,plan['total_updates']+1,plan['updates_per_epoch'])))
        c['window'].update(checkpoint_interval=d['updates_per_epoch'],evaluation_steps=d['evaluation_steps'],
                           mixed=m['runtime'])
    commit_json(entry/'design.json',design)
    compile_plan(c,entry,file_hash(entry/'design.json'),extension=extension)
    if c.get('bind_design_budget'):
        assignments = json.loads((entry/'card-plan.json').read_bytes())
        c['window']['cache_card_plan_sha256'] = publication_hash(dict(assignments, design_sha256=cache_design))
        del assignments
    parent=bind_stage1(load(c['parent'],c['parent_sha256']), c)
    native=load(c['native_cache'],c['native_cache_sha256'])
    if native['source_identity']!=parent['release_sha256'] or len(native['templates']['both']['suffix'])!=len(native['templates']['native']['suffix']) or len(native['templates']['both']['prefix'])!=len(native['templates']['native']['prefix']):
        raise ValueError('historical lengths must share the native token count')
    if not extension:
        parent['input_limit'] = native['encoding']['max_tokens']
    def block(path):
        return [(r['id'],r['input_tokens']) for r in json.loads(path.read_bytes())['records'] if r['weight']]
    lengths={}
    if extension:
        prior = m['native_lengths']
        lengths = load(prior['path'], prior['sha256'])
    else:
        with ThreadPoolExecutor(c['io_workers']) as pool:
            for values in pool.map(block, sorted(Path(c['batches']).glob('batch-*.json'))):
                for uid,length in values:
                    if lengths.setdefault(uid,length)!=length:
                        raise ValueError('inconsistent acoustic lengths')
    if len(lengths)!=native['full_verified'] or max(lengths.values())>native['max_tokens']:
        raise ValueError('full source lengths differ from cache proof')
    commit_json(entry/'native-lengths.json',lengths)
    commit_json(entry/'native-cache.json',native)
    init=load(c['init'],c['init_sha256'])
    # Carry only authenticated immutable inputs consumed from the historical release.
    old_source=Path(parent['source_root'])
    old_policy=Path(parent['source_policy'])
    parent['pins']={p:h for p,h in parent['pins'].items() if (not Path(p).is_relative_to(old_source)
        or Path(p).is_relative_to(parent['schedule']) or Path(p).is_relative_to(parent['parent_schedule']))
        and not Path(p).is_relative_to(old_policy)}
    fresh = c.get('fresh_policy')
    if fresh:
        from multimodal_jev.context_model_profile import from_hf, validate_against_config
        from multimodal_jev.policy_identity import checkpoint_identity

        base_id, files = checkpoint_identity(init['model'], io_workers=c['io_workers'])
        if base_id != fresh['base_id'] or files['model.safetensors'] != init['weights_sha256']:
            raise ValueError('fresh base does not match the registered model')
        validate_against_config(from_hf(init['model']), design)
        prior = load(fresh['plan']['path'],fresh['plan']['sha256'])
        new = json.loads((entry/'card-plan.json').read_bytes())
        if dict(prior,design_sha256=new['design_sha256']) != new:
            raise ValueError('cache reuse requires identical card panels and per-recording assignments')
        source_base = Path(parent['model'])
        parent['pins'] = {p:h for p,h in parent['pins'].items() if not Path(p).is_relative_to(source_base)}
        parent['pins'].update({str(Path(init['model'])/p):h for p,h in files.items()})
        parent.update(model=init['model'],base_id=base_id)
        for key in ('source_policy', 'source_policy_id', 'source_contract', 'stage_lora_transition'):
            parent.pop(key, None)
        init = dict(init, base_id=base_id)
        if c['seed'] != init['seed'] or c['deployment'].get('prepared_input') or 'bootstrap_input' in c['window']:
            raise ValueError('fresh policy requires its own policy and bootstrap')
        if design['stage1']['lr_schedule'] != parent['lr_schedule']:
            raise ValueError('different learning rate from parent runtime')
    commit_json(entry/'init.json',init)
    if extension:
        from multimodal_jev.stage1_mixed_schedule import schedule
        real = [json.loads(line) for line in (Path(parent['release'])/'learning.jsonl').open()]
        pop = {r['id']:r for r in real+extension[0]}
        batch_config = dict(d['batch_config'])
        if 'effective_anchors_per_rank' in c['window']:
            batch_config.update(per_dp_replica_batch=c['window']['effective_anchors_per_rank'],
                                microbatch=c['window']['execution_microbatch'])
        if 'batch_rule' in c['window']:
            devices = c['deployment']['devices']
            if (not devices or len(set(devices)) != len(devices) or
                    any(type(d) is not int or d < 0 for d in devices)):
                raise ValueError('unique physical training devices required')
            if len(devices) != c['window']['training_npus']:
                raise ValueError('physical device count differs from the registered budget')
            if c['window'].get('capacity_widths') != [batch_config['microbatch']]:
                raise ValueError('a changed microbatch needs a new schedule and package')
            batch_config.update(batch_rule=c['window']['batch_rule'], training_devices=len(devices))
            parent['physical_devices'] = devices
            derive(batch_config)
        _, schedule_proof = schedule(pop,batch_config,c['seed'])
        if c.get('bind_design_budget') and (batch_config != d['batch_config']
                or schedule_proof['updates'] != d['total_updates'] or schedule_proof['population'] != d['batch_config']['source_queries']):
            raise ValueError('published design and executed schedule differ')
        commit_json(entry/'mixed-schedule.json',schedule_proof)
        parent.update(mixed=dict(real_release_sha256=parent['release_sha256'],seed=c['seed'],
                      input_limit=m['runtime']['max_tokens']),audio_mirror_root=None,
                      batch_config=batch_config,release_sha256=m['admission_sha256'],
                      schedule_sha256=file_hash(entry/'mixed-schedule.json'),
                      parent_schedule_sha256=m['admission_sha256'],schedule_release_sha256=m['admission_sha256'])
    commit_json(entry/'training.json',parent)
    frozen=freeze(repo,entry/'source',patterns=c['source_patterns'],io_workers=c['io_workers'])
    shutil.copyfile(repo/'configs/portable/operations/native-seal-container.json',entry/'native-seal-container.json')
    deploy=c['deployment'].copy()
    devices=parent['physical_devices']
    groups=[]
    sample=json.loads((Path(parent['release'])/'learning.jsonl').open().readline())
    for role in ('prepare','finite','formal'):
        spec=dict(phase='card_stage1', role=role, entry='/jev-entry',work='/jev-work',source=dict(
            root='/jev-entry/source',sha256=frozen['manifest_sha256']),sample=dict(
            path=sample['audio'] if extension else str(Path(parent['audio_mirror_root'])/'audio'/(sample['audio_sha256']+'.audio')),sha256=sample['audio_sha256']),
            io_workers=c['io_workers'],prepared_input=deploy.get('prepared_input'),**c['window'])
        if fresh:
            spec['fresh_card_plan_sha256'] = fresh['plan']['sha256']
        commit_json(entry/(role+'.json'),dict(schema='jev-local-macros-v1',macros=dict(NATIVE_SEAL_CONTAINER_CONFIG=spec)))
        groups.append([dict(name=role,devices=[] if role=='prepare' else devices,
            collect=role+'/receipt.json',**c['resources'])])
    archive=root/'entry.tar'
    with tarfile.open(archive,'x') as tar:
        tar.add(entry,arcname='entry')
    deploy.update(archive=str(archive),entry_archive_sha256=file_hash(archive),
        entry_pins={str(p.relative_to(entry)):file_hash(p) for p in [*entry.glob('*.json'),entry/'source/FILES.json']},
        chains=[groups],device_group_lease=True,wait_for_lease=True,idle_devices=devices)
    commit_json(root/'operations-local.json',dict(schema='jev-local-macros-v1',macros=dict(NATIVE_SEAL_OPERATIONS_CONFIG=deploy)))


if __name__=='__main__':
    run(load_config(*sys.argv[1:])[0])
