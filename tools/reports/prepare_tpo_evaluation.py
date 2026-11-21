"""Bind predeclared policy endpoints to independently verified test inputs."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.source_snapshot import freeze
from multimodal_jev.model_provider import cache_namespace, require_same_provider
from tools.reports.prepare_tpo_windows import macros


def read(p):return json.loads(Path(p).read_bytes())


def evaluation_cache(source, dataset, parent, initial):
    if 'cache_config' in source:
        ref = source['cache_config']
        path = Path(ref['path'])
        if file_hash(path) != ref['sha256']:
            raise ValueError('evaluation cache config changed')
        cache = read(path)
        require_same_provider(parent, cache)
        root = Path(cache['output'])
        manifest = root/'MANIFEST.json'
        verification = read(root/'VERIFIED.json')
        metadata = read(manifest)
        if (metadata['schema'] != 'jev-native-seal-encoded-cache-v1'
                or metadata['encoding'] != cache['encoding']
                or metadata['seal_sha256'] != cache['seal']['sha256']
                or metadata['processor_sha256'] != cache['processor']['sha256']
                or metadata.get('model_provider_sha256') != cache_namespace(parent)
                or verification['status'] != 'full_stable_prefix_cache_verified'
                or verification['recordings'] != metadata['recordings']
                or verification['manifest_sha256'] != file_hash(manifest)):
            raise ValueError('model-specific evaluation cache requires a matching independent receipt')
        if initial is not None:
            from multimodal_jev.cached_context_preparation import processor_cache_identity
            if cache['processor']['sha256'] != processor_cache_identity(initial):
                raise ValueError('evaluation processor differs from the initialized model')
    elif 'scored_cache' in source:
        ref = source['scored_cache']
        if file_hash(ref['path']) != ref['sha256']:
            raise ValueError('previously scored cache configuration changed')
        cache = read(ref['path'])['macros']['NATIVE_SEAL_CONTAINER_CONFIG']['endpoints'][dataset]['cache']
        require_same_provider(parent, cache)
        if initial is not None:
            from multimodal_jev.cached_context_preparation import processor_cache_identity
            if cache['processor']['sha256'] != processor_cache_identity(initial):
                raise ValueError('evaluation processor differs from the initialized model')
    else:
        old = Path(source['package'])
        endpoint = read(old/'entry/full-102.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']['endpoint']
        cache = dict(endpoint['cache'])
        require_same_provider(parent, cache)
        old_ops = read(old/'operations-local.json')['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
        cache['output'], = [p.split(':')[0] for p in old_ops['read_only_mounts']
                            if p.endswith(':/jev-inputs/full-cache')]
    if cache.get('panel_size') is not None or cache['dataset'] != dataset:
        raise ValueError('full frozen test required')
    return cache


def prepare(config):
    pkg,repo = Path(config['package']),Path(config['repository'])
    ops=Path(config['training_ops'])
    prior=read(config['training_operations_local'])['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    if 'expected_host' in config and prior['host'] != config['expected_host']:
        raise ValueError('evaluation host differs from the registered deployment')
    job=read(Path(config['training_operations_local']).parent/'entry/finite.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    training_entry = Path(config['training_operations_local']).parent/'entry'
    joint = None
    if 'joint_sha' in job['training']:
        plan_path = training_entry/'joint.json'
        if file_hash(plan_path) != job['training']['joint_sha']:
            raise ValueError('joint training plan changed')
        joint = read(plan_path)
        binding = dict(joint_sha=job['training']['joint_sha'], target_sha=joint['target_sha'],
                       parent_sha=joint['parent_sha'], world_size=len(joint['training_devices']),
                       initial_policy=joint['source_policy'])
    else:
        binding={k:job['training'][k] for k in ('target_sha','parent_sha','world_size')}
    if 'next_macro_sha' in job['training']:
        binding['next_macro_sha'] = job['training']['next_macro_sha']
    binding['source']=job['source']['sha256']
    if (ops/'finite-collected.json').exists() and read(ops/'finite-collected.json')['binding']!=binding:
        raise ValueError('finite receipt differs from the sealed training package')
    orig=Path(config['parent_config'])
    parent=read(orig)
    full = joint is not None and joint.get('initialization') == 'pre_task_policy'
    steps = joint['updates'] if joint else parent['total_updates']
    remote_parent = Path(prior['remote_root'])/'entry' if full else Path(parent['source_policy']).parents[1]
    parent_file = remote_parent/('parent.json' if full else 'config.json')
    if 'model_provider' in config:
        require_same_provider(config,parent)
    if file_hash(orig)!=binding['parent_sha']:
        raise ValueError('parent identity changed')
    entry=pkg/'entry'
    entry.mkdir(parents=True,exist_ok=False)
    frozen=freeze(repo,entry/'source',patterns=config['source_patterns'],io_workers=config['io_workers'])
    shutil.copyfile(repo/'configs/portable/operations/native-seal-container.json',entry/'native-seal-container.json')
    common=dict(source=dict(root='/jev-entry/source',sha256=frozen['manifest_sha256']),io_workers=config['io_workers'])
    if full:
        if parent['source_policy'] != '/jev-entry/initial/policy':
            raise ValueError('unexpected single-stage policy layout')
        initial = training_entry/'initial/policy'
        if file_hash(initial/'MANIFEST.json') != joint['source_manifest_sha256']:
            raise ValueError('step-zero policy changed')
        # Recreate the sealed parent's relative container layout for evaluation.
        shutil.copytree(initial, entry/'initial/policy')
    if full and 'model_provider' in parent and config.get('full_ontology_only'):
        if ('choice_protocol' in config or 'choice_baseline' in config or 'paired_lengths' in config
                or set(config.get('arms', ())) != {'stage1', 'tpo', *config.get('checkpoint_steps', {})}):
            raise ValueError('use the bound Joint TPO objective and its policy endpoints')
        common.update(choice={'temperature':joint['objective']['temperature']},full_ontology_only=True)
    elif joint and config.get('full_ontology_only') and 'choice_protocol' not in config:
        if 'paired_lengths' in config:
            raise ValueError('full ontology and variable Choice must be scored separately')
        common.update(choice={'temperature':joint['objective']['temperature']},full_ontology_only=True)
        if config.get('external_comparisons') is True:
            if 'choice_baseline' in config:
                raise ValueError('external comparisons cannot also rerun a baseline')
            common['external_comparisons'] = True
        elif 'choice_baseline' in config:
            common['choice_baseline'] = config['choice_baseline']
        elif 'stage1' not in config.get('arms', ()):
            raise ValueError('register an audited Choice comparison endpoint')
    elif 'choice_protocol' in config:
        spec=read(config['choice_protocol'])
        common.update(choice=spec['choice'],choice_baseline=config['choice_baseline'])
        if config.get('full_ontology_only'):
            if 'paired_lengths' in config or file_hash(config['choice_protocol'])!=binding['target_sha']:
                raise ValueError('expected bound full-ontology protocol')
            common['full_ontology_only']=True
        if 'paired_lengths' in config:
            if file_hash(config['choice_protocol'])!=binding['target_sha']:raise ValueError('Choice identity differs')
            common['paired_lengths']=config['paired_lengths']
    if full and ('arms' not in config or 'direct' in config['arms']):
        raise ValueError('single-stage run has only initial and TPO endpoints')
    if 'arms' in config:
        common['arms']=config['arms']
    checkpoint_steps = config.get('checkpoint_steps', {})
    if checkpoint_steps:
        if (not full or not config.get('training_custody')
                or set(checkpoint_steps.values()) - set(joint['intermediate_export_updates'])
                or any(arm != f'step-{step}' for arm, step in checkpoint_steps.items())
                or set(checkpoint_steps) - set(common['arms'])):
            raise ValueError('only audited pre-registered intermediate checkpoints may be evaluated')
        proof = read(Path(config['training_custody'])/'formal-audit.json')
        if (proof['status'] != 'formal_training_collected_and_audited' or proof['updates'] != steps
                or proof['endpoint'] != read(ops/'tpo-collected.json')['endpoint']):
            raise ValueError('formal checkpoint audit required')
        common['checkpoints'] = {arm: proof['checkpoints'][f'policy-{arm}'] for arm in checkpoint_steps}
    jobs=[];mounts={str(remote_parent),prior['remote_root']};sets={};pieces={}
    if 'choice_baseline' in common:mounts.add(common['choice_baseline']['root'])
    datasets=config['datasets']
    if 'formal_datasets' in config:
        selected=config['formal_datasets']
        if not selected or len(set(selected))!=len(selected):
            raise ValueError('unique formal dataset IDs required')
        datasets={key:datasets[key] for key in selected}
    for dataset,source in datasets.items():
        cache=evaluation_cache(source,dataset,parent,initial if full else None)
        if 'seal_sha256' in source and cache['seal']['sha256'] != source['seal_sha256']:
            raise ValueError('comparison and evaluated population must share the registered seal')
        mounts.add(cache['output'])
        if 'cache_config' in source or 'scored_cache' in source:
            mounts.add(cache['seal']['root'])
        c=dict(cache=cache,cache_verification=cache['output']+'/VERIFIED.json',audio_token_id=cache['encoding']['audio_token_id'] if full else parent['audio_token_id'])
        sets[dataset]=c;pieces[dataset]=[]
        for i,device in enumerate(source['devices']):
            name=f"{source['name']}-{i}"
            opts=dict(common,phase='tpo_eval',endpoint=c,devices=[device],shard=i,shards=len(source['devices']),
                trained=prior['remote_root'],output='/jev-work/'+name,batch=config['batch'],binding=binding,
                updates=steps,parent=str(parent_file),parent_sha=binding['parent_sha'])
            commit_json(entry/(name+'.json'),macros('NATIVE_SEAL_CONTAINER_CONFIG',opts))
            pieces[dataset].append(opts['output'])
            jobs.append(dict(name=name,devices=[device],**config['eval_container']))
    scoring=dict(common,phase='tpo_score',output='/jev-work/metrics.json',endpoints=sets,shards=pieces,
                 steps=dict(stage1=config['stage1_step'],direct=steps,tpo=steps,**checkpoint_steps))
    if config.get('publish_seals'):
        scoring['publish_seals'] = True
    commit_json(entry/'score.json',macros('NATIVE_SEAL_CONTAINER_CONFIG',scoring))
    archive=pkg/'entry.tar'
    with tarfile.open(archive,'x') as tar:tar.add(entry,arcname='entry')
    ctl={k:prior[k] for k in ('host','lease','entrypoint','image','driver_root','shared_root','transfer_timeout')}
    if 'training_custody' in config:
        custody = Path(config['training_custody'])
        if read(custody/'operations.json') != prior:
            raise ValueError('training custody belongs to a different sealed operation')
        predecessor = dict(driver=read(custody/'started.json'), terminal=str(custody/'terminal.json'),
                           required_status='formal_training_collected_and_audited')
    else:
        predecessor = dict(driver=read(ops.with_suffix('.started.json')), terminal=str(ops/'terminal.json'))
    ctl.update(config['deployment'],archive=str(archive),read_only_mounts=[p+':'+p for p in sorted(mounts)],
        device_group_lease=True,wait_for_lease=True,predecessor=predecessor,
        chains=[[jobs,[dict(name='score',devices=[],collect='metrics.json',**config['eval_container'])]]])
    if config.get('publish_seals'):
        ctl['entry_archive_sha256'] = file_hash(archive)
        ctl['entry_pins'] = {f'{job["name"]}.json': file_hash(entry/f'{job["name"]}.json') for job in jobs}
    commit_json(pkg/'operations-local.json',macros('NATIVE_SEAL_OPERATIONS_CONFIG',ctl))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('local');a=p.parse_args()
    prepare(load_config(a.config,a.local)[0])
