"""Package TPO windows for the shared isolated container launcher."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.source_snapshot import freeze, verify
from multimodal_jev.tpo_anchor_layout import partition, window_device_plan
from tools.reports.build_tpo_view import verified


def macros(k,v):return dict(schema='jev-local-macros-v1',macros={k:v})


def prepare(c):
    pkg,repo=Path(c['package']),Path(c['repository'])
    orig=Path(c['parent_config'])
    full=c.get('full_joint')
    joint=c.get('joint')
    allocation = window_device_plan(c) if full or joint else None
    cards = c.get('card_input')
    parent=json.loads((verified(orig,full['runtime_sha256']) if full else orig).read_bytes())
    if not full:
        data=Path(c['targets'])
        m=json.loads((data/'MANIFEST.json').read_bytes())
        if (m['source_policy']!=parent['source_policy_id'] or
                m['view_sha256']!=parent['training_dataset']['manifest_sha256']):
            raise ValueError('incompatible target dataset')
        if 'actor_pair_margin' in m:
            if m.get('text_screened_new_view') is True:
                evidence = m['text_screen']
                if (evidence['schema'] != 'jev-source-reference-text-screen-v1'
                        or evidence['reference_covered'] != m['recordings']
                        or not {'pairs.json', 'text-screen.json'} <= m['files'].keys()):
                    raise ValueError('frozen text evidence required before packaging')
            elif m.get('reviewed_new_view') is True:
                evidence = m['review']
            else:
                raise ValueError('missing evidence for the new pair population')
            if (evidence['complete_pairs'] < 1 or evidence['existing_recordings_new_to_view'] < 1
                    or m['updates'] < 3 or not all(m['actor_pair_margin']['eligible_by_epoch'])):
                raise ValueError('new pair population required before freezing actor margin training')
            for name, sha in m['files'].items():
                verified(data/name, sha)
        if cards:
            if (full or joint or c.get('next_macro_reference') or not m.get('card_input')
                    or m['reference_sha256'] != parent['reference_dataset']['manifest_sha256']
                    or m['source_policy_sha256'] != cards['policy_sha256']
                    or m['schedule']['global_batch'] != len(parent['physical_devices'])*parent['batch_config']['per_dp_replica_batch']):
                raise ValueError('invalid fixed card TPO deployment')
        else:
            remote_parent=Path(parent['source_policy']).parents[1]
            remote_ref=parent['reference_dataset']['root']
    entry=pkg/'entry'
    entry.mkdir(parents=True,exist_ok=False)
    if full:
        from tools.reports.prepare_full_joint_tpo import publish
        parent,plan=publish(c,parent,entry,repo)
    if c.get('reuse_frozen_source'):
        archived = Path(c['reuse_frozen_source']['path'])
        expected = c['reuse_frozen_source']['sha256']
        verify(archived, manifest_sha256=expected, io_workers=c['io_workers'])
        shutil.copytree(archived, entry/'source')
        verify(entry/'source', manifest_sha256=expected, io_workers=c['io_workers'])
        frozen = {'manifest_sha256': expected}
    else:
        frozen=freeze(repo,entry/'source',patterns=c['source_patterns'],io_workers=c['io_workers'])
    shutil.copyfile(repo/'configs/portable/operations/native-seal-container.json',entry/'native-seal-container.json')
    ctl=c['deployment'].copy()
    devices=parent['physical_devices']
    world=len(devices)
    geometry=parent['view_execution']['container']
    groups=[]
    if joint or full:
        if joint:
            from tools.reports.prepare_joint_tpo import prepare as prepare_joint
            plan=prepare_joint(joint,entry,data,orig,parent,repo)
        parent_in='/jev-entry/parent.json' if full else str(remote_parent/'config.json')
        offline = allocation['fixed_reference']
        if (plan['training_devices'] != devices or devices != allocation['actor']
                or plan['inference_devices'] != allocation['online_reference']):
            raise ValueError('Packaged training geometry differs from the device plan')
        for steps in ([2, plan['updates']] if full else [plan['updates']]):
            partition(world, steps, len(offline))
        plan['offline_reference_devices'] = offline
        commit_json(entry/'joint.json',plan)
        digest=file_hash(entry/'joint.json')
        infer=plan['inference_devices']
        stages=([('finite-anchor','finite-anchor',offline),('finite','finite',devices+infer),('anchor','anchor',offline)]
                if full else [('anchor','anchor',offline),('finite','finite',devices+infer)])
        for role,window,scope in [*stages,('tpo','formal',devices+infer)]:
            opts=dict(phase='joint_tpo',source=dict(root='/jev-entry/source',sha256=frozen['manifest_sha256']),
                io_workers=c['io_workers'],training=dict(parent=parent_in,plan='/jev-entry/joint.json',
                output='/jev-work/'+role,joint_sha=digest,window=window))
            commit_json(entry/(role+'.json'),macros('NATIVE_SEAL_CONTAINER_CONFIG',opts))
            groups.append([dict(name=role,devices=scope,cpus=geometry['cpus'],memory=f"{geometry['memory_gib']}g",
                shm=f"{geometry['shm_gib']}g",collect=role+'/receipt.json')])
        archive=pkg/'entry.tar'
        with tarfile.open(archive,'x') as t:t.add(entry,arcname='entry')
        ctl.update(archive=str(archive),chains=[groups],read_only_mounts=ctl.get('read_only_mounts', []) if full else [
            str(remote_parent)+':'+str(remote_parent),joint['control_root']+':/jev-control'],
            device_group_lease=True,wait_for_lease=True)
        if full:
            ctl['entry_archive_sha256']=file_hash(archive)
            ctl['entry_pins']={str(p.relative_to(entry)):file_hash(p) for p in (
                entry/'joint.json',entry/'parent.json',entry/'native-seal-container.json',
                entry/'initial/policy/MANIFEST.json',entry/'targets/MANIFEST.json',
                *(entry/(j[0]['name']+'.json') for j in groups))}
            ctl['idle_devices']=devices+infer
        commit_json(pkg/'operations-local.json',macros('NATIVE_SEAL_OPERATIONS_CONFIG',ctl))
        return
    shutil.copytree(data,entry/'targets')
    windows = [('finite','tpo','finite'),('tpo','tpo','formal')]
    parity = []
    if cards:
        shutil.copyfile(orig, entry/'parent.json')
        parent_in = '/jev-entry/parent.json'
        inference = cards['inference_devices']
        if len(inference) != 2 or len(set(devices+inference)) != len(devices)+len(inference):
            raise ValueError('two independent inference cards are required')
        probes = []
        for index, device in enumerate(inference):
            name = f'infer-{index}'
            receipt = f'{name}/receipt.json'
            scope = dict(kind='same_container_runtime_visibility_v1', mounted_physical=[device],
                         visible_ordinal='0')
            opts = dict(phase='card_tpo_parity', source=dict(root='/jev-entry/source',sha256=frozen['manifest_sha256']),
                io_workers=c['io_workers'], training=dict(physical_device=device, probe_start=index*4,
                process_device_scope=scope,
                parent=parent_in, parent_sha=file_hash(orig), targets='/jev-entry/targets',
                target_sha=file_hash(data/'MANIFEST.json'), output=f'/jev-work/{name}'))
            commit_json(entry/(name+'.json'),macros('NATIVE_SEAL_CONTAINER_CONFIG',opts))
            parity.append(dict(path='/jev-work/'+receipt, physical_device=device))
            probes.append(dict(name=name,devices=[device],cpus=cards['inference_cpus'],memory=f"{geometry['memory_gib']}g",
                               shm=f"{geometry['shm_gib']}g",collect=receipt,
                               visible_ordinal=scope['visible_ordinal']))
        groups.append(probes)
    else:
        parent_in = str(remote_parent/'config.json')
    dynamic=c.get('next_macro_reference')
    if dynamic:
        from tools.reports.tpo_next_macro_reference import prepare as prepare_next
        prepare_next(dynamic,entry,data,m,file_hash(orig),world,repo)
    if not dynamic and m['schema'] == 'jev-tpo-choices-v1':
        windows.insert(0,('reference','tpo','reference'))
    if m['schema'] == 'jev-tpo-view-v1' and 'ranking_repair' not in m:
        windows.append(('direct','direct','formal'))
    for name,arm,window in windows:
        opts=dict(phase='tpo_train',source=dict(root='/jev-entry/source',sha256=frozen['manifest_sha256']),
                  io_workers=c['io_workers'],training=dict(world_size=world,parent=parent_in,
                  targets='/jev-entry/targets',output='/jev-work/'+name,parent_sha=file_hash(orig),
                  target_sha=file_hash(data/'MANIFEST.json'),arm=arm,window=window,finite_receipt='/jev-work/finite/receipt.json'))
        if cards:
            opts['training']['card_inference_parity'] = parity
        if dynamic:
            opts['training']['next_macro_sha']=file_hash(entry/'next-macro.json')
        commit_json(entry/(name+'.json'),macros('NATIVE_SEAL_CONTAINER_CONFIG',opts))
        groups.append([dict(name=name,devices=devices,cpus=geometry['cpus'],memory=f"{geometry['memory_gib']}g",
                    shm=f"{geometry['shm_gib']}g",collect=name+'/receipt.json')])
    archive=pkg/'entry.tar'
    with tarfile.open(archive,'x') as t:t.add(entry,arcname='entry')
    mounts = ctl['read_only_mounts'] if cards else [str(remote_parent)+':'+str(remote_parent),remote_ref+':'+remote_ref]
    ctl.update(archive=str(archive),chains=[groups],read_only_mounts=mounts,
               device_group_lease=True,wait_for_lease=True)
    if 'ranking_repair' in m or 'bidirectional_contrast' in m:
        ctl['entry_archive_sha256'] = file_hash(archive)
        names = ['source/FILES.json', 'finite.json', 'tpo.json', 'targets/MANIFEST.json',
                 *(f'targets/{name}' for name in m['files'])]
        if cards:
            names += ['parent.json', *(j['name']+'.json' for j in groups[0])]
            ctl.pop('idle_devices', None)
        else:
            ctl['idle_devices'] = devices
        ctl['entry_pins'] = {name: file_hash(entry/name) for name in names}
    if dynamic:
        ctl['read_only_mounts'].append(dynamic['control_root']+':/jev-control')
    commit_json(pkg/'operations-local.json',macros('NATIVE_SEAL_OPERATIONS_CONFIG',ctl))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('config');ap.add_argument('local');a=ap.parse_args()
    config = load_config(a.config,a.local)[0]
    # Under run_experiment.py this is the committed checkout, even when the
    # operator also has a mutable working checkout on the same machine.
    if config.get('full_joint') or config.get('joint') or config.get('freeze_committed_repository'):
        config['repository'] = str(Path(__file__).resolve().parents[2])
    prepare(config)
