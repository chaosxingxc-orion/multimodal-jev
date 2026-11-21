"""Package the paired TRAIN probe for the existing native CPU/HF runtime."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.source_snapshot import verify
from tools.reports.prepare_native_seal_cache import plan
from tools.reports.jev_train_confusions import pinned


def bind(key,value):
    return dict(schema='jev-local-macros-v1',macros={key:value})


def reuse_prep(c, manifest, policy):
    binding = c['reuse_prep']
    root = Path(binding['root'])
    receipt = json.loads(pinned(root/'receipt.json', binding['receipt_sha256']))
    if receipt['policy'] != policy or receipt['seal_sha256'] != c['seal_sha256']:
        raise ValueError('prior CPU cache must bind the same endpoint and panel')
    config = json.loads(pinned(root/'prep/local.json', receipt['packages']['prep']['local_sha256']))
    operations = config['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    if operations['host'] != c['machine']['host'] or operations['image'] != c['machine']['image']:
        raise ValueError('prior CPU cache host and image differ')
    collected = Path(operations['operations'])
    if json.loads((collected/'terminal.json').read_bytes()) != dict(
            exit_code=0, status='declared_stages_finished'):
        raise ValueError('complete prior CPU preparation required')
    for arm, spec in manifest['datasets'].items():
        verified = json.loads((collected/f'verify-{arm}-collected.json').read_bytes())
        if verified['status'] != 'full_stable_prefix_cache_verified' or verified['recordings'] != spec['recordings']:
            raise ValueError('prior CPU cache does not cover the complete arm: ' + arm)
    return operations['remote_root']


def prepare(c):
    out=Path(c['output']);check_paths({'package':out});out.mkdir(parents=True,exist_ok=False)
    parent=Path(c['parent']); seal=Path(c['seal'])
    manifest=json.loads(pinned(seal/'MANIFEST.json',c['seal_sha256']))
    arms=list(manifest['datasets'])
    if len(c['devices'])!=len(arms) or len(set(c['devices']))!=len(arms):raise ValueError('one device per arm')
    step=c.get('endpoint_step',102)
    if type(step) is not int or step<1:raise ValueError('positive endpoint step required')
    hf_name = c.get('hf_config', f'hf-{step}.json')
    if Path(hf_name).name != hf_name:
        raise ValueError('endpoint config must be in the frozen parent')
    h=json.loads(pinned(parent/hf_name,c['hf_config_sha256']))['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    ep=h['endpoint']
    source=json.loads(pinned(parent/'source-plan.json',ep['source_plan']['sha256']))
    snapshot = ep['snapshot'].removeprefix('/jev-entry/')
    if (source.get('fixed_terminal_step',source['checkpoints'][-1]['step'])!=step
        or ep['point']!=source['checkpoints'][-1] or ep['point']['step']!=step
        or ep['snapshot']!=f'/jev-entry/{snapshot}' or snapshot not in ('policy', f'policy-{step}')
        or file_hash(parent/snapshot/'MANIFEST.json')!=ep['point']['manifest_sha256']
        or (c.get('expected_policy_id',ep['point']['policy_id'])!=ep['point']['policy_id'])):
        raise ValueError('original policy is not the bound endpoint')
    runtime_source=Path(c.get('runtime_source',parent/'source'))
    verify(runtime_source,manifest_sha256=h['source']['sha256'],io_workers=c['io_workers'])
    result={}
    prep_root = (reuse_prep(c, manifest, ep['point']['policy_id']) if 'reuse_prep' in c
                 else c['remote_root']+'-prep')
    for phase in (('hf',) if 'reuse_prep' in c else ('prep', 'hf')):
        work=out/phase;entry=work/'entry';entry.mkdir(parents=True)
        shutil.copytree(runtime_source,entry/'source')
        shutil.copy(parent/'native-seal-container.json',entry)
        (entry/snapshot).mkdir()
        (entry/'panel').mkdir()
        for rel in ['MANIFEST.json',*[r for r in manifest['files'] if r.startswith('inference/')]]:
            dest=entry/'panel'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy(seal/rel,dest)
        ops={**c['machine'], **c.get(phase+'_operation_options', {})}
        if phase == 'hf' and 'hf_predecessor' in c:
            ops['predecessor'] = c['hf_predecessor']
        ops.update(remote_root=c['remote_root']+'-'+phase,tag=c['tag']+'-'+phase,
                   lease=c[phase+'_lease'],operations=str(work/'run'),archive=str(work/'entry.tar.gz'))
        mounts=[c['snapshot_mount']+':'+ep['snapshot']]
        chains=[]
        for arm,dev in zip(arms,c['devices'],strict=True):
            n=manifest['datasets'][arm]['recordings']
            cache=dict(ep['cache'],dataset=arm,panel_size=n,output='/jev-work/cache-'+arm,
                       seal=dict(root='/jev-entry/panel',sha256=c['seal_sha256']),
                       io_workers=c['io_workers'],prefetch=c['prefetch'])
            cache['processor']=dict(cache['processor'],policy=ep['snapshot'])
            if len(plan(dict(cache,seal=dict(root=str(seal),sha256=c['seal_sha256']))))!=n:
                raise ValueError('finite panel was truncated')
            chain=[]
            for role in (('encode','verify') if phase=='prep' else ('hf',)):
                job=role+'-'+arm
                cfg=dict(source=h['source'],io_workers=c['io_workers'],phase=role)
                if role=='hf':
                    cache['output']='/jev-cache-'+arm
                    mounts.append(prep_root+'/cache-'+arm+':'+cache['output'])
                    # Same registered endpoint, new TRAIN diagnostic evaluation contract.
                    sp=dict(source,evaluation_dataset=arm,seal_sha256=c['seal_sha256'],
                            population_role='source_train_diagnostic')
                    path=entry/f'plan-{arm}.json';commit_json(path,sp)
                    cfg['endpoint']=dict(ep,source_plan=dict(path='/jev-entry/'+path.name,sha256=file_hash(path)),
                        physical_devices=[dev],cache=cache,cache_verification=cache['output']+'/VERIFIED.json',
                        output='/jev-work/'+job)
                else: cfg['cache']=cache
                commit_json(entry/(job+'.json'),bind('NATIVE_SEAL_CONTAINER_CONFIG',cfg))
                spec=dict(name=job,devices=[dev] if role=='hf' else [],**c['resources'][phase])
                if role=='hf':spec['collect']=job+'/reference.json'
                elif role=='verify':spec['collect']='cache-'+arm+'/VERIFIED.json'
                chain.append([spec])
            chains.append(chain)
        if ops.get('device_group_lease'):
            if phase != 'hf':
                raise ValueError('CPU and NPU leasing must be configured separately')
            chains = [[[arm[0][0] for arm in chains]]]
        ops.update(chains=chains,read_only_mounts=mounts)
        with tarfile.open(ops['archive'],'w:gz',compresslevel=1) as tar:tar.add(entry,arcname='entry')
        commit_json(work/'local.json',bind('NATIVE_SEAL_OPERATIONS_CONFIG',ops))
        result[phase]=dict(archive_sha256=file_hash(ops['archive']),local_sha256=file_hash(work/'local.json'))
    commit_json(out/'receipt.json',dict(packages=result,policy=ep['point']['policy_id'],seal_sha256=c['seal_sha256'],
                **({'reuse_prep':c['reuse_prep']} if 'reuse_prep' in c else {})))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();print(json.dumps(prepare(load_config(a.config,a.local)[0])))
