"""Seal matched batch1/batch4 training prefixes; no remote calls or dispatch."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from multimodal_jev.local_config import load_config
from tools.reports.prepare_context_dual_training_candidate import rebase


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def prepare(c,provenance):
    source=Path(c['source']);output=Path(c['output'])
    receipt=json.loads((source/'PREPARED.json').read_text())
    if sha(source/'PREPARED.json')!=c['source_sha256']:
        raise ValueError('accepted integrated package identity changed')
    actual={str(p.relative_to(source)):sha(p) for p in source.rglob('*')
            if p.is_file() and p!=source/'PREPARED.json'}
    if any(p.is_symlink() for p in source.rglob('*')) or actual!=receipt['files']:
        raise ValueError('integrated payload changed')
    config=json.loads((source/'config.json').read_text())
    service=json.loads((source/'service-config.json').read_text())
    if (receipt['status']!='integrated_prepared_only' or
            config['start_update']!=0 or config['stop_update']!=2 or
            config['engineering_prefix'] is not True or config['training_dispatch_allowed'] is not False or
            config['algorithm']!='dual_credit_rloo_clipped_v1' or config['dual_arm']!='joint' or
            config['checkpoint_interval']!=1 or config['global_batch']!=128 or config['microbatch']!=8):
        raise ValueError('matched two-update joint RLOO candidate required')
    old=receipt['remote_root'];remote=c['remote_root']
    if (not Path(remote).is_absolute() or '..' in Path(remote).parts or remote==old or
            Path(remote).is_relative_to(old) or Path(old).is_relative_to(remote) or
            output.exists() or output.resolve().is_relative_to(source.resolve()) or
            source.resolve().is_relative_to(output.resolve())):
        raise ValueError('new independent comparison paths required')
    output.mkdir(parents=True)
    arms={}
    for name,batch in (('serial',1),('batch4',4)):
        dest=output/name;dest.mkdir();target=remote+'/'+name
        shutil.copytree(source/'source',dest/'source')
        for filename in ('run.py','RUNTIME-SOURCES.json'):
            shutil.copy2(source/filename,dest/filename)
        actor=rebase(config,old,target);server=rebase(service,old,target)
        # Both start with separate empty reference caches. Sharing between
        # timed arms would give the second arm an unearned warm-cache gain.
        actor['reference_cache']=target+'/reference-cache'
        server['policy']['max_batch_queries']=batch
        server['policy']['max_batch_tokens']=16384
        server['pins']=dict(actor['pins'])
        write(dest/'config.json',actor);write(dest/'service-config.json',server)
        record=dict(receipt,remote_root=target,
            parent_sha256=c['source_sha256'],config_provenance=provenance,
            comparison=dict(arm=name,max_batch_queries=batch,reference_condition='empty_at_start'),
            files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()})
        write(dest/'PREPARED.json',record)
        arms[name]=dict(package=str(dest.resolve()),remote_root=target,
            manifest_sha256=sha(dest/'PREPARED.json'),max_batch_queries=batch)
    result=dict(schema='context-dual-throughput-pair-v1',status='prepared_only',
        source_manifest_sha256=c['source_sha256'],arms=arms,
        same=['source','image','initial_policy','schedule','objective','global128_micro8_DP4',
              'two_updates','checkpoint_every_update','token_budget16384'],
        changed=['max_batch_queries_1_vs_4','isolated_output_paths','separate_empty_reference_caches'],
        timed_scope='slowest-rank full update: input, reference, behavior, actor, checkpoint, activation',
        limitations=['two updates are an engineering prefix, not a sustained throughput claim',
                     'batch1 uses the new segmented runtime; old-runtime parity is a separate probe',
                     'startup and dataset initialization require separate wall-time reporting'],
        training_dispatch_allowed=False,switch_old_training_allowed=False)
    write(output/'PAIR.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    r=prepare(c,provenance);print(json.dumps(dict(status=r['status'],arms=r['arms']),indent=2))
