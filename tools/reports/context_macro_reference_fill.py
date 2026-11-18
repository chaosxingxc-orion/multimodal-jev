"""Precompute fixed-reference probabilities for the frozen native TRAIN population.

A fixed-reference record depends only on the reference policy/base/contract and
the execution runtime identity (image, sources, processor, profile, protocol);
it never depends on the moving actor. Filling the retained content-addressed
cache once therefore converts the per-step reference pass into a cache read for
every arm and every later run that executes byte-identical sources.

This worker consumes only the label-free encoded audio cache. It writes no
labels, actions, reward or actor state, performs no optimizer update and starts
no engine of its own: it connects to the sealed dual services exactly like the
training client does.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_macro_rollout import digest
from multimodal_jev.context_policy_artifacts import read_execution_bundle, verify_native_adapter
from multimodal_jev.encoded_audio_cache import EncodedAudioCache, key_for
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.span_request_identity import tensor_identity
from tools.reports.audit_context_macro_training import audit_journal
from tools.reports.context_dual_runtime_candidate import verify_runtime
from tools.reports.context_dual_session_candidate import RuntimeReferenceCache, probability_record
from tools.reports.context_macro_client import connect_pool
from tools.reports.run_context_batch_probe_candidate import verify

SCHEMA='context-fixed-reference-fill-v1'
INPUT_KEYS={'input_ids','attention_mask','input_features','input_features_mask'}
RECORD_KEYS=('request_id','request_identity','prob','policy_id')
ENTRY_KEYS={'binding','inputs','positions','audio_receipt','tensor_hashes'}
CONFIG_KEYS={'schema','source_root','source_policy','source_policy_id','base_id','source_contract',
    'reference_adapter','dual_runtime_identity','dual_runtime_manifest','encoded_audio_cache',
    'audio_mirror_root','reference_cache','audio_token_id','expected_entries','macro_client','output','pins'}
MAX_ROWS=128


def read(path):
    return json.loads(Path(path).read_text())


def sha256_text(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def load_entry(path):
    """Authenticate one cached processor payload and return label-free inputs."""
    path=Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('plain encoded audio entry required')
    value=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(value,dict) or set(value)!=ENTRY_KEYS:
        raise ValueError('unexpected encoded audio payload')
    inputs,positions=value['inputs'],value['positions']
    EncodedAudioCache._validate(inputs,positions)
    # Macro rows carry exactly four processor tensors; position_ids is a separate
    # readout definition and would silently change the reference binding.
    if set(inputs)!=INPUT_KEYS:
        raise ValueError('exactly four label-free processor inputs required')
    identity=value['binding']
    if (not isinstance(identity,dict) or identity.get('schema')!='encoded-audio-cache-v1'
            or key_for(identity)!=path.stem):
        raise ValueError('encoded audio binding or entry name differs')
    actual={k:tensor_identity(v) for k,v in dict(inputs,positions=positions).items()}
    if actual!=value['tensor_hashes']:
        raise ValueError('encoded audio tensor integrity differs')
    return inputs,positions


def entry_paths(root,expected):
    if type(expected) is not int or expected<1:
        raise ValueError('positive expected entry count required')
    root=Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('encoded audio cache directory required')
    paths=[]
    for path in sorted(root.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ValueError('plain encoded audio files required')
        if path.suffix!='.pt' or not sha256_text(path.stem):
            raise ValueError('hashed encoded audio entry names required')
        paths.append(path)
    if len(paths)!=expected:
        raise ValueError('encoded audio entry count differs from the frozen population')
    return paths


def fill(config, provenance):
    root=Path(config['output']);check_paths({'reference_fill':root})
    if set(config)!=CONFIG_KEYS:
        raise ValueError('explicit reference fill configuration required')
    if config['schema']!=SCHEMA:
        raise ValueError('reference fill schema differs')
    if root.exists():
        raise ValueError('existing finite reference fill attempt')
    encoded=Path(config['encoded_audio_cache']);cache_root=Path(config['reference_cache'])
    if encoded.is_relative_to(root) or root.is_relative_to(encoded):
        raise ValueError('encoded audio cache must stay outside the fill root')
    if (not cache_root.is_absolute() or '..' in cache_root.parts or ':' in str(cache_root)
            or cache_root.is_relative_to(root) or root.is_relative_to(cache_root)):
        raise ValueError('retained reference cache must be disjoint from the fill root')
    for path,expected in config['pins'].items():
        if file_hash(path)!=expected:
            raise ValueError('pinned dependency changed: '+path)
    verify(Path(config['source_root']));verify_runtime(config)
    bundle=read_execution_bundle(config['source_policy'],expected_policy_id=config['source_policy_id'],
        expected_contract=config['source_contract'],expected_base_id=config['base_id'])
    verify_native_adapter(config['reference_adapter'],bundle)
    cache=RuntimeReferenceCache(cache_root,policy_id=bundle['policy_id'],base_id=config['base_id'],
        contract=config['source_contract'],runtime_identity=config['dual_runtime_identity'])
    paths=entry_paths(encoded,config['expected_entries'])
    # Content-key uniqueness is decided before any GPU work so a duplicate or
    # malformed population fails cheaply instead of half-way through the fill.
    unique=[];keys=set()
    for path in paths:
        inputs,positions=load_entry(path)
        key=cache.digest(cache.binding(inputs,positions))
        if key in keys:
            continue
        keys.add(key);unique.append(path)
    root.mkdir(parents=True)
    commit_json(root/'provenance.json',provenance)
    index={};filled=0;warm=0;started=time.monotonic();pool=None

    def record_index(record):
        key=record['request_id']
        if key in index:
            raise ValueError('duplicate fixed reference request id')
        index[key]=digest(record)

    try:
        pool=connect_pool(config['macro_client'],bundle=bundle)
        for start in range(0,len(unique),MAX_ROWS):
            chunk=unique[start:start+MAX_ROWS];work=[];cached=[]
            for path in chunk:
                inputs,positions=load_entry(path)
                record=cache.read(inputs,positions)
                if record is None:
                    work.append((inputs,positions))
                else:
                    if record['policy_id']!=bundle['policy_id']:
                        raise ValueError('retained reference policy differs')
                    cached.append(record)
            for record in cached:
                record_index(probability_record(record));warm+=1
            if work:
                rows=[dict(inputs=inputs,positions=positions) for inputs,positions in work]
                values=pool.score_many(rows,audio_token_id=config['audio_token_id'])
                if len(values)!=len(work) or pool.bundle['policy_id']!=bundle['policy_id']:
                    raise ValueError('incomplete or switched fixed reference batch')
                for (inputs,positions),record in zip(work,values,strict=True):
                    probability=probability_record(record)
                    if probability['policy_id']!=bundle['policy_id']:
                        raise ValueError('scored reference policy differs')
                    cache.write(probability,inputs,positions)
                    record_index(probability);filled+=1
        if len(index)!=len(unique):
            raise ValueError('incomplete fixed reference index')
        seconds=time.monotonic()-started
        commit_json(root/'reference-index.json',{k:index[k] for k in sorted(index)})
        result=dict(schema=SCHEMA,policy_id=bundle['policy_id'],base_id=config['base_id'],
            contract_digest=digest(config['source_contract']),
            runtime_identity=config['dual_runtime_identity'],reference_identity=cache.identity,
            namespace=cache.namespace.name,entries=len(paths),unique=len(unique),filled=filled,warm=warm,
            seconds=seconds,index_sha256=file_hash(root/'reference-index.json'),
            audio_token_id=config['audio_token_id'],actor_update_performed=False,
            scope='Fixed Stage1 reference probabilities over the frozen TRAIN population; no labels, reward, optimizer or actor state.')
        commit_json(root/'reference-fill.json',result)
        pool.close();pool=None
        commit_json(root/'terminal.json',dict(exit_code=0))
        return result
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc)))
        raise
    finally:
        if pool is not None:
            try: pool.close()
            except Exception: pass


def audit_fill(package, collection):
    """Independently re-audit the fill journal and its published index."""
    package,collection=Path(package),Path(collection);c=read(package/'fill.json')
    run=collection/'run';client=c['macro_client']
    if (c['schema']!=SCHEMA or type(c['expected_entries']) is not int or c['expected_entries']<1):
        raise ValueError('sealed reference fill configuration required')
    if read(run/'terminal.json')['exit_code']!=0:
        raise ValueError('successful reference fill worker required')
    result=read(run/'reference-fill.json');index=read(run/'reference-index.json')
    if (not isinstance(result['reference_identity'],dict)
            or result['schema']!=SCHEMA or result['policy_id']!=c['source_policy_id']
            or result['base_id']!=c['base_id']
            or result['runtime_identity']!=c['dual_runtime_identity']
            or result['entries']!=c['expected_entries']
            or type(result['unique']) is not int or result['unique']!=len(index)
            or result['filled']+result['warm']!=result['unique']
            or result['actor_update_performed'] is not False
            or file_hash(run/'reference-index.json')!=result['index_sha256']
            or result['reference_identity'].get('runtime_identity')!=c['dual_runtime_identity']
            or Path(result['namespace']).name!=result['namespace']):
        raise ValueError('complete reference fill receipt required')
    if (any(not sha256_text(k) for k in index)
            or any(not sha256_text(v) for v in index.values())):
        raise ValueError('hashed fixed reference index required')
    journal=collection/'journal'/client['attempt_id']
    if not journal.is_dir() or journal.is_symlink():
        raise ValueError('reference fill journal required')
    names=sorted(p.name for p in journal.iterdir())
    if names!=[f'batch-{i:08d}' for i in range(len(names))]:
        raise ValueError('accounted complete fill batches required')
    scored={};summaries=[]
    for name in names:
        directory=journal/name;manifest=read(directory/'INPUT.json')
        ordered=[None]*len(manifest['inputs'])
        for job in manifest['jobs']:
            receipt=read(directory/(job['rollout_id']+'.json'))
            if len(receipt['records'])!=len(job['indices']):
                raise ValueError('incomplete fill rollout')
            for slot,record in zip(job['indices'],receipt['records'],strict=True):
                if ordered[slot] is not None:
                    raise ValueError('duplicate fill record')
                ordered[slot]=record
        if any(record is None for record in ordered):
            raise ValueError('incomplete fill batch')
        summaries.append(audit_journal(directory,ordered,policy=c['source_policy_id'],client=client))
        for record in ordered:
            probability=probability_record(record)
            if probability['policy_id']!=c['source_policy_id']:
                raise ValueError('fill record policy differs')
            key=probability['request_id']
            if key in scored:
                raise ValueError('duplicate fill record')
            scored[key]=digest(probability)
    if len(scored)!=result['filled']:
        raise ValueError('fill journal coverage differs from the published count')
    if any(index.get(k)!=v for k,v in scored.items()):
        raise ValueError('published index differs from the audited fill journal')
    return dict(status='finite_reference_fill_audited',policy_id=c['source_policy_id'],
        base_id=c['base_id'],runtime_identity=c['dual_runtime_identity'],
        reference_cache=c['reference_cache'],reference_identity=result['reference_identity'],
        namespace=result['namespace'],entries=result['entries'],unique=result['unique'],
        filled=result['filled'],warm=result['warm'],scored=len(scored),
        index_sha256=result['index_sha256'],seconds=result['seconds'],batches=summaries,
        training_dispatch_allowed=False,
        scope='Fixed-reference precompute audit; the training run must still match every journal reference record to this index')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    print(json.dumps(fill(config,provenance),indent=2))
