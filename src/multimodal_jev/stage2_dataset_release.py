"""Atomic, reproducible TRAIN difficulty releases and ordinary consumer APIs.

No model call or source rewriting. Reference producer/numerical audits remain
separate; this component binds their endpoint and produces a data-only release.
"""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from .baseline import file_hash
from .context_execution import map_io, widths
from .context_reference_dataset import digest
from .context_topology_batch_plan import derive
from .local_checkpoint import commit_json, sync_directory
from .local_runtime_storage import check_paths
from .stage2_training_view import sample, make_schedule, validate_options

SCHEMA='jev-stage2-dataset-release-v2'
FILES=('selection.json','batches.json','consumer.json')
# These original v2 files implement the same legacy threshold sampler.
# The remaining reader pins and the entire resulting payload must still match.
LEGACY_READER_HASHES={
    'stage2_dataset_release.py':'9a83648db3e5341d14e7889a2e91b24e4f988d6c57f142f4e0c586dcfed3ab3b',
    'stage2_training_view.py':'28178ba7d595995f4bb922972df434e4e4b1c5bd93cc8b889f6ac6939946194c',
}
PAIRED_V1_READER_HASHES={
    'stage2_dataset_release.py':'ff60f236183d5272a02b8b5aa3fefdb0e4c0deeaf8bd7eea9466212475920122',
    'stage2_paired_sampling.py':'c2c0e8de83d0c001733b403273650f3f9a0847104f957f4f2d0933423cc33dfe',
}


def validate_execution(c):
    if set(c)!={'io_workers','cpu_workers','min_parallel','chunk','window'}:
        raise ValueError('explicit bounded execution settings required')
    widths({k:c[k] for k in ('io_workers','cpu_workers')})
    if any(type(c[k]) is not int or c[k]<1 for k in ('min_parallel','chunk','window')) or c['window']>32:
        raise ValueError('positive chunk/threshold and bounded window required')


def reader_identity(options=None):
    root=Path(__file__).parent
    names=('stage2_dataset_release.py','stage2_training_view.py','context_reference_dataset.py',
           'context_execution.py','span_class_spread.py','context_topology_batch_plan.py')
    if options and options.get('mode')=='paired_train_v1':
        names+=('stage2_paired_sampling.py',)
    return {name:file_hash(root/name) for name in names}


def require_endpoint_layout(contract, expected):
    """Actual and consumer-declared research identities must agree in all three dimensions."""
    import re
    if (not isinstance(expected,dict) or set(expected)!={'design','plan','cache'}
            or not all(isinstance(x,str) and re.fullmatch('[0-9a-f]{64}',x) for x in expected.values())
            or contract.get('stage1_layout')!=expected):
        raise ValueError('expected design, layout plan and cache must match the Stage1 endpoint')
    return dict(expected)


def endpoint_binding(checkpoint, *, commit_sha256, reference, source_expected, sampling):
    """Require caller-pinned final checkpoint, reference producer and objective."""
    from .context_infonce import validate_objective
    c=checkpoint['contract'];plan=derive(c['batch_config'])
    validate_objective(c['objective']);validate_options(sampling)
    expected=dict(policy_id=checkpoint['joint_policy']['policy_id'],stage1_commit_sha256=commit_sha256,
        endpoint=checkpoint['updates'])
    if 'stage1_layout' in c or 'stage1_layout' in source_expected:
        # A native-only historical endpoint may have the same budget and loss.
        # The receiver must pin the intended design, assignment and cache before scoring.
        expected['stage1_layout']=require_endpoint_layout(c,source_expected.get('stage1_layout'))
    if (source_expected!=expected or checkpoint['updates']!=plan['total_updates']
            or c['stage']!='instruction' or c['world_size']!=plan['dp_replicas'] or c['world_size']!=c['batch_config']['training_devices']
            or c['batch_config']['inference_instances']!=0
            or c['release_sha256']!=reference.identity['release_sha256']
            or c['base_id']!=reference.identity['base_id']
            or c['source_contract']!=reference.identity['contract']
            or expected['policy_id']!=reference.policy_id
            or c['batch_config']['source_queries']!=len(reference.population)):
        raise ValueError('same completed full TRAIN Stage1 endpoint required')
    if c['objective']['kind']!='candidate_infonce_probability_margin_v1':
        raise ValueError('current difficulty release requires margin Stage1 endpoint')
    margin=c['objective']['probability_margin']
    if (sampling.get('mode')!='paired_train_v1' and (sampling['probability_margin']!=margin['margin']
            or sampling['decision_threshold']!=margin['decision_threshold'])):
        raise ValueError('sampling thresholds differ from Stage1 objective')
    return dict(expected,objective=copy.deepcopy(c['objective']),
        prompt_layout=c.get('prompt_layout'),reference_manifest_sha256=reference.manifest_sha256)


def _inputs(rows,task,reference,endpoint,options,batch_config,evaluation,execution):
    validate_options(options);validate_execution(execution)
    if (set(evaluation)!={'interval_updates'} or type(evaluation['interval_updates']) is not int
            or evaluation['interval_updates']<1):raise ValueError('explicit evaluation interval required')
    candidates=[x['id'] for x in task['schema']['candidates']]
    if (len(rows)!=len({r['id'] for r in rows}) or set(reference.population)!={r['id'] for r in rows}
            or reference.identity['candidate_ids']!=candidates
            or endpoint['policy_id']!=reference.policy_id
            or endpoint['reference_manifest_sha256']!=reference.manifest_sha256):
        raise ValueError('complete TRAIN/reference/task/endpoint binding required')
    margin=endpoint['objective']['probability_margin']
    if options.get('mode')!='paired_train_v1' and (options['probability_margin']!=margin['margin']
            or options['decision_threshold']!=margin['decision_threshold']):
        raise ValueError('endpoint and sampling thresholds differ')
    if options.get('mode')=='paired_train_v1':
        from .stage2_paired_sampling import validate_binding
        validate_binding(options['evidence'],rows,reference.identity['release_sha256'])
    if 'source_queries' in batch_config:raise ValueError('source_queries must be derived from selected IDs')
    plan=derive(dict(batch_config,source_queries=len(rows)))
    if (batch_config['training_devices']!=plan['dp_replicas'] or batch_config['inference_instances']<1 or batch_config['epochs']<2):
        raise ValueError('Stage2 requires data-parallel training, separate inference and at least two epochs')
    return dict(schema=SCHEMA,rows_sha256=digest(sorted(rows,key=lambda r:r['id'])),task_sha256=digest(task),
        reference_manifest_sha256=reference.manifest_sha256,reference_identity=reference.identity,
        endpoint=endpoint,options=options,batch_config=batch_config,evaluation=evaluation,
        implementation=reader_identity(options) if options.get('mode')=='paired_train_v1' else reader_identity())


def expected_files(rows,task,reference,options,batch_config,evaluation,execution,timings):
    started=time.monotonic()
    probabilities={r['id']:reference.read_sample(r['id'])['prob'] for r in rows}
    timings['reference_read_seconds']=time.monotonic()-started
    selected,diagnostics,summary=sample(rows,reference.identity['candidate_ids'],probabilities,options,
        execution=execution,timings=timings)
    batch=dict(batch_config,source_queries=len(selected));plan=derive(batch)
    started=time.monotonic();batches=make_schedule(rows,selected,batch,options['seed'])
    timings['schedule_seconds']=time.monotonic()-started
    total=plan['total_updates'];interval=evaluation['interval_updates']
    consumer=dict(component='stage2_difficulty_release_v2',stage='task',batch_config=batch,
        global_batch=plan['global_batch'],macro_queries=plan['macro_queries'],microbatch=batch['microbatch'],
        rollout_queries=batch['inference_batch_queries'],start_update=0,stop_update=total,
        evaluation_steps=[0,*range(interval,total,interval),total],real_exposures=len(selected)*batch['epochs'],
        selected_ids_sha256=digest(selected),reference_manifest_sha256=reference.manifest_sha256,
        source_policy_id=reference.policy_id,reference_policy_id=reference.policy_id,seed=options['seed'],
        release_sha256=reference.identity['release_sha256'],reference_mode='full_train_readonly',
        npu_admitted=False,training_dispatch_allowed=False)
    return {'selection.json':dict(selected_ids=selected,diagnostics=diagnostics,summary=summary),
        'batches.json':batches,'consumer.json':consumer}


def _read(root,manifest_sha256,execution):
    root=Path(root)
    if root.is_symlink() or any((root/n).is_symlink() for n in ('MANIFEST.json',*FILES)):
        raise ValueError('ordinary dataset release files required')
    raw=(root/'MANIFEST.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest_sha256:raise ValueError('dataset manifest changed')
    m=json.loads(raw)
    if m['schema']!=SCHEMA or set(m['files'])!=set(FILES):raise ValueError('dataset file contract differs')
    def checked(name):
        raw=(root/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=m['files'][name]:raise ValueError('dataset files changed')
        return json.loads(raw)
    payload,stats=map_io(checked,FILES,workers=execution['io_workers'])
    if {p.name for p in root.iterdir()}!={'MANIFEST.json',*FILES}:raise ValueError('unexpected release files')
    return m,dict(zip(FILES,payload)),stats


def audit(root,manifest_sha256,*,rows,task,reference,endpoint,options,batch_config,evaluation,execution):
    return _audit_payload(_read(root,manifest_sha256,execution),manifest_sha256,rows=rows,task=task,
        reference=reference,endpoint=endpoint,options=options,batch_config=batch_config,evaluation=evaluation,execution=execution)


def _audit_payload(loaded,manifest_sha256,*,rows,task,reference,endpoint,options,batch_config,evaluation,execution):
    contract=_inputs(rows,task,reference,endpoint,options,batch_config,evaluation,execution)
    m,actual,stats=loaded;timings={'hash':stats}
    if (options.get('mode')!='paired_train_v1'
            and m['contract']['implementation']=={**contract['implementation'],**LEGACY_READER_HASHES}):
        contract['implementation']=m['contract']['implementation']
    if (options.get('evidence',{}).get('schema')=='jev-paired-train-evidence-v1'
            and m['contract']['implementation']=={**contract['implementation'],**PAIRED_V1_READER_HASHES}):
        contract['implementation']=m['contract']['implementation']
    if m['contract']!=contract or m['content_key']!=digest(contract):raise ValueError('dataset source/config/code identity differs')
    expected=expected_files(rows,task,reference,options,batch_config,evaluation,execution,timings)
    if actual!=expected:raise ValueError('dataset sampling/schedule not independently reproducible')
    return dict(status='stage2_dataset_release_audited',manifest_sha256=manifest_sha256,
        content_key=m['content_key'],recordings=len(actual['selection.json']['selected_ids']),
        consumer=actual['consumer.json'],summary=actual['selection.json']['summary'],timings=timings,
        model_calls=0,npu_executed=False,training_dispatch_allowed=False)


def publish(root,*,rows,task,reference,endpoint,options,batch_config,evaluation,execution):
    """Lock, write staging, verify, rename; identical rerun re-audits and reuses."""
    args=dict(rows=rows,task=task,reference=reference,endpoint=endpoint,options=options,
        batch_config=batch_config,evaluation=evaluation,execution=execution)
    contract=_inputs(**args);root=Path(root);check_paths({'dataset_release':root})
    root.parent.mkdir(parents=True,exist_ok=True)
    with (root.parent/(root.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if root.exists():
            proof=audit(root,file_hash(root/'MANIFEST.json'),**args)
            return dict(proof,reused=True)
        staging=root.parent/(root.name+'.building')
        # This deterministic staging directory is owned by this exact publisher.
        # Retain interrupted staging for diagnosis instead of deleting unknown data.
        if staging.exists():raise ValueError('interrupted staging exists; use a new versioned output')
        staging.mkdir();timings={}
        payload=expected_files(rows,task,reference,options,batch_config,evaluation,execution,timings)
        for n,value in payload.items():commit_json(staging/n,value)
        hashes,_=map_io(file_hash,[staging/n for n in FILES],workers=execution['io_workers'])
        commit_json(staging/'MANIFEST.json',dict(schema=SCHEMA,contract=contract,
            content_key=digest(contract),files=dict(zip(FILES,hashes))))
        proof=audit(staging,file_hash(staging/'MANIFEST.json'),**args)
        os.rename(staging,root);sync_directory(root.parent)
        return dict(proof,reused=False,build_timings=timings)


class TrainingDataset:
    """Consumer of an audited release; full reference stays shared and read-only."""
    def __init__(self,root,manifest_sha256,*,rows,task,reference,execution):
        validate_execution(execution)
        loaded=_read(root,manifest_sha256,execution);m,payload,_=loaded;c=m['contract']
        self.audit=_audit_payload(loaded,manifest_sha256,rows=rows,task=task,reference=reference,
            endpoint=c['endpoint'],options=c['options'],batch_config=c['batch_config'],
            evaluation=c['evaluation'],execution=execution)
        self.manifest_sha256=manifest_sha256;self.root=str(root)
        selected=set(payload['selection.json']['selected_ids'])
        self.population={r['id']:copy.deepcopy(r) for r in rows if r['id'] in selected}
        self.batches=payload['batches.json'];self.consumer=payload['consumer.json']
        self.reference=reference;self.task=copy.deepcopy(task)

    def bind(self,config):
        return bind_config(config,root=self.root,manifest_sha256=self.manifest_sha256,
            consumer=self.consumer,reference=self.reference)

    def stream(self,preparer,*,start_update=0,stop_update=None):
        from .cached_context_batches import CachedContextBatches
        return CachedContextBatches(self.batches,self.population,self.task,preparer,
            start_update=start_update,stop_update=len(self.batches) if stop_update is None else stop_update,
            global_batch=self.consumer['global_batch'])


def load_training_dataset(config,*,rows,task,reference,execution):
    """Explicit component seam for trainer/auditor composition; reject stale budgets."""
    from .context_component_registry import resolve
    spec=config['training_dataset']
    if set(spec)!={'component','root','manifest_sha256'} or spec['component']!='stage2_difficulty_release_v2':
        raise ValueError('registered difficulty data component required')
    cls=resolve('release',spec['component'])
    data=cls(spec['root'],spec['manifest_sha256'],rows=rows,task=task,reference=reference,execution=execution)
    if config['stage']!='task' or config['reference_dataset']['manifest_sha256']!=reference.manifest_sha256:
        raise ValueError('Stage2 reference mismatch')
    for key in ('batch_config','global_batch','microbatch','stop_update','evaluation_steps'):
        if config[key]!=data.consumer[key]:raise ValueError('dataset-derived training budget differs: '+key)
    if config['schedule_sha256']!=spec['manifest_sha256'] or config['schedule']!=spec['root']:
        raise ValueError('training schedule differs from dataset')
    if config['release_sha256']!=reference.identity['release_sha256']:
        raise ValueError('training release differs')
    return data


def bind_config(config,*,root,manifest_sha256,consumer,reference):
    """Generate a new non-dispatchable config; never mutate frozen inputs."""
    if config['stage']!='task' or config.get('start_update',0)!=0 or config.get('resume_root'):
        raise ValueError('fresh Stage2 configuration required')
    if (config['reference_dataset']['manifest_sha256']!=reference.manifest_sha256
            or config['reference_dataset']['identity']!=reference.identity
            or config['reference_dataset']['population']!=reference.population
            or config['release_sha256']!=reference.identity['release_sha256']):
        raise ValueError('Stage2 reference/release differs from mined endpoint')
    # A separate Stage2 adapter requires its own independently admitted
    # transition. The data binder must not invent that evidence.
    if config['source_policy_id']!=reference.policy_id:
        raise ValueError('different Stage2 identity needs independently admitted adapter transition; not admitted by data binding')
    result=copy.deepcopy(config)
    for k in ('batch_config','global_batch','microbatch','macro_queries','rollout_queries','start_update','stop_update','evaluation_steps'):
        result[k]=copy.deepcopy(consumer[k])
    result.update(training_dataset=dict(component='stage2_difficulty_release_v2',root=str(root),
        manifest_sha256=manifest_sha256),schedule=str(root),schedule_sha256=manifest_sha256,
        reference_policy_id=reference.policy_id,training_dispatch_allowed=False)
    for k in ('parent_schedule','parent_schedule_sha256','schedule_release_sha256','checkpoint_interval'):
        result.pop(k,None)
    if 'macro_client' in result:
        result['macro_client']['max_queries']=consumer['macro_queries']
        result['macro_client']['rollout_queries']=consumer['rollout_queries']
    result['seed']=consumer['seed']
    result['batch_plan']=derive(result['batch_config'])
    result['total_updates']=consumer['stop_update']
    return result
