"""Independent audit of any declared replica training layout.

DP width, service count, device numbers, batch arithmetic and the source
schedule all come from the sealed package and its resolved configuration; this
module carries no per-topology constants. It consumes only already collected
terminal bytes, never dispatches or restarts training, and makes no efficacy or
sustained-speed claim.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_logits_wire import validate_logits_record
from multimodal_jev.context_lora_checkpoint_scope import validate_trainable_scope
from multimodal_jev.context_macro_rollout import digest
from multimodal_jev.context_topology_batch_plan import derive
from multimodal_jev.local_config import load_config
from multimodal_jev.span_request_identity import tensor_identity


def read(path):
    return json.loads(Path(path).read_text())


def positive(v):
    return type(v) in (int, float) and math.isfinite(v) and v > 0


def published_policy_commit(run, step, *, formal):
    """Read the publication layout actually emitted by each sealed trainer.

    Finite runs commit policy metadata in checkpoint-*-committed.json, with
    rank_shards when optimizer state is saved. Formal runs publish a separate
    policy commit on every step. Never synthesize or modify collected files.
    """
    run=Path(run)
    if formal:
        return read(run/f'policy-{step:04d}-committed.json')
    commit=read(run/f'checkpoint-{step:04d}-committed.json')
    if not isinstance(commit.get('rank_shards'),dict) or not commit['rank_shards']:
        raise ValueError('finite acceptance requires committed optimizer shards')
    return {key:value for key,value in commit.items() if key!='rank_shards'}


def activation(receipt, policy, replicas):
    """Declared replica count variant of the DP2 activation check.

    tools/reports/audit_context_macro_training.py pins its own bytes as evidence
    of an already audited run, so width-general checks live here instead.
    """
    if (receipt['policy_id'] != policy or receipt['head_marker_exact'] is not True
            or receipt['native_loaded_tensors'] != 112
            or len(receipt['replica_receipts']) != replicas):
        raise ValueError('every replica must acknowledge full adapter and head activation')
    for r in receipt['replica_receipts']:
        if (r['policy_id'] != policy or r['head_marker_exact'] is not True
                or r['native_loaded_tensors'] != 112):
            raise ValueError('individual replica activation differs')


@lru_cache(maxsize=4096)
def prefix_mask_count(encoded):
    """Recover the valid length only when a binary prefix mask matches its hash."""
    identity = json.loads(encoded); shape = identity['shape']
    dtypes = {'torch.int64': torch.int64, 'torch.int32': torch.int32,
              'torch.bool': torch.bool, 'torch.float32': torch.float32}
    if len(shape) != 2 or shape[0] != 1 or not 0 < shape[1] <= 32768 or identity['dtype'] not in dtypes:
        raise ValueError('bounded binary prefix mask required')
    dtype = dtypes[identity['dtype']]; n = shape[1]
    zero = torch.zeros(1, dtype=dtype).view(torch.uint8).numpy().tobytes()
    one = torch.ones(1, dtype=dtype).view(torch.uint8).numpy().tobytes()
    raw = bytearray(one*n); width = len(one)
    for count in range(n, -1, -1):
        if hashlib.sha256(raw).hexdigest() == identity['sha256']:
            return count
        if count:
            raw[(count-1)*width:count*width] = zero
    raise ValueError('mask hash does not match any binary valid-prefix length')


def audit_journal(directory, records, *, policy, client, planner):
    """Bind recorded macro results to label-free identities and declared owners."""
    directory = Path(directory)
    manifest = read(directory/'INPUT.json'); complete = read(directory/'COMPLETE.json')
    if (directory/'FAILED.json').exists():
        raise ValueError('failed macro cannot be consumed')
    if not records or len(records) > client['max_queries']:
        raise ValueError('bounded complete macro required')
    identities = []; costs = []
    for r in records:
        validate_logits_record(r); identity = r['request_identity']
        if identity['policy_id'] != policy or r['policy_id'] != policy or digest(identity) != r['request_id']:
            raise ValueError('actual request identity or policy differs')
        pos = torch.tensor([identity['positions']], dtype=torch.long)
        identities.append(dict(identity['inputs'], positions=tensor_identity(pos)))
        costs.append(max(1, sum(prefix_mask_count(json.dumps(identity['inputs'][k], sort_keys=True))
                                for k in ('attention_mask', 'input_features_mask'))))
    jobs = planner(costs, client['rollout_queries'])
    if manifest != dict(batch_id=directory.name, policy_id=policy,
                        runtime_identity=client['runtime_identity'], inputs=identities,
                        expected_indices=list(range(len(records))), jobs=jobs):
        raise ValueError('macro input identity or whole-rollout assignment differs')
    hashes = {}; ordered = [None]*len(records)
    for job in jobs:
        r = read(directory/(job['rollout_id']+'.json'))
        if (any(r[k] != v for k, v in job.items()) or r['policy_id'] != policy
                or len(r['records']) != len(job['indices']) or not positive(r['seconds'])):
            raise ValueError('rollout owner, coverage or timer differs')
        hashes[job['rollout_id']] = digest(r)
        for slot, record in zip(job['indices'], r['records'], strict=True):
            if ordered[slot] is not None or record != records[slot]:
                raise ValueError('rollout membership differs')
            ordered[slot] = record
    expected = dict(batch_id=directory.name, policy_id=policy, status='complete_macro_inference',
                    input_sha256=digest(manifest), rollout_hashes=hashes,
                    completed_indices=list(range(len(records))),
                    result_sha256=digest(ordered), actor_update_performed=False)
    if (any(complete[k] != v for k, v in expected.items()) or not positive(complete['seconds'])
            or any(x is None for x in ordered)):
        raise ValueError('complete macro receipt differs')
    return dict(batch_id=directory.name, rows=len(records), policy_id=policy,
                seconds=complete['seconds'])


def reference_transition(receipt, seen, *, current, reference, replicas):
    missing = []
    for q in receipt['queries']:
        r = q['reference_record']; key = r['request_id']
        if key in seen:
            if seen[key] != r:
                raise ValueError('fixed reference changed across repeated input')
        else:
            missing.append(r); seen[key] = r
    if (receipt['reference_misses'] != len(missing)
            or receipt['bulk_reference_calls'] != int(bool(missing))
            or receipt['bulk_behavior_calls'] != 1):
        raise ValueError('reference cache coverage differs')
    expected = [reference, current] if missing and current != reference else []
    if [r['policy_id'] for r in receipt['switches']] != expected:
        raise ValueError('reference and behavior activation sequence differs')
    for r, policy in zip(receipt['switches'], expected, strict=True):
        activation(r, policy, replicas)
    return missing


def mounted_names(manifest):
    """Container-visible physical cards: declared training plus inference."""
    devices = [*manifest['training_devices'], *[s['device'] for s in manifest['inference_services']]]
    return [f'davinci{d}' for d in sorted(devices)]


def expected_contract(c, mounted):
    from tools.reports.audit_context_dual_training import expected_contract as dual_contract
    from tools.reports.context_replica_trainer_candidate import topology
    plan = derive(c['batch_config'])
    contract = dual_contract(c)
    contract.update(world_size=plan['dp_replicas'], macro_layout=topology(c, plan['dp_replicas'], mounted),
                    schedule_release_sha256=c['schedule_release_sha256'])
    return contract


def check_result(result, start, stop, plan, engineering=True):
    expected = dict(passed=True, start_update=start, stop_update=stop,
                    schedule_updates=plan['total_updates'],
                    learning_population=plan['configuration']['source_queries'],
                    engineering_prefix=engineering)
    if result != expected:
        raise ValueError('complete configured window result required')


def check_schedule(batches, plan):
    """The tail keeps real queries as the loss denominator; zero-weight padding
    only completes the last macro batch, and total exposure stays exact."""
    if len(batches) != plan['total_updates']:
        raise ValueError('declared update count differs from the authenticated schedule')
    tail = batches[-1]
    if (tail['real_queries'] != plan['tail_real_queries']
            or plan['global_batch'] - tail['real_queries'] != plan['tail_padding_slots']):
        raise ValueError('tail real/zero-weight denominator differs')
    epochs = plan['configuration']['epochs']
    if sum(b['real_queries'] for b in batches) != plan['configuration']['source_queries'] * epochs:
        raise ValueError('total real exposures differ')


def audit_outputs(package, collection, *, initial_policy, **locations):
    from multimodal_jev.context_policy_artifacts import read_execution_bundle, verify_native_adapter
    from multimodal_jev.context_replica_macro_rollout import plan_rollouts
    from multimodal_jev.context_replica_schedule import load_schedule, rank_indices
    from multimodal_jev.slurp_portable_release import load
    from tools.jev.audit_context_rl_training import update_targets, verify_checkpoint
    from tools.reports.context_dual_group_audit import audit as audit_groups
    from tools.reports.context_macro_trainer_candidate import experiment_window
    from tools.reports.context_master_export_audit import audit as audit_masters
    from tools.reports.context_replica_trainer_candidate import topology as replica_topology

    package, collection = Path(package), Path(collection)
    c = read(package/'config.json'); manifest = read(package/'PREPARED.json')
    plan = derive(c['batch_config']); world = plan['dp_replicas']
    start, stop = c['start_update'], c['stop_update']
    formal = None
    view = 'view_binding' in c
    if view:
        from multimodal_jev.stage2_view_runtime import indices, checkpoint_contract, consume
        from tools.reports.audit_context_cached_training import ReferenceConsumption,expected_contract as cached_contract
        from tools.reports.audit_objective_training import packaged_plugin
        from multimodal_jev.objective_plugin import ObjectivePlugin
        selected=indices(c)
        engineering=c['view_window']!='formal'
        if not engineering:
            from tools.reports.context_replica_formal_candidate import contract as formal_contract
            formal=formal_contract(c)
    elif c.get('experiment_kind', 'engineering_prefix') == 'formal_source':
        from tools.reports.context_replica_formal_candidate import contract as formal_contract
        formal = formal_contract(c)
        engineering = False
        if start != 0 or stop != plan['total_updates']:
            raise ValueError('complete declared formal source interval required')
    else:
        engineering = True
        if experiment_window(c) != 'engineering_prefix' or start != 0:
            raise ValueError('sealed initial two-update acceptance window required')
    run = collection/'run'; client = c['macro_client']; runtime = c['dual_runtime_identity']
    service_names = list(manifest['layout']['services'])
    laid_out = [binding['name'] for binding in manifest['inference_services']]
    if laid_out != service_names:
        raise ValueError('declared service order differs from the packaged layout')
    mounted = mounted_names(manifest)
    layout = replica_topology(c, world, mounted)
    if view:
        contract=cached_contract(dict(c,schedule_release_sha256=c['mirror_release_sha256']),mounted)
        from multimodal_jev.stage2_view_runtime import actor_contract
        contract.update(actor_contract(c))
        contract.update(training_dataset=checkpoint_contract(c),objective_plugin=ObjectivePlugin(packaged_plugin(package,c)).identity)
        if formal:contract['formal_experiment']=formal
    else:contract = expected_contract(c, mounted)
    check_result(read(run/'result.json'), start, stop, plan, engineering)
    report = read(collection/'supervision'/'terminal.json')
    if (report['passed'] is not True or report['actor_started'] is not True
            or report['forced_cleanup'] or report['actor_exit_code'] != 0
            or [s['name'] for s in report['services']] != service_names
            or any(s.get('exit_code') != 0 for s in report['services'])):
        raise ValueError('normal completion of every declared service and actor required')
    initial = read_execution_bundle(initial_policy, expected_policy_id=c['source_policy_id'],
        expected_contract=c['source_contract'], expected_base_id=c['base_id'])
    rows, task, _ = load(c['release'], c['release_sha256'], 'learning', purpose='optimize')
    population = {r['id']: r for r in rows}; candidates = [v['id'] for v in task['schema']['candidates']]
    if view:
        population,batches,_=consume(c,rows,task,directory=package/'view-data')
        if len(candidates)!=60:raise ValueError('complete SLURP candidate panel required')
        verifier=ReferenceConsumption(c['reference_dataset'],locations.get('reference_root',c['reference_dataset']['root']))
        if c['view_window']=='recovery':
            restored=read(package/'resume/checkpoint-0001-committed.json')
            if restored['contract']!=contract or restored['updates']!=1:raise ValueError('foreign resume checkpoint')
            initial=read_execution_bundle(package/'resume/policy-0001',expected_policy_id=restored['joint_policy']['policy_id'],
                expected_contract=c['source_contract'],expected_base_id=c['base_id'])
    else:
        batches, _ = load_schedule(package/'source/schedule', c['schedule_sha256'],
            release_sha256=c['release_sha256'], population=population, batch_config=c['batch_config'],
            parent_root=package/'source/parent-schedule',
            parent_manifest_sha256=c['parent_schedule_sha256'],
            parent_release_sha256=c['schedule_release_sha256'])
        if len(population) != plan['configuration']['source_queries'] or len(candidates) != 60:
            raise ValueError('full frozen TRAIN population and candidate panel required')
        check_schedule(batches, plan)
        if formal:
            from tools.reports.context_replica_formal_candidate import validate_schedule
            validate_schedule(batches, population.keys(), c['batch_config'])
    for name, binding in zip(service_names, manifest['inference_services'], strict=True):
        service = read(package/manifest['layout']['services'][name])
        ready = read(collection/f'service-{name}/ready.json')
        if (binding['name'] != name
                or read(collection/f'service-{name}/terminal.json')['exit_code'] != 0
                or ready['physical_devices'] != [binding['device']]
                or ready['policy_id'] != (initial['policy_id'] if view else c['source_policy_id'])
                or ready['contract'] != c['source_contract']
                or ready['max_batch_queries'] != service['policy']['max_batch_queries']
                or ready['model_forward_mode'] != runtime['profile']):
            raise ValueError('declared matching service startup/completion required: ' + name)
    replicas = len(service_names)
    journal = run/'pool-journal'/client['attempt_id']
    scheduler = lambda costs, n: plan_rollouts(costs, n, replicas=replicas)
    sequence = 0; macros = []; summaries = []
    current = c['source_policy_id']; prior = initial['tensors']; reference_seen = {}
    if view:current=initial['policy_id']
    for index in range(start, stop):
        step = index+1
        receipt = read(run/f'rollout-{index:04d}.json'); batch = read(run/f'batch-{index:04d}.json')
        if receipt['runtime_identity'] != runtime:
            raise ValueError('recorded rollout runtime differs')
        records = [q['behavior_record'] for q in receipt['queries']]
        lengths = [(r['request_identity']['inputs']['input_ids']['shape'][-1],
                    r['request_identity']['inputs']['input_features']['shape'][-1]) for r in records]
        planned = dict(batches[index],
                       rank_indices=rank_indices(batches[index], lengths, c['batch_config']))
        padded = [s for s in planned['slots'] if s['padding']]
        if view and padded and selected[index]%plan['updates_per_epoch']!=plan['updates_per_epoch']-1:
            raise ValueError('padding outside authenticated view tail')
        if padded and (engineering
                or index % plan['updates_per_epoch'] != plan['updates_per_epoch'] - 1):
            if not view:raise ValueError('only the declared epoch tail may pad zero-weight slots')
        if padded and (len(padded) != plan['tail_padding_slots']
                       or batches[index]['real_queries'] != plan['tail_real_queries']):
            raise ValueError('tail padding count differs from the authenticated schedule')
        targets = update_targets(batch, receipt, planned, population, task, update=index,
            policy_id=current, reference_id=c['source_policy_id'], global_batch=plan['global_batch'])
        if view:
            verifier.check(receipt,batch);missing=[]
        else:
            missing = reference_transition(receipt, reference_seen, current=current,
                reference=c['source_policy_id'], replicas=replicas)
        if missing:
            directory = journal/f'batch-{sequence:08d}'; entry = read(directory/'INPUT.json')
            full = [None]*len(missing)
            for job in entry['jobs']:
                for slot, record in zip(job['indices'],
                        read(directory/(job['rollout_id']+'.json'))['records'], strict=True):
                    full[slot] = record
            keys = ('request_id', 'request_identity', 'prob', 'policy_id')
            if [{k: r[k] for k in keys} for r in full] != missing:
                raise ValueError('reference macro membership differs')
            macros.append(audit_journal(directory, full, policy=c['source_policy_id'],
                                        client=client, planner=scheduler)); sequence += 1
        macros.append(audit_journal(journal/f'batch-{sequence:08d}', records, policy=current,
                                    client=client, planner=scheduler)); sequence += 1
        groups = torch.load(run/f'groups-{index:04d}.pt', map_location='cpu', weights_only=True)
        expected_order = [batch['records'][i]['request_id']
                          for rank in planned['rank_indices'] for i in rank]
        if [g['request_id'] for g in groups] != expected_order:
            raise ValueError('actual declared-width schedule grouping differs')
        summary = audit_groups(groups, receipt['queries'], targets, candidates=candidates,
            algorithm=c['algorithm'], arm=c['dual_arm'], update=index, policy_id=current,
            reference_policy_id=c['source_policy_id'],
            **({'actor_options':contract['dual_objective']} if view and 'objective_config' in c else {}))
        if summary['real_queries'] != batches[index]['real_queries']:
            raise ValueError('actual real-query denominator differs from the authenticated schedule')
        saved = engineering or step in formal['recovery_checkpoint_steps']
        policy_commit = published_policy_commit(run, step, formal=formal is not None)
        if saved:
            commit = read(run/f'checkpoint-{step:04d}-committed.json')
            shards = {f'rank-{rank}/checkpoint-{step:04d}.pt':
                      file_hash(run/f'rank-{rank}/checkpoint-{step:04d}.pt') for rank in range(world)}
            if (commit['rank_shards'] != shards
                    or {k: v for k, v in commit.items() if k != 'rank_shards'} != policy_commit):
                raise ValueError('committed checkpoint and published policy identity differ')
        else:
            if (run/f'checkpoint-{step:04d}-committed.json').exists():
                raise ValueError('undeclared checkpoint state committed')
            commit = policy_commit
        if (commit['updates'] != step or commit['contract'] != contract
                or commit['execution_file'] != f'execution-{step:04d}.pt'
                or commit['execution_sha256'] != file_hash(run/f'execution-{step:04d}.pt')
                or commit['joint_policy_directory'] != f'policy-{step:04d}'
                or commit['joint_policy_manifest_sha256'] != file_hash(run/f'policy-{step:04d}/MANIFEST.json')):
            raise ValueError('published policy identity differs')
        current = commit['joint_policy']['policy_id']
        bundle = read_execution_bundle(run/f'policy-{step:04d}', expected_policy_id=current,
            expected_contract=c['source_contract'], expected_base_id=c['base_id'])
        verify_native_adapter(run/f'adapter-{step:04d}', bundle)
        execution = torch.load(run/f'execution-{step:04d}.pt', map_location='cpu', weights_only=True)
        if (set(bundle['tensors']) != set(execution) | {'decision_token_row'}
                or len(bundle['tensors']) != len(prior)
                or any(v.dtype != torch.bfloat16 or not torch.equal(v, bundle['tensors'][k])
                       for k, v in execution.items())):
            raise ValueError('every update export must match the published BF16 policy')
        if saved:
            verify_checkpoint(run, step, contract, bundle, world_size=world)
            states = [torch.load(run/f'rank-{rank}/checkpoint-{step:04d}.pt',
                                 map_location='cpu', weights_only=True) for rank in range(world)]
            master = audit_masters(states, execution, world_size=world)
        else:
            master = None
        if not torch.equal(bundle['tensors']['decision_token_row'],
                           initial['tensors']['decision_token_row']):
            raise ValueError('frozen marker changed')
        changes = {kind: sum(not torch.equal(v, prior[k]) for k, v in bundle['tensors'].items()
                             if k.startswith(prefix))
                   for kind, prefix in (('lora', 'backbone.'), ('head', 'decision_head.'))}
        prior = bundle['tensors']
        published = read(run/f'activation-{step:04d}.json')
        if published['error'] is not None or published['policy_id'] != current:
            raise ValueError('updated policy not activated')
        activation(published['receipt'], current, replicas=replicas)
        times = []
        for rank in range(world):
            record = read(run/f'rank-{rank}/update-{step:04d}.json')
            parity = record['behavior_parity_history']
            if (record['update'] != step or record['real_queries'] != summary['real_queries']
                    or record['next_policy_id'] != current or record['rollout'] != receipt
                    or record['checkpoint'] is not saved or len(parity) != layout['accumulation']
                    or any(len(group) != world
                           or any(p != dict(passed=True, error=None) for p in group) for group in parity)
                    or not all(positive(record[k])
                               for k in ('total_seconds', 'actor_seconds', 'preparation_seconds'))):
                raise ValueError('complete per-rank parity/update/timing required')
            if saved:
                scope = validate_trainable_scope(
                    [(n, torch.nn.Parameter(v)) for n, v in states[rank]['parameters'].items()])
                if read(run/f'rank-{rank}/checkpoint-scope.json') != scope:
                    raise ValueError('trainable-only checkpoint scope differs')
            times.append(record['total_seconds'])
        summary.update(slowest_rank_step_seconds=max(times), changed_execution_tensors=changes,
                       master_export=master)
        summaries.append(summary)
    if {p.name for p in journal.glob('batch-*')} != {f'batch-{i:08d}' for i in range(sequence)}:
        raise ValueError('unaccounted macro batch')
    if any(sum(row['changed_execution_tensors'][kind] for row in summaries) == 0
           for kind in ('lora', 'head')):
        raise ValueError('LoRA and head updates must both be exercised')
    for rank in range(world):
        result = read(run/f'rank-{rank}/result.json')
        if (result['passed'] is not True or result['start_update'] != start
                or result['stop_update'] != stop or len(result['executions']) != 113
                or set(result['executions'].values()) != {'bfloat16'}):
            raise ValueError('all-rank BF16 completion required')
    if view:
        if sum(r['real_queries'] for r in summaries)!=sum(batches[i]['real_queries'] for i in range(start,stop)) or verifier.samples!=(stop-start)*plan['global_batch']:
            raise ValueError('selected view exposure differs')
    return dict(status=('formal_replica_training_audited' if formal
                        else 'finite_replica_training_audited'),
                schema='context-replica-training-audit-v1', world_size=world,
                experiment_kind=('formal_source' if formal else 'engineering_prefix'),
                formal_contract=formal, services=service_names, batch_plan=plan,
                updates=summaries, macros=macros,
                fresh_process_replay_verified=False, training_dispatch_allowed=False,
                scope='Declared-topology journal/credit/export/activation/full-reduction '
                      'audit; resume, sustained speed and efficacy remain separate')


def audit_collected(c):
    from tools.reports.context_dual_probe_operations import check_runtime
    from tools.reports.context_replica_training_operations import IMAGE, command, validate
    operations, _ = load_config(c['operations_config'], c['operations_local'])
    manifest, config, _ = validate(operations)
    root = Path(operations['operations_root']); tag = operations['tag']
    launch = read(root/(tag+'-launch.json'))
    inspected = read(root/(tag+'-inspect.json'))
    check_runtime(dict(argv=command(operations, manifest, config), image=IMAGE), launch, inspected)
    if read(root/(tag+'-wait.json')) != dict(container_id=launch['container_id'], exit_code=0):
        raise ValueError('successful terminal container required')
    collection = root/(tag+'-collected')
    transport = read(root/(tag+'-collected.transport.json'))
    if (transport['status'] != 'transport_complete_independent_audit_required'
            or transport['container_id'] != launch['container_id']
            or transport['remote_root'] != manifest['remote_root']
            or transport['image'] != IMAGE or transport['state'] != inspected['State']
            or any(transport[k] for k in ('bad_pins', 'bad_package', 'launch_binding_mismatches'))):
        raise ValueError('successful unchanged actual runtime transport required')
    if file_hash(collection.with_suffix('.tar')) != transport['archive_sha256']:
        raise ValueError('transport archive changed')
    actual = {str(p.relative_to(collection)): file_hash(p)
              for p in collection.rglob('*') if p.is_file()}
    if actual != transport['files']:
        raise ValueError('complete collected bytes changed')
    result = audit_outputs(Path(operations['package']), collection,
                           initial_policy=c['initial_policy'])
    result.update(container_id=launch['container_id'], image=IMAGE,
                  manifest_sha256=operations['manifest_sha256'],
                  files_sha256=transport['files'], training_dispatch_allowed=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--local', required=True)
    args = parser.parse_args(); c, provenance = load_config(args.config, args.local)
    torch.set_num_threads(1)
    result = audit_collected(c)
    result['config_provenance'] = provenance
    output = Path(c['output']); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'world_size', 'services',
                                             'container_id', 'training_dispatch_allowed')}))
