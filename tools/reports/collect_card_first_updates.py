"""Verify the first formal card Stage1 updates from every optimizer rank."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess

from multimodal_jev.baseline import file_hash
from multimodal_jev.card_contrast import KIND, TRIPLET_KINDS, Directions
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.watch_stage1_first_updates import REMOTE


def read(path):
    return json.loads(path.read_bytes())


def summarize(rows, step, config):
    world = len(config['physical_devices'])
    n = rows[0]['real_queries']
    if len(rows) != world or n != config['global_batch']:
        raise ValueError('incomplete startup batch')
    sums = {}
    gradu = []
    rates = set()
    obj = config['instruction_objective']
    triplet = obj['kind'] in TRIPLET_KINDS
    for row in rows:
        metrics = row['metrics']
        owned = [v for v in row['optimizer_audit'].values() if v['numel'] > 0]
        if (row['update'] != step or row['real_queries'] != n or not owned
                or any(v['state'] != 'initialized' or v['step'] != step for v in owned)):
            raise ValueError('rank optimizer update missing')
        if triplet:
            if metrics != rows[0]['metrics']:
                raise ValueError('distributed triplet metrics differ across ranks')
            if not math.isfinite(metrics['actor/grad_norm']) or metrics['actor/grad_norm'] <= 0:
                raise ValueError('invalid triplet gradient norm')
            rates.add(row['learning_rate'])
            continue
        if not math.isclose(sum(metrics['jev/context_loss']), metrics['actor/pg_loss'], abs_tol=1e-6):
            raise ValueError('microbatch loss differs from rank loss')
        rates.add(row['learning_rate'])
        gradu.extend(metrics['actor/grad_norm'])
        for key in ('infonce_numerator', 'probability_margin_numerator', 'threshold_anchor_numerator', 'local_real_queries'):
            sums[key] = sums.get(key, 0.) + sum(metrics['jev/'+key])
    margin = obj.get('probability_margin')
    if triplet:
        metrics = rows[0]['metrics']
        if obj['kind'] == KIND:
            directions = Directions.from_objective(obj)
            rebuilt = sum(weight * metrics[key] for key, weight in (
                ('jev/fixed_audio_nce', directions.audio_weight), ('jev/fixed_card_nce', directions.card_weight))
                if weight)
        else:
            rebuilt = (metrics['jev/infonce'] + margin['gap_weight'] * metrics['jev/gap']
                       + margin['anchor_weight'] * metrics['jev/anchor'])
        loss = metrics['actor/pg_loss']
        if (not math.isfinite(loss) or not math.isclose(loss, rebuilt, abs_tol=1e-6)
                or len(rates) != 1 or not 0 < min(rates) < math.inf):
            raise ValueError('global triplet components or learning rate differ')
        return dict(step=step, real_queries=n, ranks=world, loss=loss, recomputed_loss=rebuilt,
                    learning_rate=rates.pop(), grad_norm=metrics['actor/grad_norm'],
                    max_rank_seconds=max(row['total_seconds'] for row in rows),
                    max_peak_allocated=max(row['peak_allocated'] for row in rows))
    objective = (sums['infonce_numerator'] + margin['gap_weight'] * sums['probability_margin_numerator']
                 + margin['anchor_weight'] * sums['threshold_anchor_numerator']) / n
    actor_loss = sum(r['metrics']['actor/pg_loss'] for r in rows) / world
    if (sums['local_real_queries'] != n or len(rates) != 1 or not min(rates) > 0
            or not gradu or not all(math.isfinite(v) and v > 0 for v in gradu)
            or not math.isclose(objective, actor_loss, abs_tol=1e-6)):
        raise ValueError('invalid optimizer evidence or global loss')
    return dict(step=step, real_queries=n, ranks=world, loss=actor_loss,
                recomputed_loss=objective, learning_rate=rates.pop(), grad_norm=max(gradu))


def collect(package, count, timeout):
    ops_config = read(package/'operations-local.json')['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    ops = Path(ops_config['operations'])
    h = read(ops/'formal-launch.json')
    entry = package/'entry'
    host = h['host']
    base = Path(h['remote_root'])
    if host != ops_config['host'] or str(base) != ops_config['remote_root']:
        raise ValueError('unexpected formal location')

    def remote(*args):
        return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', host, shlex.join(args)])

    # The launcher records the container ID before Docker starts and before
    # formal/config.json exists. Observe its first commits before reading it.
    planned_devices = read(entry/'training.json')['physical_devices']
    world = len(planned_devices)
    prefix = dict(root=str(base/'formal'), world=world, updates=count, timeout=timeout)
    observed = json.loads(remote('python3', '-c', REMOTE.replace('CONFIG', repr(prefix), 1)))
    cfg_raw = remote('cat', str(base/'formal/config.json'))
    config = json.loads(cfg_raw)
    prepare = read(ops/'prepare-collected.json')
    finite = read(ops/'finite-collected.json')
    design = read(entry/'design.json')
    expected_stop = design['stage1']['total_updates']
    if 'triplet_objective' in (window := read(entry/'formal.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']):
        from multimodal_jev.context_topology_batch_plan import derive
        batch = dict(read(entry/'training.json')['batch_config'], per_dp_replica_batch=window['effective_anchors_per_rank'],
                     microbatch=window['execution_microbatch'])
        plan = derive(batch)
        expected_stop = (plan['updates_per_epoch'] * window['stop_after_epoch']
                         if 'stop_after_epoch' in window else plan['total_updates'])
        if (config['global_batch'] != plan['global_batch'] or config['microbatch'] != finite['microbatch']
                or config['instruction_objective'] != window['triplet_objective']):
            raise ValueError('triplet objective or admitted batch differs')
    if (finite['engineering_prefix'] is not True or finite['updates'] != 2
            or config['engineering_prefix'] is not False or config['training_dispatch_allowed'] is not True
            or config['start_update'] != 0 or config['stop_update'] != expected_stop
            or config['physical_devices'] != planned_devices
            or config['source_policy_id'] != prepare['policy_id']
            or config['stage1_layout']['plan']['sha256'] != file_hash(entry/'card-plan.json')
            or config['layout_cache']['sha256'] != prepare['cache_sha256']):
        raise ValueError('formal config or finite receipt differs')
    obj, = json.loads(remote('docker', 'inspect', h['container_id']))
    mapped = {d['PathOnHost'] for d in obj['HostConfig']['Devices'] if d['PathOnHost'].removeprefix('/dev/davinci').isdigit()}
    if (mapped != {f'/dev/davinci{i}' for i in config['physical_devices']}
            or obj['Id'] != h['container_id'] or obj['Image'] != ops_config['image']
            or obj['Args'][-1] != '/jev-entry/formal.json' or obj['Name'] != '/'+ops_config['tag']+'-formal'):
        raise ValueError('formal container identity differs')
    rows = observed['records']
    if len(rows) != world * count:
        raise ValueError('incomplete formal prefix')
    result = dict(container_id=obj['Id'], image=obj['Image'], state=obj['State']['Status'],
                  config_sha256=hashlib.sha256(cfg_raw).hexdigest(), policy_id=prepare['policy_id'],
                  plan_sha256=file_hash(entry/'card-plan.json'), finite=finite, total_updates=config['stop_update'],
                  updates=[summarize(rows[i*world:(i+1)*world], i+1, config) for i in range(count)])
    commit_json(ops/'first-updates.json', result)
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('package', type=Path)
    cli.add_argument('--count', type=int, default=3)
    cli.add_argument('--timeout', type=int, default=900)
    args = cli.parse_args()
    print(json.dumps(collect(args.package, args.count, args.timeout), indent=2))
