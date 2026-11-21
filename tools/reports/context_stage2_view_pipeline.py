"""Bind a published TRAIN view, then run the declared Stage2 windows in order."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.stage2_view_runtime import pinned
from tools.reports import context_stage2_package_components as components
from tools.reports import prepare_objective_training_candidate as objective
from tools.reports import prepare_stage2_dataset as dataset
from tools.reports import prepare_stage2_view_package as windows
from tools.reports import run_stage2_view_window as execution
from tools.reports.context_cached_acceptance_sequence import audit_config
from tools.reports.launch_context_batch_probe_candidate import wait_exact


def pin(path):
    return dict(path=str(path), sha256=file_hash(path))


def wait_for_predecessors(config, root):
    predecessors = config.get('device_predecessors', [])
    if not predecessors:
        return
    deployment = config['deployment']
    host = deployment['host']
    if host.startswith('-') or not re.fullmatch(r'[\w.@-]+', host):
        raise ValueError('explicit valid target host required')

    def remote(args, bounded=True):
        return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                               host, shlex.join(args)], capture_output=True, text=True,
                              timeout=60 if bounded else None)

    def checked(args):
        result = remote(args)
        result.check_returncode()
        return result.stdout

    events = root/'device-predecessors'
    events.mkdir()
    for predecessor in predecessors:
        cid = predecessor['container_id']
        if not re.fullmatch('[a-f0-9]{64}', cid):
            raise ValueError('exact predecessor container ID required')
        def inspect():
            actual, = json.loads(checked(['docker', 'inspect', cid]))
            devices = {d['PathOnHost'] for d in actual['HostConfig']['Devices'] or []
                       if re.fullmatch(r'/dev/davinci[0-9]+', d['PathOnHost'])}
            if (actual['Id'] != cid or actual['Name'] != predecessor['name']
                    or actual['Image'] != predecessor['image']
                    or devices != {f'/dev/davinci{i}' for i in predecessor['devices']}):
                raise ValueError('predecessor image, name or device mapping differs')
            return actual['State']
        if inspect()['Status'] not in ('running', 'exited'):
            raise ValueError('predecessor must be running or exited')
        commit_json(events/(cid+'-waiting.json'), predecessor)
        code = wait_exact(remote, cid)
        state = inspect()
        if state['Status'] != 'exited' or state['Running'] or state['ExitCode'] != code:
            raise ValueError('predecessor has not released its devices')
        commit_json(events/(cid+'-released.json'), dict(container_id=cid, state=state))
    # Docker exit wakes both controllers. Let the previous owner finish its
    # exit inspection and release the lease before readmitting the devices.
    with Path(deployment['lease']).open('r') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)


def run(config, provenance, *, before_window=None, completed_windows=None):
    root = Path(config['output'])
    root.mkdir(parents=True, exist_ok=False)
    stage = 'objective_parent'
    commit_json(root/'config.json', config)
    try:
        parent = config['objective_parent']
        package = Path(parent['output'])
        c = config['dataset']
        reuse = config.get('prepared_inputs')
        if reuse is None:
            prepared = objective.prepare(parent, provenance, builder=components.expected)
            stage = 'view_bind'
            c = dict(c, stage2_config=pin(package/'config.json'))
            dataset.run(c, provenance, 'bind')
            bound, receipt = pin(c['stage2_output']), pin(c['receipts']['bind'])
        else:
            # The window packager rechecks all parent bytes and reconstructs the
            # pinned binding before producing a fresh attempt.
            if (set(reuse)!={'parent_sha256','bound_config','dataset_receipt'}
                    or reuse['bound_config']['path']!=c['stage2_output']
                    or reuse['dataset_receipt']['path']!=c['receipts']['bind']):
                raise ValueError('explicit prepared input identities required')
            manifest=pinned(package/'PREPARED.json',reuse['parent_sha256'])
            if (manifest['status']!='objective_training_candidate_prepared_only'
                    or manifest['parent_sha256']!=parent['parent']['manifest_sha256']):
                raise ValueError('prepared objective parent differs')
            prepared=dict(manifest_sha256=reuse['parent_sha256'])
            bound,receipt=reuse['bound_config'],reuse['dataset_receipt']
            checked=pinned(receipt['path'],receipt['sha256'])
            if checked['manifest_sha256']!=c['manifest_sha256']:
                raise ValueError('prepared TRAIN view differs')
        source = dict(config['window_package'],
                      parent=dict(root=str(package), manifest_sha256=prepared['manifest_sha256']),
                      bound_config=bound, dataset_receipt=receipt)
        completed_windows = completed_windows or {}
        if tuple(completed_windows) != ('prefix', 'recovery', 'tail')[:len(completed_windows)]:
            raise ValueError('completed windows must be a contiguous finite prefix')
        replay = None
        if 'trace_replay' in config and not {'prefix', 'recovery'} <= set(completed_windows):
            from tools.reports.context_backend_trace_replay import TraceReplayEngine
            replay = TraceReplayEngine(config['trace_replay']).audit
        kwargs={'trace_auditor':replay} if replay else {}
        proofs, prefix = {}, None
        for stage in ('prefix', 'recovery', 'tail', 'formal'):
            if stage in completed_windows:
                checked = completed_windows[stage]
                proofs[stage] = checked['audit']
                if stage == 'prefix':
                    prefix = checked
                continue
            if before_window is not None:
                before_window(stage, proofs)
            loc = config['windows'][stage]
            if stage == 'prefix' and 'collected_prefix' in config:
                pin_ops=config['collected_prefix']
                ops=pinned(pin_ops['path'],pin_ops['sha256'])
                original=ops['deployment']
                package=Path(original['package'])
                prepared_prefix=pinned(package/'PREPARED.json',original['manifest_sha256'])
                existing=pinned(package/'config.json',prepared_prefix['files']['config.json'])
                if (replay is None or ops['window']!='prefix' or original['tag']!=loc['tag']
                        or original['operations_root']!=loc['operations_root']
                        or original['package']!=loc['package'] or original['remote_root']!=loc['remote_root']
                        or any(original[k]!=config['deployment'][k] for k in ('host','lease'))
                        or existing['view_lineage']['parent_sha256']!=source['parent']['manifest_sha256']
                        or existing['view_lineage']['dataset_sha256']!=c['manifest_sha256']
                        or existing['view_lineage']['objective_design_sha256']!=source['objective_design']['sha256']
                        or existing['view_lineage']['topology']!=source['topology']
                        or ops['training_package']['reference_job']!=c['reference_job']
                        or ops['objective_loss_atol']!=config['window_operations']['objective_loss_atol']):
                    raise ValueError('collected prefix differs from this release and deployment')
                ac=audit_config(dict(original,training_package=ops['training_package']),None)
                ac.update(objective_loss_atol=ops['objective_loss_atol'],prefix=None)
                (root/stage).mkdir(exist_ok=False)
                commit_json(root/stage/'audit-config.json',ac)
                proof=execution.audit(ac,provenance,**kwargs)
                commit_json(root/stage/'audit.json',proof)
                if proof['status']!='stage2_view_prefix_outputs_audited':
                    raise ValueError('independently audited collected prefix required')
                proofs[stage]=pin(root/stage/'audit.json')
                prefix=dict(audit=proofs[stage],transport=pin(ac['transport']),run=str(Path(ac['collection'])/'run'))
                continue
            local, remote = loc['package'], loc['remote_root']
            recovery = prefix if stage == 'recovery' else None
            admission = proofs if stage == 'formal' else None
            spec = dict(source, window=stage, output=local, remote_root=remote,
                        recovery=recovery, admission=admission)
            commit_json(root/(stage+'-prepare.json'), spec)
            result = windows.run(spec, provenance)
            manifest = result['manifest_sha256']
            deployment = dict(config['deployment'], package=local, remote_root=remote,
                              manifest_sha256=manifest, tag=loc['tag'], operations_root=loc['operations_root'])
            ops = dict(config['window_operations'], window=stage, deployment=deployment,
                       output=str(root/stage), admission=admission,
                       prefix=({k: prefix[k] for k in ('audit', 'run')} if recovery else None),
                       training_package=dict(output=local, manifest_sha256=manifest,
                                             reference_job=c['reference_job']))
            commit_json(root/(stage+'-window.json'), ops)
            if stage == 'prefix' and config.get('device_predecessors'):
                execution.check_image_cpu(ops, pinned(Path(local)/'PREPARED.json',manifest), Path(local))
                wait_for_predecessors(config, root)
            proof = execution.run(ops, provenance, **kwargs)
            if proof['status'] != 'stage2_view_'+stage+'_outputs_audited':
                raise ValueError('independent window audit required: '+stage)
            proofs[stage] = pin(root/stage/'audit.json')
            if stage == 'prefix':
                stem = Path(loc['operations_root'])/loc['tag']
                prefix = dict(audit=proofs[stage], transport=pin(str(stem)+'-collected.transport.json'),
                              run=str(stem)+'-collected/run')
        commit_json(root/'terminal.json', dict(exit_code=0, status='formal_stage2_view_audited',
                                               updates=len(proof['updates']), proofs=proofs))
    except BaseException as exc:
        commit_json(root/'terminal.json', dict(exit_code=1, stage=stage, error=repr(exc), restart=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    c, p = load_config(args.config, args.local)
    if args.detach:
        driver = Path(c['output']+'-driver')
        driver.mkdir(parents=True, exist_ok=False)
        argv = [sys.executable, '-B', '-m', __spec__.name, '--config', str(Path(args.config).resolve()),
                '--local', str(Path(args.local).resolve())]
        with (driver/'driver.log').open('x') as log:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        tick = Path(f'/proc/{child.pid}/stat').read_text().split(') ', 1)[1].split()[19]
        commit_json(driver/'started.json', dict(pid=child.pid, process_start_ticks=tick, argv=argv))
        print(child.pid)
    else:
        run(c, p)
