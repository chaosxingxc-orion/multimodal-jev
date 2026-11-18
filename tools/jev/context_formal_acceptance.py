"""Rebind collected NPU evidence before publishing a formal runtime.

Re-executes replay and checkpoint-frequency comparisons. Successful collection
reports establish the existing sampling/tensor audits; hashes bind their exact
artifacts and launches. No new NPU workload and no training dispatch.
"""
import json
from pathlib import Path

from tools.jev.context_rl_evaluation_plan import bound, sha
from tools.jev.context_replay_audit import verify_update
from tools.jev.context_checkpoint_comparison import compare
from tools.jev.audit_context_checkpoint_lifecycle import audit as lifecycle

IMAGE = 'sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'


def verify(*, continuous, fresh_replay, checkpoint_frequency, candidate_package, candidate_manifest_sha256):
    entries = dict(continuous=continuous, fresh_replay=fresh_replay,
                   checkpoint_frequency=checkpoint_frequency)
    roots, containers, configs, bindings = {}, {}, {}, {}
    for role, entry in entries.items():
        root = Path(entry['collection']);roots[role] = root
        report_path, launch_path = Path(entry['report']), Path(entry['launch'])
        report, launch = json.loads(report_path.read_text()), json.loads(launch_path.read_text())
        state = report['state']
        if (report.get('passed') is not True or state['Status'] != 'exited'
                or state['ExitCode'] != 0 or state['OOMKilled']
                or report['actor_container'] != report['service_container']
                or report['actor_container'] != launch['container_id']
                or report['image'] != IMAGE or launch['image'] != IMAGE
                or report.get('bad_pins') != [] or report.get('bad_package') != []):
            raise ValueError('successful independently collected exact shared-container run required: ' + role)
        containers[role] = report['actor_container']
        if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):
            raise ValueError('collected artifacts must not use symlinks')
        for name in report['files']:bound(root, report, name)
        for name, field in [('config.json','config_sha256'), ('service-config.json','service_config_sha256'),
                            ('PREPARED.json','manifest_sha256')]:
            if sha(bound(root, report, name)) != launch[field]:
                raise ValueError('launch and collection differ: ' + name)
        def read(name):return json.loads(bound(root, report, name).read_text())
        terminal = read('supervision/terminal.json')
        if (terminal['passed'] is not True or terminal['actor_started'] is not True
                or terminal['forced_cleanup'] or terminal['actor_exit_code'] != 0
                or terminal['service_exit_code'] != 0 or terminal['child_exit_codes'] != [0,0]
                or read('service/terminal.json')['exit_code'] != 0):
            raise ValueError('normally completed actor/service pair required')
        c = read('config.json');configs[role] = c
        start = 1 if role == 'fresh_replay' else 0
        interval = 64 if role == 'checkpoint_frequency' else 1
        if (c['start_update'] != start or c['stop_update'] != 2 or c['checkpoint_interval'] != interval
                or c['engineering_prefix'] is not True or c['training_dispatch_allowed'] is not False
                or c['stage'] != 'task' or c['algorithm'] != 'grpo_v1'
                or c['physical_devices'] != [4,5,6,7] or c['global_batch'] != 128 or c['microbatch'] != 8):
            raise ValueError('expected matched bounded acceptance configuration required')
        result = read('run/result.json')
        if (result['passed'] is not True or result['start_update'] != start or result['stop_update'] != 2
                or result['learning_population'] != 40513 or result['schedule_updates'] != 634):
            raise ValueError('completed acceptance prefix required')
        bindings[role] = dict(collection=str(root), report=str(report_path), report_sha256=sha(report_path),
                              launch=str(launch_path), launch_sha256=sha(launch_path))
    if len(set(containers.values())) != 3:raise ValueError('three distinct actual executions required')
    candidate = Path(candidate_package)
    if candidate.is_symlink() or sha(candidate / 'PREPARED.json') != candidate_manifest_sha256:
        raise ValueError('checkpoint candidate package changed')
    manifest = json.loads((candidate / 'PREPARED.json').read_text())
    for path in candidate.rglob('*'):
        if path.is_symlink():raise ValueError('candidate symlink forbidden')
    actual = {str(p.relative_to(candidate)):sha(p) for p in candidate.rglob('*')
              if p.is_file() and p != candidate / 'PREPARED.json'}
    if actual != manifest['files']:raise ValueError('candidate package population differs')
    if sha(roots['checkpoint_frequency'] / 'PREPARED.json') != candidate_manifest_sha256:
        raise ValueError('NPU checkpoint run did not execute this candidate')
    shared = ('source_policy_id','source_contract','base_id','runtime_profile','release_sha256',
              'schedule_sha256','algorithm','behavior_bounds')
    for name in shared:
        if any(c[name] != configs['continuous'][name] for c in configs.values()):
            raise ValueError('acceptance source/algorithm differs: ' + name)
    replay_result = verify_update(roots['continuous'] / 'run', roots['fresh_replay'] / 'run')
    checkpoint_result = compare(roots['continuous'], roots['checkpoint_frequency'])
    proof = json.loads(Path(checkpoint_frequency['report']).read_text())
    lifecycle_result = lifecycle(roots['checkpoint_frequency'] / 'run',
        files={k[4:]:v for k,v in proof['files'].items() if k.startswith('run/')},
        start_update=0, stop_update=2, checkpoint_interval=64)
    return dict(status='accepted_runtime_evidence_only_no_dispatch', containers=containers,
        bindings=bindings, candidate_manifest_sha256=candidate_manifest_sha256,
        replay=replay_result, checkpoint_comparison=checkpoint_result, lifecycle=lifecycle_result,
        scope='Existing independent NPU collection rebound; fresh replay and checkpoint comparisons rerun. '
              'Full Stage1 source, formal packaging and efficacy remain separate.')
