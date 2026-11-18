"""Audit a formal resumed segment and bind it to its independently audited prefix.

Output audit only: deployment/container/transport admission for the resumed
attempt remains a separate responsibility. No claim of equality to an otherwise
uninterrupted run, and no model efficacy claim.
"""
import inspect
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports import audit_context_macro_training as shared
from tools.reports.audit_context_macro_formal_prefix import AUDITOR_SHA256
from tools.reports.context_macro_formal_resume import resume_contract, validate_resume
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.run_context_batch_probe_candidate import verify


def audit_segment(package, collection):
    if file_hash(Path(shared.__file__)) != AUDITOR_SHA256:
        raise ValueError('frozen shared segment auditor changed')
    source = inspect.getsource(shared.audit_outputs)
    edits = [
        ('formal_plan=formal_contract(c)', 'formal_plan=resume_contract(c)'),
        ("    if start not in (0,1):raise ValueError('initial prefix or one-step replay required')",
         "    resume_contract(c)"),
        ("binding=read(package/'REPLAY.json') if start else None",
         "binding=read(package/'RECOVERY.json')"),
        ("checkpoint_steps=formal_plan['recovery_checkpoint_steps'] if formal else list(range(start+1,stop+1))",
         "checkpoint_steps=[s for s in formal_plan['recovery_checkpoint_steps'] if start<s<=stop]"),
        ('for s in range(1,635)', 'for s in range(start+1,stop+1)'),
        ("package/'resume/policy-0001'", "package/'resume'/f'policy-{start:04d}'"),
        ('completed_updates=1)', 'completed_updates=start)'),
    ]
    for before, after in edits:
        if source.count(before) != 1:
            raise ValueError('unique segment audit anchor required: ' + before)
        source = source.replace(before, after)
    begin = '    parent=None\n'
    end = "    layout=topology(c,2,['davinci4','davinci5']);contract=expected_contract(c)"
    left, middle = source.split(begin)
    _, right = middle.split(end)
    source = left + end + right
    begin = '    if start:\n        from tools.reports.compare_context_dual_throughput_pair import equal'
    end = "    return dict(status='formal_dp2_outputs_audited'"
    left, middle = source.split(begin)
    _, right = middle.split(end)
    source = left + end + right
    namespace = dict(vars(shared), resume_contract=resume_contract)
    exec(compile(source, '<formal-recovery-segment-auditor>', 'exec'), namespace)
    result = namespace['audit_outputs'](package, collection, formal=True)
    result.update(status='formal_dp2_resumed_segment_outputs_audited',
                  fresh_process_replay_verified=False,
                  scope='Resumed segment output audit; exact startup master/Adam restoration '
                        'receipts and every consumed update checked. Full provenance and '
                        'uninterrupted-run equivalence are separate.')
    return result


def bind_prefix(package):
    """Re-read source evidence, never accept the copied prefix audit on its own."""
    from tools.reports.prepare_context_macro_sparse_resume import checkpoint
    package = Path(package)
    verify(package)
    read = lambda name: json.loads((package/name).read_text())
    binding = read('RECOVERY.json'); c = read('training.json')
    parent = binding['source_operations']; step = binding['step']
    original, commit, required, evidence = checkpoint(
        dict(step=step, transport_sha256=binding['transport_sha256']), parent)
    for key, value in evidence.items():
        if binding.get(key) != value:
            raise ValueError('independently re-audited recovery evidence differs: ' + key)
    if binding['initial_behavior_policy_id'] != commit['joint_policy']['policy_id']:
        raise ValueError('recovery behavior differs from verified prefix')
    actual = {str(p.relative_to(package/'resume')): file_hash(p)
              for p in (package/'resume').rglob('*') if p.is_file()}
    if actual != evidence['recovery_files']:
        raise ValueError('resumed assets differ from verified parent checkpoint')
    if any(p.is_symlink() for p in (package/'resume').rglob('*')):
        raise ValueError('recovery symlink forbidden')
    remote = str(Path(c['source_root']).parent)
    expected = relocate(original, parent['remote_root'], remote)
    expected.update(start_update=step, resume_root=remote+'/package/resume',
                    initial_adapter=remote+f'/package/resume/adapter-{step:04d}')
    expected['macro_client']['attempt_id'] = f'formal-resume-{step:04d}'
    expected['pins'].update({remote+'/package/resume/'+name: evidence['recovery_files'][name]
                             for name in required})
    expected['pins'][remote+'/package/RECOVERY.json'] = file_hash(package/'RECOVERY.json')
    if c != expected:
        raise ValueError('resume changed frozen experiment or dependency configuration')
    for i in range(2):
        source=json.loads((Path(parent['prepared'])/f'package/service-{i}.json').read_text())
        service=relocate(source,parent['remote_root'],remote)
        service['pins']={p:h for p,h in c['pins'].items()
                         if Path(p).is_relative_to(remote+'/package')}
        service['policy'].update(snapshot=remote+f'/package/resume/policy-{step:04d}',
            adapter=c['initial_adapter'],policy_id=commit['joint_policy']['policy_id'])
        if read(f'service-{i}.json')!=service:
            raise ValueError('resume changed frozen service configuration')
    original_sources = json.loads((Path(parent['prepared'])/'package/RUNTIME-SOURCES.json').read_text())
    if (read('RUNTIME-SOURCES.json') != original_sources
            or {str(p.relative_to(package)): file_hash(p) for p in package.rglob('*.py')}
            != original_sources):
        raise ValueError('resume changed frozen runtime sources')
    validate_resume(relocate(c, c['source_root'], str(package.resolve())))
    return evidence['prefix_audit'], step


def audit_outputs(package, collection):
    prefix, step = bind_prefix(package)
    segment = audit_segment(package, collection)
    if (prefix['status'] != 'formal_dp2_prefix_outputs_audited'
            or segment['status'] != 'formal_dp2_resumed_segment_outputs_audited'
            or [u['update'] for u in prefix['updates']] != list(range(step))
            or [u['update'] for u in segment['updates']] != list(range(step, 634))):
        raise ValueError('exact gap-free nonoverlapping full634 learning history required')
    return dict(status='formal_dp2_recovered_outputs_audited',
                updates=prefix['updates']+segment['updates'],
                prefix=prefix, resumed_segment=segment, recovery_update=step,
                prefix_learning_history_verified=True, training_dispatch_allowed=False,
                resumed_runtime_transport_verified=False,
                fresh_process_replay_verified=False,
                scope='Full634 output history composed from a re-audited prefix and resumed '
                      'segment. Resumed container/transport admission, uninterrupted-run '
                      'equivalence, source and transfer efficacy remain separate.')


def report(c, provenance):
    output = Path(c['output'])
    if not output.is_absolute() or output.exists():
        raise ValueError('new absolute audit output required')
    for source in (Path(c['package']), Path(c['collection'])):
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError('audit output must not mutate sealed inputs')
    check_paths({'recovery_audit': output.parent})
    result = audit_outputs(c['package'], c['collection'])
    result['config_provenance'] = provenance
    from tools.reports.launch_context_batch_probe_candidate import write_new
    write_new(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    args = parser.parse_args()
    config, provenance = load_config(args.config, args.local)
    shared.torch.set_num_threads(1)
    result = report(config, provenance)
    print(json.dumps({k: result[k] for k in (
        'status', 'recovery_update', 'prefix_learning_history_verified',
        'resumed_runtime_transport_verified', 'training_dispatch_allowed')}, indent=2))
