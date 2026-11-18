"""Prepare formal source GRPO from a freshly audited matched efficiency study.

No dispatch. Reuse the chosen frozen runtime and initial Stage1, adding only
the formal trainer. All machine bindings come from local macros.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.slurp_portable_release import load as load_release
from multimodal_jev.context_training_schedule import load_schedule
from tools.reports.compare_context_macro_efficiency import compare
from tools.reports.context_macro_formal_candidate import contract, STEPS, validate_schedule
from tools.reports.context_macro_formal_operations import validate
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.prepare_context_macro_replay import write
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.audit_context_macro_training import audit_collected
from tools.reports.compare_context_service_batch import compare as compare_services

FORMAL_SOURCE = 'tools/reports/context_macro_formal_candidate.py'
FORMAL_SOURCES = (FORMAL_SOURCE, 'tools/reports/context_macro_formal_resume.py')
SELECTION_RULE = 'minimum_all_100_complete_update_seconds_then_smaller_microbatch'
SERVICE_SELECTION_RULE = 'minimum_all_100_complete_update_seconds_then_smaller_service_batch'


def select_service_arm(comparison):
    if (comparison.get('schema') != 'jev-service-batch-efficiency-comparison-v1'
            or comparison.get('matched_actual_queries') is not True
            or comparison.get('matched_reference_misses') is not True
            or comparison.get('changed') != ['service batch4->32','token capacity16384->131072']
            or len(comparison.get('source_audits', [])) != 2
            or any(a.get('status') != 'finite_dp2_efficiency_audited'
                   for a in comparison['source_audits'])):
        raise ValueError('complete independently audited service capacity pair required')
    choices=[]
    for name,capacity in [('baseline',4),('candidate',32)]:
        stats=comparison[name]; total=stats['all_updates']; seconds=total['complete_seconds']
        if (stats['optimizer_steps'] != 100 or stats['warmup_updates'] != 2
                or total['updates'] != 100 or total['real_query_exposures'] != 12800
                or type(seconds) not in (float,int) or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError('complete100 fixed-micro8 service comparison required')
        choices.append((seconds,capacity,name))
    return min(choices)[2]


def efficiency_choice(c):
    """Select only from freshly re-audited finished arms, never from an OOM prefix."""
    comparison_config, provenance=load_config(c['comparison_config'],c['comparison_local'])
    kind=c.get('comparison_kind','microbatch')
    if kind=='microbatch':
        comparison=compare(comparison_config)
        selected=select_arm(comparison)
        rule=SELECTION_RULE
    elif kind=='service_batch':
        if set(comparison_config['arms'])!={'baseline','candidate'}:
            raise ValueError('explicit service baseline/candidate pair required')
        ops=[load_config(comparison_config['arms'][name]['config'],
                         comparison_config['arms'][name]['local'])[0]
             for name in ('baseline','candidate')]
        audits=[audit_collected(o) for o in ops]
        comparison=compare_services(*ops,audits)
        selected=select_service_arm(comparison)
        rule=SERVICE_SELECTION_RULE
    else:
        raise ValueError('registered microbatch or service_batch comparison required')
    spec=comparison_config['arms'][selected]
    parent,_=load_config(spec['config'],spec['local'])
    manifest=(comparison['arms'][selected]['audit_manifest_sha256'] if kind=='microbatch'
              else comparison['source_audits'][0 if selected=='baseline' else 1]['files_sha256'])
    return comparison_config,provenance,comparison,selected,parent,manifest,rule


def validate_generator(package):
    # Isolate imports from the working tree: the package executes frozen parents.
    script = """import sys
from pathlib import Path
root=Path(sys.argv[1])
sys.path[:0]=[str(root),str(root/'src'),str(root/'tools/jev')]
from tools.reports.context_macro_formal_candidate import generate
generate(root)
from tools.reports.context_macro_formal_resume import generate as resume_generate
resume_generate(root)
"""
    subprocess.run([sys.executable, '-B', '-c', script, str(package.resolve())],
                   cwd=package, check=True, timeout=60, capture_output=True, text=True)


def select_arm(comparison):
    if (comparison['schema'] != 'context-macro-efficiency-comparison-v1'
            or comparison['independent_audits'] is not True
            or comparison['matched_actual_queries'] is not True
            or comparison['matched_reference_misses'] is not True
            or comparison['warmup_updates'] != 2
            or set(comparison['arms']) != {'micro8', 'micro16', 'micro32'}):
        raise ValueError('all three matched independently audited efficiency arms required')
    choices = []
    for name, arm in comparison['arms'].items():
        micro = arm['microbatch']; stats = arm['statistics']; total = stats['all_updates']
        seconds = total['complete_seconds']
        if (type(micro) is not int or name != 'micro'+str(micro)
                or stats['optimizer_steps'] != 100 or total['updates'] != 100
                or total['real_query_exposures'] != 12800
                or type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError('complete positive-timed 100-update efficiency evidence required')
        choices.append((seconds, micro, name))
    return min(choices)[2]


def audit_schedule(training, package):
    # Runtime paths are remote; resolve sealed inputs against the local package.
    training = relocate(training, training['source_root'], str(Path(package).resolve()))
    rows, _, _ = load_release(training['release'], training['release_sha256'],
                              'learning', purpose='optimize')
    population = {r['id']: r for r in rows}
    batches, _ = load_schedule(training['schedule'], training['schedule_sha256'],
        release_sha256=training.get('schedule_release_sha256', training['release_sha256']),
        population=population)
    counts = validate_schedule(batches, population.keys())
    return dict(real_exposures=sum(counts), updates=len(counts), tail_real_queries=counts[-1],
                release_sha256=training['release_sha256'], schedule_sha256=training['schedule_sha256'])


def prepare(c, provenance):
    (comparison_config,comparison_provenance,comparison,selected,parent,
     selected_manifest,selection_rule)=efficiency_choice(c)
    source = Path(parent['prepared'])/'package'
    if verify(source) != selected_manifest:
        raise ValueError('selected runtime no longer matches audited efficiency evidence')
    output, remote, old = Path(c['output']), Path(c['remote_root']), Path(parent['remote_root'])
    if (output.exists() or not output.is_absolute()
            or output.resolve().is_relative_to(source.resolve())
            or source.resolve().is_relative_to(output.resolve())
            or not remote.is_absolute() or '..' in remote.parts or ':' in str(remote)
            or len(str(remote/'ipc').encode())+8 > 107):
        raise ValueError('new disjoint bounded formal output and runtime paths required')
    for arm in comparison_config['arms'].values():
        other, _ = load_config(arm['config'], arm['local'])
        if remote.is_relative_to(other['remote_root']) or Path(other['remote_root']).is_relative_to(remote):
            raise ValueError('formal runtime overlaps an efficiency arm')
    original = json.loads((source/'training.json').read_text())
    if (original['start_update'] != 0 or original['stop_update'] != 100
            or original['experiment_kind'] != 'efficiency' or original.get('resume_root')):
        raise ValueError('fresh Stage1 initialization from selected efficiency arm required')
    schedule = audit_schedule(original, source)
    check_paths({'formal_prepared': output})
    output.mkdir(parents=True)
    dest = output/'package'; shutil.copytree(source, dest)
    (dest/'FILES.json').unlink()
    for entry in FORMAL_SOURCES:
        if (dest/entry).exists():
            raise ValueError('efficiency runtime unexpectedly already contains formal entry')
        (dest/entry).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(c['project'])/entry, dest/entry)
    validate_generator(dest)  # Reject incompatible frozen generator before sealing.
    runtime = {str(p.relative_to(dest)): file_hash(p) for p in dest.rglob('*.py')}
    previous = json.loads((source/'RUNTIME-SOURCES.json').read_text())
    if {k:v for k,v in runtime.items() if k not in FORMAL_SOURCES} != previous:
        raise ValueError('formal preparation must preserve all prior runtime source bytes')
    write(dest/'RUNTIME-SOURCES.json', runtime)
    training = relocate(original, str(old), str(remote))
    training.update(experiment_kind='formal_source', start_update=0, stop_update=634,
        engineering_prefix=False, training_dispatch_allowed=True, checkpoint_interval=64,
        evaluation_steps=list(STEPS), seed=317)
    training['macro_client']['attempt_id'] = 'formal-source-v1'
    training['dual_runtime_identity']['source_manifest_sha256'] = file_hash(dest/'RUNTIME-SOURCES.json')
    training['macro_client']['runtime_identity'] = training['dual_runtime_identity']
    formal = contract(training)
    evidence = dict(schema='context-macro-formal-preparation-v1', selected_arm=selected,
        selection_rule=selection_rule, comparison=comparison, comparison_provenance=comparison_provenance,
        parent_manifest_sha256=verify(source), schedule=schedule, formal_contract=formal,
        initial_policy_id=training['source_policy_id'],
        source_changes=list(FORMAL_SOURCES), formal_launch_admitted=False)
    write(dest/'FORMAL.json', evidence)
    internal = {str(remote/'package'/p.relative_to(dest)): file_hash(p) for p in dest.rglob('*')
        if p.is_file() and p.name not in ('training.json', 'service-0.json', 'service-1.json')}
    training['pins'] = {p:h for p,h in training['pins'].items() if not Path(p).is_relative_to(remote/'package')}
    training['pins'].update(internal)
    write(dest/'training.json', training)
    for i in range(2):
        service = relocate(json.loads((source/f'service-{i}.json').read_text()), str(old), str(remote))
        service.update(pins=internal, dual_runtime_identity=training['dual_runtime_identity'])
        write(dest/f'service-{i}.json', service)
    write(dest/'FILES.json', {str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()})
    receipt = json.loads((source.parent/'PREPARED.json').read_text())
    receipt.pop('efficiency', None)
    receipt.update(status='macro_formal_prepared_only', files_sha256=verify(dest),
        remote_root=str(remote), ipc_root=str(remote/'ipc'), runtime_identity=training['dual_runtime_identity'],
        formal_launch_admitted=False, training_dispatch_allowed=False,
        selected_microbatch=training['microbatch'], formal_evidence_sha256=file_hash(dest/'FORMAL.json'),
        config_provenance=provenance, pending=['formal NPU tail and sparse recovery acceptance',
            'source evaluation integration and launch admission'])
    write(output/'PREPARED.json', receipt)
    operations = dict(parent, prepared=str(output), remote_root=str(remote),
        manifest_sha256=receipt['files_sha256'], operations_root=c['operations_root'], tag=c['tag'])
    operations.pop('finite_training_dispatch_allowed', None)
    validate(operations)
    write(output/'operations.json', operations)
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--local', required=True)
    args=parser.parse_args(); config, provenance=load_config(args.config, args.local)
    import torch
    torch.set_num_threads(1)
    print(json.dumps(prepare(config, provenance), indent=2))
