"""Compare complete GRPO/RLOO runs after re-auditing their actual artifacts.

Read-only postprocessing. Does not select a new protocol, train or use TEST.
Single-run differences are descriptive; no seed-level significance claim.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

from tools.jev.audit_context_rl_curve_v2 import from_spec as audit_curve
from tools.reports.audit_context_arithmetic_training import audit as audit_training
from tools.jev.context_rl_evaluation_plan import sha
from tools.jev.prepare_context_formal_package import verify as verify_package


def read(path):
    return json.loads(Path(path).read_text())


def normalized(value, remote):
    if isinstance(value, str):
        return value.replace(remote, '<RUN>')
    if isinstance(value, list):
        return [normalized(v, remote) for v in value]
    if isinstance(value, dict):
        return {normalized(k, remote):normalized(v, remote) for k,v in value.items()}
    return value


def load_run(directory, expected_algorithm):
    root = Path(directory)
    if read(root/'status.json')['phase'] != 'complete_training_and_selection_curve':
        raise ValueError('complete training and 11-point curve required')
    spec = read(root/'input.json');package = Path(spec['package'])
    manifest = verify_package(package, spec['manifest_sha256'])
    if manifest['algorithm'] != expected_algorithm:
        raise ValueError('expected algorithm package required')
    audit_spec = read(root/'training-audit-spec.json')
    args = dict(audit_spec)
    for name in ('transport','launch','experiment'):
        args[name] = read(args[name])
    training = audit_training(**args)
    if training != read(root/'training-audit.json'):
        raise ValueError('persisted training audit differs from actual artifacts')
    curve_spec = read(root/'curve-spec.json')
    if (Path(curve_spec['rl_root']).resolve() != Path(args['collection']).resolve()
            or Path(curve_spec['rl_proof']).resolve() != (root/'training-audit.json').resolve()
            or read(curve_spec['experiment']) != args['experiment']
            or curve_spec['algorithm'] != expected_algorithm):
        raise ValueError('curve and training sources differ')
    curve = audit_curve(root/'curve-spec.json')
    if curve != read(root/'curve.json') or not curve['complete']:
        raise ValueError('persisted complete curve differs from actual artifacts')
    c = read(Path(args['collection'])/'config.json')
    if (c != read(package/'config.json') or training['algorithm'] != expected_algorithm
            or training['source_policy_id'] != curve['initial_source_policy_id']
            or training['container_id'] != args['launch']['container_id']):
        raise ValueError('published configuration/training/curve binding differs')
    first = curve['independently_audited_points'][0]
    predictions_path = Path(first['collection'])/'evaluation/selection/predictions.json'
    if sha(predictions_path) != first['predictions_sha256']:
        raise ValueError('step0 predictions changed')
    predictions = read(predictions_path)['records']
    if len({p['id'] for p in predictions}) != len(predictions):
        raise ValueError('unique step0 records required')
    comparable = {k:v for k,v in c.items()
                  if k not in ('algorithm','pins','formal_release','formal_release_sha256')}
    release = read(package/'RELEASE.json')
    source_files = {k:v for k,v in manifest['files'].items() if k.startswith(('source/','initial/'))}
    return dict(curve=curve, training=training, experiment=args['experiment'],
        config=normalized(comparable, manifest['remote_root']), source_files=source_files,
        source_manifest=release['source_package_manifest_sha256'],
        candidate_manifest=release['candidate_manifest_sha256'],
        step0={p['id']:p['prob'] for p in predictions},
        provenance=dict(directory=str(root.resolve()), package_manifest_sha256=spec['manifest_sha256'],
            curve_spec_sha256=sha(root/'curve-spec.json'), curve_sha256=sha(root/'curve.json'),
            training_audit_sha256=sha(root/'training-audit.json')))


def cost(training):
    rows = training['updates']
    if len(rows) != 634 or [r['update'] for r in rows] != list(range(634)):
        raise ValueError('complete 634-update costs required')
    for row in rows:
        if (type(row['real_queries']) is not int or row['real_queries'] <= 0
                or type(row['zero_signal_groups']) is not int
                or not 0 <= row['zero_signal_groups'] <= row['real_queries']):
            raise ValueError('valid real population/zero-signal counts required')
        for key in ('slowest_rank_step_seconds','mean_reward','mean_group_reward_std','mean_abs_advantage'):
            value = row[key]
            if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                raise ValueError('finite nonnegative cost/statistic required')
    queries = sum(r['real_queries'] for r in rows)
    if queries != 2*40513:
        raise ValueError('two complete learning epochs required')
    state = training['state']
    start = datetime.fromisoformat(state['StartedAt'].replace('Z','+00:00'))
    end = datetime.fromisoformat(state['FinishedAt'].replace('Z','+00:00'))
    wall = (end-start).total_seconds()
    seconds = sum(r['slowest_rank_step_seconds'] for r in rows)
    if wall <= 0 or seconds <= 0:
        raise ValueError('positive actual execution time required')
    weighted = lambda key:sum(r[key]*r['real_queries'] for r in rows)/queries
    return dict(query_exposures=queries, sampled_actions=queries*16,
        container_wall_seconds=wall, allocated_npu_hours=wall*4/3600,
        summed_slowest_step_seconds=seconds, queries_per_step_second=queries/seconds,
        zero_signal_fraction=sum(r['zero_signal_groups'] for r in rows)/queries,
        mean_reward=weighted('mean_reward'), mean_group_reward_std=weighted('mean_group_reward_std'),
        mean_abs_advantage=weighted('mean_abs_advantage'),
        scope='Four allocated physical cards counted once, including shared service; evaluation cost excluded. '
              'Summed slowest-rank step time is not container wall time.')


def compare(grpo, rloo):
    a,b = grpo,rloo
    for key in ('experiment','config','source_files','source_manifest','candidate_manifest','step0'):
        if a[key] != b[key]:
            raise ValueError('unmatched comparison: ' + key)
    x,y = a['curve'],b['curve']
    if (x['algorithm'] != 'grpo_v1' or y['algorithm'] != 'rloo_v1'
            or not x['complete'] or not y['complete']
            or x['initial_source_policy_id'] != y['initial_source_policy_id']
            or x['run_id'] == y['run_id']
            or a['training']['container_id'] == b['training']['container_id']):
        raise ValueError('distinct complete matched algorithm runs required')
    planned = a['experiment']['evaluation']['checkpoints']
    if any(c['observed_steps'] != planned for c in (x,y)):
        raise ValueError('identical complete evaluation steps required')
    if x['curve'][0]['metrics'] != y['curve'][0]['metrics']:
        raise ValueError('same source step0 metrics required')
    fields = [a['experiment']['evaluation']['primary'], *a['experiment']['evaluation']['secondary']]
    points=[]
    for left,right in zip(x['curve'],y['curve'],strict=True):
        if left['step'] != right['step']:
            raise ValueError('matched steps required')
        delta={k:None if left['metrics'][k] is None or right['metrics'][k] is None
               else left['metrics'][k]-right['metrics'][k] for k in fields}
        points.append(dict(step=left['step'],grpo=left['metrics'],rloo=right['metrics'],
            grpo_minus_rloo=delta, grpo_delta_from_step0=left['delta_from_step0'],
            rloo_delta_from_step0=right['delta_from_step0']))
    arms={}
    for arm in (a,b):
        curve=arm['curve'];primary=a['experiment']['evaluation']['primary']
        selected=next(p for p in curve['curve'] if p['step']==curve['selected_checkpoint']['step'])
        costs=cost(arm['training'])
        arms[curve['algorithm']]=dict(run_id=curve['run_id'], selected_checkpoint=curve['selected_checkpoint'],
            selected_source_checkpoint=curve['selected_source_checkpoint'],
            selected_primary_delta=selected['delta_from_step0'][primary],
            final_primary_delta=curve['last_primary_delta'],
            adjacent_decreases=curve['primary_adjacent_decreases'], costs=costs,
            final_primary_delta_per_allocated_npu_hour=curve['last_primary_delta']/costs['allocated_npu_hours'])
    return dict(status='matched_complete_selection_comparison',
        initial_source_policy_id=x['initial_source_policy_id'], arms=arms, points=points,
        provenance=dict(grpo=a['provenance'],rloo=b['provenance']),
        scope='Descriptive single-run GRPO versus RLOO-clipped comparison, including decaying supervision. '
              'Selected checkpoints use selection and are not unbiased TEST estimates. '
              'No repeated-seed significance, pure-RL causal effect, transfer or monotonic-improvement claim.')


def run(grpo_directory, rloo_directory):
    return compare(load_run(grpo_directory,'grpo_v1'),load_run(rloo_directory,'rloo_v1'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--grpo',required=True);parser.add_argument('--rloo',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=run(args.grpo,args.rloo)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
