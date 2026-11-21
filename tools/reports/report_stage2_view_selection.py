"""Collect audited Stage2 metrics after the exact evaluation driver exits."""
import argparse
import json
import math
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.stage2_view_runtime import pinned
from tools.reports.context_macro_probe_operations import inspect_runtime
from tools.reports.finalize_context_infonce import wait_exact
from tools.reports.seal_stage2_view_selection import DATASET, read
from tools.reports.summarize_audited_top1 import report


def collect(c, provenance):
    root = Path(c['operations'])
    output = Path(c['output'])
    result = output/'curve.json'
    output.mkdir(parents=True, exist_ok=False)
    driver = c['driver']
    try:
        wait_exact(pinned(driver['path'], driver['sha256']), root/'terminal.json')
        terminal = read(root/'terminal.json')
        if terminal != dict(exit_code=0, status='declared_stages_finished'):
            raise ValueError('evaluation controller did not complete all stages: '+repr(terminal))
        seal = c['seal']
        seal_root = Path(seal['root'])
        seal_manifest = pinned(seal_root/'MANIFEST.json', seal['sha256'])
        heldout = c.get('heldout_plan')
        dataset = DATASET
        if heldout is None:
            plan = pinned(seal_root/'source-plan.json', seal_manifest['files']['source-plan.json'])
        else:
            plan = pinned(heldout['path'], heldout['sha256'])
            dataset = plan['evaluation_dataset']
            if (plan['schema'] != 'jev-stage2-view-heldout-endpoint-v1' or dataset == DATASET
                    or plan['seal_sha256'] != seal['sha256'] or len(plan['checkpoints']) != 1
                    or plan['checkpoints'][0]['step'] != plan['fixed_terminal_step']):
                raise ValueError('frozen held-out endpoint and original dataset seal required')
        recordings = seal_manifest['datasets'][dataset]['recordings']
        decisions = seal_manifest['datasets'][dataset]['candidates']
        points = []
        for point in plan['checkpoints']:
            step = point['step']
            handles = {}
            for phase in ('hf', 'parity', 'audit-parity', 'full', 'audit-full'):
                stem = root/f'{phase}-{step}'
                launch, inspect, wait = [read(f'{stem}-{k}.json') for k in ('launch','inspect','wait')]
                inspect_runtime(inspect,launch,terminal=True)
                if wait != {'exit_code':0}:
                    raise ValueError('exact evaluation container failed: '+str(stem))
                handles[phase] = launch['container_id']
            parity_path = root/f'audit-parity-{step}-collected.json'
            audit_path = root/f'audit-full-{step}-collected.json'
            parity, audit = read(parity_path), read(audit_path)
            if (parity['exact'] is not True or parity['policy_id'] != point['policy_id']
                    or parity['dataset'] != dataset or parity['seal_sha256'] != seal['sha256']
                    or parity['queries'] != c['panel_size'] or parity['decisions'] != decisions
                    or audit['policy_id'] != point['policy_id'] or audit['step'] != step
                    or audit['dataset'] != dataset or audit['seal_sha256'] != seal['sha256']
                    or audit['recordings'] != recordings or audit['decisions'] != decisions):
                raise ValueError('evaluation population, parity or policy differs from registered point')
            mean = audit['metrics']['mean_top1_probability']
            values = report(audit,mean)
            if any(not math.isfinite(values[k]) or not 0 <= values[k] <= 1
                   for k in ('top1_mean_probability','top1_macro_f1','top1_accuracy')):
                raise ValueError('invalid Top-1 metrics')
            points.append(dict(values, container_ids=handles, parity_sha256=file_hash(parity_path),
                               audit_sha256=file_hash(audit_path)))
        status = 'stage2_view_heldout_endpoint_audited' if heldout else 'stage2_view_selection_curve_audited'
        value = dict(status=status, points=points,
                     seal_sha256=seal['sha256'], training_audit=plan['training_audit'],
                     evaluation_terminal_sha256=file_hash(root/'terminal.json'),
                     model_selection_performed=False, target_tuning_allowed=False, provenance=provenance)
        commit_json(result,value)
        commit_json(output/'terminal.json',dict(exit_code=0,curve_sha256=file_hash(result)))
        return value
    except BaseException as exc:
        commit_json(output/'terminal.json',dict(exit_code=1,error=repr(exc)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    a = p.parse_args()
    cfg, provenance = load_config(a.config, a.local)
    print(json.dumps(collect(cfg, provenance)))
