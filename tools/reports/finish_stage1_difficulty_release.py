"""Wait for the exact audited reference job, then release the DP6 TRAIN view."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.finalize_context_infonce import wait_exact
from tools.reports import prepare_context_cached_training as training
from tools.reports.context_cached_storage_operations import reference_command


def read(path):
    return json.loads(Path(path).read_text())


def run(c, provenance):
    root = Path(c['output'])
    root.mkdir(exist_ok=False)
    try:
        wait_exact(c['reference_driver'], c['reference_terminal'])
        terminal = read(c['reference_terminal'])
        if (terminal.get('exit_code') != 0
                or terminal.get('status') != 'reference_job_and_same_endpoint_parity_audited'):
            raise ValueError('reference controller did not complete independent audit: '
                             + str(terminal.get('error', terminal.get('status', 'missing status'))))
        finalization = Path(c['reference_finalization'])
        a = read(finalization/'audit-config.json')
        job = dict(a['reference_job'], **{k:a[k] for k in
                   ('deployment','launch','inspect','transport','collection','results_audit_output')})
        prepared = Path(job['output'])
        job['expected_argv'] = reference_command(a['deployment'], read(prepared/'PREPARED.json'),
                                                 read(prepared/'config.json'))
        t = dict(c['training'], parent=job['parent'], reference_job=job)
        training.prepare(t, provenance)
        cfg = Path(t['output'])/'config.json'
        point = read(prepared/'initial/endpoint-commit.json')
        frozen = Path(c['dataset_source']['root'])
        macros = dict(AUDITED_STAGE1_REFERENCE_JOB=job,
                      STAGE1_DATASET_ENDPOINT=dict(policy_id=point['joint_policy']['policy_id'],
                          stage1_commit_sha256=file_hash(prepared/'initial/endpoint-commit.json'),
                          endpoint=point['updates']),
                      STAGE2_DATASET_OUTPUT=c['view'], STAGE2_DATASET_SHA256=None,
                      STAGE2_DATASET_RECEIPTS={'release':str(root/'dataset-audit.json')},
                      STAGE2_INPUT_CONFIG=dict(path=str(cfg),sha256=file_hash(cfg)),
                      STAGE2_BOUND_CONFIG=str(root/'bound-config.json'),
                      EXPERIMENT_STAGE='stage2_dataset_dp6', EXPERIMENT_SOURCE_SNAPSHOT=c['dataset_source'],
                      EXPERIMENT_IO_WORKERS=c['io_workers'], EXPERIMENT_DRIVER_ROOT=str(root/'dataset-driver'))
        if 'expected_stage1_layout' in c:
            macros['STAGE1_DATASET_ENDPOINT']['stage1_layout']=c['expected_stage1_layout']
        bindings = root/'dataset-local.json'
        def save(value):
            commit_json(value,dict(schema='jev-local-macros-v1',macros=macros))
        first = root/'dataset-backend-local.json'
        save(first)
        _, p = load_config(frozen/'configs/portable/operations/stage2-dataset-dp6.json', first)
        macros['EXPERIMENT_BACKEND_SHA256'] = p['resolved_semantic_sha256']
        save(bindings)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [*(str(frozen/x) for x in ('src','','tools/jev')), *c['python_dependency_paths']]))
        subprocess.run([sys.executable,'-B','-m','tools.reports.run_simple_experiment',
                        '--config','configs/portable/operations/simple-experiment.json',
                        '--local',str(bindings),'--action','run'], cwd=frozen, env=env, check=True)
        receipt = read(root/'dataset-audit.json')
        commit_json(root/'terminal.json',dict(exit_code=0, status=receipt['status'],
                    receipt_sha256=file_hash(root/'dataset-audit.json'), training_dispatched=False))
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    a = p.parse_args()
    run(*load_config(a.config, a.local))
