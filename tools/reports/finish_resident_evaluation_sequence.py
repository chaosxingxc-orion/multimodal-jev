"""Exact predecessor wait, independent audit, finite capacity sweep.

Fixed configuration only. No model invocation, retries, queue resume or target
scoring. The ordinary four-card lease is reacquired by each bounded candidate.
"""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.finalize_context_infonce import wait_exact
from tools.reports.audit_resident_evaluation_candidate import audit
from tools.reports.run_resident_evaluation_candidate import run as run_candidate


def read(path):return json.loads(Path(path).read_text())


def run(c):
    out=Path(c['output']);out.mkdir(exist_ok=False);commit_json(out/'config.json',c)
    try:
        record=read(c['predecessor_driver']);record['process_start_ticks']=record.pop('start_ticks')
        wait_exact(record,Path(c['predecessor_run'])/'terminal.json')
        if read(Path(c['predecessor_run'])/'terminal.json')['exit_code']:raise ValueError('predecessor failed')
        proof=audit(c['predecessor_audit']);commit_json(out/'full-campaign-audit.json',proof)
        results=[]
        for candidate in c['candidates']:
            cfg,provenance=load_config(candidate['template'],candidate['local'])
            receipt=run_candidate(cfg,provenance)
            if receipt['exit_code']:raise ValueError('capacity candidate failed; retain exact evidence')
            runroot=Path(cfg['output']);exp=read(runroot/'experiment.json')
            ac=dict(run=str(runroot),prior={p['job_id']:c['capacity_prior'] for p in exp['points']},
                    output=str(runroot/'audit.json'))
            proof=audit(ac);workers=read(runroot/'collected/result.json')['workers']
            rows=[]
            for point in exp['points']:
                durations=[];forwards=[];peaks=[]
                for rank,w in enumerate(workers):
                    durations.append(sum(e['seconds'] for e in w['timing']['events'] if e['job_id']==point['job_id'] and e['stage']=='inference'))
                    telemetry=read(runroot/f'collected/worker-{rank}'/point['job_id']/'forward-telemetry.json')[0]
                    forwards.extend(map(len,telemetry['counts']['language']));peaks.append(telemetry['peak_allocated_bytes'])
                rows.append(dict(submission_queries=point['submission_queries'],recordings=exp['limit'],
                    inference_max_worker_seconds=max(durations),actual_forward_requests=forwards,
                    max_actual_forward_requests=max(forwards),peak_allocated_bytes=max(peaks),exact=True))
            result=dict(capacity=exp['capacity'],points=rows,formal_admitted=False,
                        scope='equal-population finite capacity probe; measured hooks active; no full-set extrapolation')
            commit_json(runroot/'capacity-comparison.json',result);results.append(result)
        commit_json(out/'capacity-results.json',results)
        commit_json(out/'terminal.json',dict(exit_code=0,status='full_campaign_and_finite_capacity_audited',formal_admitted=False))
        return results
    except BaseException as exc:
        commit_json(out/'terminal.json',dict(exit_code=1,error=repr(exc),automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    run(read(a.config))
