#!/usr/bin/env python3
"""Authenticate locked historical Omni inputs before comparing cohort support."""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.slurp_intent_pilot import load_pilot
from multimodal_jev.reference_population import compare
from queue_span_native_actor_npu import read


def run(omni,pilot,manifest):
    omni=Path(omni);pilot=Path(pilot);baseline_path=omni/'delivery/ps-instruct-baseline-v1/BASELINE.json'
    baseline=read(baseline_path);prefix='delivery/s-instruct-matched-comparison-v1/P'
    job_path=omni/prefix/'inputs/prepared-jobs.private.json';result_path=omni/prefix/'RESULT.json'
    evidence={str(baseline_path):file_hash(baseline_path)}
    for p in (job_path,result_path):
        h=baseline['files'][str(p.relative_to(omni))]['sha256']
        if file_hash(p)!=h:raise ValueError('locked historical baseline changed')
        evidence[str(p)]=h
    jobs=read(job_path);result=read(result_path)
    if len(jobs)!=baseline['P']['effective_samples'] or result['effective']!=len(jobs):raise ValueError('historical denominator differs')
    for job in jobs:
        path=(omni/prefix/job['wire_file']).resolve();path.relative_to((omni/prefix).resolve())
        if file_hash(path)!=job['wire_sha256']:raise ValueError('actual reference wire changed')
        evidence[str(path)]=file_hash(path)
    items,admission=load_pilot(pilot,manifest,'D')
    release=Path(read(pilot/'MANIFEST.json')['release']);task_path=release/'tasks.json';task=read(task_path)['intent']
    roster=[dict(id=i['id'],audio_sha256=i['audio_sha256'],gold=i['target']['target_values'][0]) for i in items]
    report=compare(roster,jobs,task['candidate_pairs'])
    for p in (pilot/'MANIFEST.json',task_path,Path(__file__),Path(__file__).resolve().parents[2]/'src/multimodal_jev/reference_population.py'):
        evidence[str(p.resolve())]=file_hash(p)
    return dict(report=report,learning_admission=admission,historical_model=baseline['model'],
        historical_input=baseline['P']['input'],historical_output=baseline['P']['output'],
        historical_counts=baseline['P'],files=evidence,
        limitations=['Historical result not a fresh model reproduction; no actual original container re-observation.',
            'Variable retrieved candidates and knowledge cards differ from current full60 typed Span task.',
            'Matched examples belong to current learning population, not independent evaluation.',
            'No new score, teacher feedback, training reward or evaluation admission produced.'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('omni','pilot','manifest','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('immutable reference audit exists')
    atomic_json(out,run(a.omni,a.pilot,a.manifest))
