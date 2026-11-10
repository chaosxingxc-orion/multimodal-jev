#!/usr/bin/env python3
"""Score a complete SFC-Jev split, then bridge typed calls to native metrics."""
import argparse
from collections import Counter
import json
import importlib.util
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_jev import restore_bundle
from multimodal_jev.sfc_jev_score import score_bundle

def evaluate_native(*args):
    spec=importlib.util.spec_from_file_location('sfc_native_evaluator',Path(__file__).with_name('evaluate_sfc.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.evaluate(*args)


def evaluate(release, split, predictions, official_scorer, output, expected_scorer_hash):
    manifest=json.loads((release/'MANIFEST.json').read_text())
    if split not in manifest['splits']:raise ValueError('unknown split')
    pin=manifest['splits'][split]
    for kind in ('requests','targets'):
        if file_hash(release/f'{split}.{kind}.jsonl')!=pin[kind+'_sha256']:raise ValueError('release changed')
    if file_hash(official_scorer)!=expected_scorer_hash:raise ValueError('official scorer changed')
    by_id={};count=0
    with predictions.open() as handle:
        for line in handle:
            row=json.loads(line);by_id.setdefault(row['example_id'],[]).append(row);count+=1
    if count!=pin['decisions'] or len(by_id)!=pin['bundles']:raise ValueError('complete split predictions required')
    output.mkdir(parents=True,exist_ok=False)
    counts=Counter();seen=set()
    try:
        with (release/f'{split}.requests.jsonl').open() as ri, (release/f'{split}.targets.jsonl').open() as ti, \
                (output/'native-manifest.jsonl').open('wb') as nm, (output/'native-predictions.jsonl').open('wb') as np, \
                (output/'decision-scores.jsonl').open('wb') as ds:
            for rline,tline in zip(ri,ti,strict=True):
                request,gold=json.loads(rline),json.loads(tline);uid=request['id']
                if uid in seen or uid not in by_id or request['split']!=split:raise ValueError('duplicate/missing/wrong-split bundle')
                seen.add(uid);result=score_bundle(request,gold,by_id[uid]);counts.update(result['counts'])
                for key in ('native_calls_valid','native_calls_exact','all_decisions_correct'):counts[key]+=result[key]
                counts['bundles']+=1
                ds.write(canonical(dict(id=uid,**result)))
                nm.write(canonical(restore_bundle(request,gold)))
                np.write(canonical(dict(example_id=uid,raw_text=result['native_prediction'])))
        if seen!=set(by_id):raise ValueError('unknown prediction bundle')
        native=evaluate_native(output/'native-manifest.jsonl',output/'native-predictions.jsonl',official_scorer,
                               output/'native-score',expected_scorer_hash)
        report=dict(release=str(release.resolve()),manifest_sha256=file_hash(release/'MANIFEST.json'),split=split,
            predictions_sha256=file_hash(predictions),counts=dict(counts),native_score=native,
            scope='All decisions scored from unmodified raw text. Typed calls serialized without repair; '
                  'Noul predictions never gate calls. Invalid call branch invalidates assembled prediction. '
                  'Official native metrics retain upstream leniencies; strict typed correctness is separate.')
        atomic_json(output/'metrics.json',report);atomic_json(output/'terminal.json',dict(exit_code=0))
        return report
    except BaseException as exc:
        atomic_json(output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('release','predictions','official-scorer','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split',choices=('train_id','test_id','test_ood'),required=True)
    p.add_argument('--expected-scorer-hash',default='fadf9ffd6a446dbe4e95b5f86b9a8998f5bba88cd515cca50db253962a51bd96')
    a=p.parse_args()
    print(json.dumps(evaluate(a.release,a.split,a.predictions,a.official_scorer,a.output,a.expected_scorer_hash),indent=2))
