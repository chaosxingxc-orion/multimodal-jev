#!/usr/bin/env python3
"""Exercise pinned upstream metrics; fixtures are not benchmark results."""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import sys
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_native import native_target, parse_prediction


def verify(repository):
    fetch=json.loads((repository/'FETCH.json').read_text())
    if fetch['revision']!='8eb16545762be97ace75334109d73824217311f1':
        raise ValueError('official scorer revision changed')
    for name, digest in fetch['files'].items():
        if file_hash(repository/name)!=digest:raise ValueError('upstream files changed')
    expected={'progress':'1.5','tabulate':'0.9.0','textdistance':'4.1.5','jiwer':'2.0.0',
              'numpy':'1.26.4','python-Levenshtein':'0.12.2'}
    installed={k:importlib.metadata.version(k) for k in expected}
    if installed!=expected:raise ValueError('scorer dependency versions changed')
    sys.path.insert(0,str(repository/'scripts/evaluation'))
    from metrics import ErrorMetric
    from util import release2prediction
    source={'slurp_id':1,'scenario':'email','action':'send','intent':'send',
            'tokens':[{'id':0,'surface':'Amy'},{'id':1,'surface':'Amy'}],
            'entities':[{'type':'person','span':[0]},{'type':'person','span':[1]}],
            'recordings':[{'file':'fixture.flac'}]}
    reference=release2prediction(source)['fixture.flac']
    target=native_target(source)
    assert target=={k:reference[k] for k in ('scenario','action','entities')}
    cases={
        'perfect':(target['entities'],target['entities']),
        'missing_duplicate':(target['entities'],target['entities'][:1]),
        'extra_duplicate':(target['entities'][:1],target['entities']),
        'wrong_type':([{'type':'person','filler':'amy'}],[{'type':'place','filler':'amy'}]),
        'wrong_filler':([{'type':'person','filler':'amy smith'}],[{'type':'person','filler':'amy'}]),
        'both_empty':([],[]),
        'missing_all':(target['entities'],[]),
    }
    results={}
    for name,(gold,pred) in cases.items():
        saved=copy.deepcopy((gold,pred));metrics={}
        for label,kind,distance in [('exact','span_f1','word'),('word','span_distance_f1','word'),('char','span_distance_f1','char')]:
            metric=ErrorMetric.get_instance(kind,average='micro',distance=distance)
            metric(gold,pred);metrics[label]=metric.get_metric()
        combined=ErrorMetric.get_instance('slu_f1',average='micro')
        combined(metrics['word']);combined(metrics['char']);metrics['slu']=combined.get_metric()
        assert (gold,pred)==saved
        for i in (3,4,5):
            assert abs(metrics['slu']['overall'][i]-(metrics['word']['overall'][i]+metrics['char']['overall'][i]))<1e-12
        results[name]={k:v['overall'] for k,v in metrics.items()}
    assert results['perfect']['exact'][2]==1
    assert results['missing_duplicate']['exact'][3:]==(1.,0.,1.)
    assert results['extra_duplicate']['exact'][3:]==(1.,1.,0.)
    assert results['wrong_type']['exact'][2]==0
    assert results['wrong_filler']['exact'][2]==0
    assert 0<results['wrong_filler']['slu'][2]<1
    assert results['both_empty']['exact'][3:]==(0,0,0)
    assert results['missing_all']['exact'][5]==2
    # Combined F1 is not the arithmetic mean of distance F1 values.
    assert abs(results['wrong_filler']['slu'][2]-
               (results['wrong_filler']['word'][2]+results['wrong_filler']['char'][2])/2)>1e-6
    rejected=0
    for raw in ['invalid','{"scenario":"a","action":"b","entities":null}',
                '{"scenario":"a","action":"b","entities":[],"entities":[]}']:
        try:parse_prediction(raw)
        except ValueError:rejected+=1
    assert rejected==3
    return {'passed':True,'repository':str(repository),'revision':fetch['revision'],
            'fetch_sha256':file_hash(repository/'FETCH.json'),'dependencies':installed,
            'compatibility_exception':'Upstream tabulate0.8.7 cannot import collections.Iterable on Python3.11; use tabulate0.9.0 for report formatting only. Metric source and distance dependencies unchanged.',
            'native_filler_conversion_matches_upstream':True,'malformed_composites_rejected':rejected,
            'fixtures':results,'scope':'metric and conversion fixtures only; not full ID adapter or SLURP inference'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report=verify(args.repository)
    with args.output.open('xb') as handle:handle.write(canonical(report))
    print(json.dumps(report,indent=2))
