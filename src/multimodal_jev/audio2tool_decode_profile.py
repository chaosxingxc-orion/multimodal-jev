"""Timing of pinned native/Jev boundaries with the actual full-decode contract.

Boundary observations are not a representative task sample or a forecast for
the complete population. Outputs remain raw and are never used for prompt
selection, label scoring, or replacing a later complete inference run.
"""
import hashlib
import json
import math
from pathlib import Path
import time

from .audio2tool_inference import accept_generation,digest,write_once


def profile_case(processor,engine,sampling,case,prepare,output,finite):
    """One attempt, no retry; preparation includes original waveform validation."""
    output=Path(output);start=time.monotonic()
    if any((output/(case['name']+suffix)).exists() for suffix in ('.attempt.json','.json')):
        raise FileExistsError('boundary already attempted; no replay')
    request,receipts=prepare()
    waves=[wave for wave,rate in request['multi_modal_data']['audio']]
    if any(rate!=16000 for wave,rate in request['multi_modal_data']['audio']):
        raise ValueError('registered sample rate required')
    batch=processor(text=[request['prompt']],audio=waves,sampling_rate=16000,
                    return_tensors='pt',padding=True,truncation=False)
    ids=batch['input_ids'][0].tolist()
    frames=batch['feature_attention_mask'].sum(-1).tolist()
    prior=case['prior']
    id_hash=(hashlib.sha256(json.dumps(ids).encode()).hexdigest() if case['family']=='native' else digest(ids))
    if (len(ids)!=prior['input_tokens'] or id_hash!=prior['input_ids_sha256']
            or frames!=prior['frames'] or frames!=[r['expected_frames'] for r in receipts]
            or len(ids)+1024>32768 or not finite(batch['input_features'])):
        raise ValueError('complete real HF input must match accepted boundary')
    del batch
    attempt=dict(name=case['name'],family=case['family'],id=case['id'],decision_id=case.get('decision_id'),
        prompt_sha256=hashlib.sha256(request['prompt'].encode()).hexdigest(),audio_receipts=receipts,
        input_tokens=len(ids),input_ids_sha256=digest(ids),preparation_seconds=time.monotonic()-start)
    write_once(output/(case['name']+'.attempt.json'),attempt)
    if engine is None:
        record=dict(**attempt,model_loaded=False)
    else:
        begin=time.monotonic();response=None
        try:
            response=accept_generation(engine.generate([request],sampling,use_tqdm=False),ids)
            # Independently decode actual IDs; preserve invalid JSON and EOS.
            if processor.tokenizer.decode(response['token_ids'],skip_special_tokens=True)!=response['raw_text']:
                raise ValueError('actual output IDs and raw text differ')
        except Exception as exc:
            write_once(output/(case['name']+'.json'),dict(**attempt,model_loaded=True,
                error=repr(exc),response=response,engine_seconds=time.monotonic()-begin))
            raise
        record=dict(**attempt,model_loaded=True,engine_seconds=time.monotonic()-begin,
                    response={k:v for k,v in response.items() if k not in ('input_tokens','input_ids_sha256')})
    write_once(output/(case['name']+'.json'),record)
    return record


def summarize(records,expected_names):
    if (len(records)!=16 or len(expected_names)!=16 or len(set(expected_names))!=16
            or [r['name'] for r in records]!=list(expected_names)
            or sum(r['family']=='native' for r in records)!=4
            or sum(r['family']=='jev' for r in records)!=12):
        raise ValueError('complete ordered four native and twelve Jev boundaries required')
    result={}
    for family in ('native','jev'):
        rows=[r for r in records if r['family']==family]
        if any(r.get('error') or r['model_loaded'] is not True for r in rows):
            raise ValueError('real successful generations required for timing')
        for r in rows:
            for key in ('preparation_seconds','engine_seconds'):
                if type(r[key]) not in (int,float) or not math.isfinite(r[key]) or r[key]<0:
                    raise ValueError('finite nonnegative durations required')
            if r['engine_seconds']<=0:raise ValueError('positive engine duration required')
        result[family]=dict(cases=len(rows),input_tokens=sum(r['input_tokens'] for r in rows),
            output_tokens=sum(len(r['response']['token_ids']) for r in rows),
            preparation_seconds=sum(r['preparation_seconds'] for r in rows),
            engine_seconds=sum(r['engine_seconds'] for r in rows),
            min_engine_seconds=min(r['engine_seconds'] for r in rows),
            max_engine_seconds=max(r['engine_seconds'] for r in rows),
            length_limited=sum(r['response']['length_limited'] for r in rows))
    return dict(boundary_observations=result,population_throughput_estimate=None,
                reason='Input length/count boundaries, one ordered pass and natural EOS; not a representative throughput sample or guaranteed 1024-token stress test.')
