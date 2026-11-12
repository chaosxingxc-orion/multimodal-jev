#!/usr/bin/env python3
"""Read-only asset audit; never infer semantic alignment from signal validity."""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from multimodal_jev.baseline import atomic_json,file_hash


def inspect_audio(path,expected_hash):
    path=Path(path)
    if file_hash(path)!=expected_hash:raise ValueError('audio bytes differ from source')
    waveform,rate=sf.read(path,dtype='float32',always_2d=True)
    if rate!=16000 or waveform.shape[1]!=1 or not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError('nonempty finite mono 16kHz audio required')
    peak=float(np.abs(waveform).max());rms=float(np.sqrt(np.mean(waveform.astype(np.float64)**2)))
    if peak==0:raise ValueError('all-zero recording')
    return dict(audio_sha256=expected_hash,samples=len(waveform),sample_rate=rate,channels=1,
        seconds=len(waveform)/rate,peak=peak,rms=rms,
        clipping_fraction=float(np.mean(np.abs(waveform)>=.999)),
        waveform_sha256=__import__('hashlib').sha256(waveform.tobytes()).hexdigest())


def run(root):
    root=Path(root);report=json.loads((root/'report.json').read_text())
    if file_hash(root/'packet.json')!=report['packet_sha256']:raise ValueError('fixed packet changed')
    packet=json.loads((root/'packet.json').read_text());records=[];failures=[]
    for row in packet:
        request=row['request']
        try:
            result=inspect_audio(request['input']['audio'],request['audio_sha256'])
            records.append(dict(id=request['id'],source_group=request['source_group'],**result))
        except Exception as exc:failures.append(dict(id=request['id'],error=repr(exc)))
    hashes=Counter(r['waveform_sha256'] for r in records)
    return dict(passed=not failures,recordings=len(packet),decoded=len(records),failures=failures,
        minimum_seconds=min((r['seconds'] for r in records),default=None),
        maximum_seconds=max((r['seconds'] for r in records),default=None),
        duplicate_waveform_groups=sum(n>1 for n in hashes.values()),records=records,
        packet_sha256=file_hash(root/'packet.json'),implementation_sha256=file_hash(__file__),
        versions={name:importlib.metadata.version(name) for name in ('numpy','soundfile')},
        audio_semantics_reviewed=False,evaluation_allowed=False,optimization_allowed=False,
        scope='File identity and decoded signal checks only. No speech/reference alignment, intelligibility or label adjudication.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable audio receipt exists')
    result=run(a.root);atomic_json(a.output,result)
    if not result['passed']:raise SystemExit(1)
