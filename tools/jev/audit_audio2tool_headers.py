#!/usr/bin/env python3
"""Audit all indexed Audio2Tool WAV headers without decoding or opening labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
from pathlib import Path
import time
import wave

from multimodal_jev.baseline import atomic_json, file_hash


def header(root, audio):
    relative=Path(audio['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('relative release path required')
    path=root/relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('audio path escapes release')
    if path.stat().st_size!=audio['bytes']:
        raise ValueError('audio byte length differs from manifest')
    with wave.open(str(path),'rb') as wav:
        frames=wav.getnframes(); rate=wav.getframerate(); channels=wav.getnchannels()
        width=wav.getsampwidth(); compression=wav.getcomptype()
    if frames<=0 or rate<=0 or channels<=0 or compression!='NONE':
        raise ValueError('positive uncompressed PCM audio required')
    return dict(path=audio['path'],frames=frames,sample_rate=rate,channels=channels,
                sample_width_bytes=width,seconds=frames/rate,manifest_sha256=audio['sha256'])


def audit(root, inventory, output):
    rows=[json.loads(line) for line in inventory.open()]
    tasks=[a for row in rows for a in row['audios']]
    if not rows or len({a['path'] for a in tasks})!=len(tasks):
        raise ValueError('nonempty unique complete audio inventory required')
    started=time.monotonic();results={};errors=[]
    def checked(audio):
        try:return header(root,audio)
        except Exception as exc:return dict(path=audio['path'],error=repr(exc))
    with ThreadPoolExecutor(max_workers=4) as pool, (output/'headers.jsonl').open('x') as handle:
        for i,result in enumerate(pool.map(checked,tasks),1):
            handle.write(json.dumps(result)+'\n')
            if 'error' in result:errors.append(result)
            else:results[result['path']]=result
            if i%100==0 or i==len(tasks):
                handle.flush();atomic_json(output/'progress.json',dict(done=i,total=len(tasks),
                    errors=len(errors),heartbeat_unix=time.time()))
    tiers={}
    for tier in range(1,9):
        selected=[r for r in rows if r['tier']==tier]
        valid=[r for r in selected if all(a['path'] in results for a in r['audios'])]
        lengths=[sum(results[a['path']]['seconds'] for a in r['audios']) if tier==7 else
                 max(results[a['path']]['seconds'] for a in r['audios']) for r in valid]
        clips=[results[a['path']]['seconds'] for r in selected for a in r['audios'] if a['path'] in results]
        tiers[str(tier)]=dict(queries=len(selected),complete_queries=len(valid),audio_files=len(clips),
            max_clip_seconds=max(clips,default=None),max_request_audio_seconds=max(lengths,default=None),
            clips_over_30_seconds=sum(s>30 for s in clips),
            request_semantics='sum ordered turns' if tier==7 else 'one independent variant')
    result=dict(status='passed' if not errors else 'failed',inventory_sha256=file_hash(inventory),
        audio_files_expected=len(tasks),headers_valid=len(results),errors=errors,tiers=tiers,
        sample_rates=dict(Counter(r['sample_rate'] for r in results.values())),
        channels=dict(Counter(r['channels'] for r in results.values())),
        sample_width_bytes=dict(Counter(r['sample_width_bytes'] for r in results.values())),
        elapsed_seconds=time.monotonic()-started,headers_sha256=file_hash(output/'headers.jsonl'),
        runner_sha256=file_hash(__file__),scope='All indexed current WAV sizes and PCM headers; no waveform hash recomputation, decoding, labels, processor expansion, inference or quality acceptance.')
    atomic_json(output/'result.json',result)
    if errors:raise ValueError('audio header/size errors; preserve full inventory denominator')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','inventory','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--inventory-hash',required=True);a=p.parse_args()
    if file_hash(a.inventory)!=a.inventory_hash:raise ValueError('inventory changed')
    a.output.mkdir(parents=True,exist_ok=False)
    try:audit(a.root,a.inventory,a.output)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0))
