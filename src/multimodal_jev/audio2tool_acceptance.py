"""Independent read-only ledger and Docker-envelope checks before gold access."""
import hashlib
import json
import math
from pathlib import Path
import re

from .audio2tool_inference import digest

IMAGE='sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
HEADERS_SHA='472340a98cca0b3a74f89b22af0935f85d38c7d464caec36875aa6a2a668d8da'
TARGETS_SHA='3aee2d3bff479982a6a34e955a1b65e1a42fb44b872a062c2125907a8284ca35'


def check_container(item,plan,container_id):
    argv=plan['argv'];host=item['HostConfig'];state=item['State']
    devices={d['PathOnHost'] for d in host.get('Devices') or []}
    expected_devices={argv[i+1].split(':')[0] for i,v in enumerate(argv[:-1]) if v=='--device'}
    env={argv[i+1] for i,v in enumerate(argv[:-1]) if v=='-e'}
    binds={argv[i+1] for i,v in enumerate(argv[:-1]) if v=='-v'}
    if (not re.fullmatch('[a-f0-9]{64}',container_id) or item['Id']!=container_id
            or plan['image']!=IMAGE or item['Image']!=IMAGE or argv[:2]!=['docker','create']
            or plan['automatic_retry'] is not False or plan['physical_devices']!=[6,7]
            or {d for d in devices if d.removeprefix('/dev/davinci').isdigit()}!={'/dev/davinci6','/dev/davinci7'}
            or state['Status']!='exited' or state['ExitCode']!=0 or state['OOMKilled']
            or state['Running'] or state['Restarting'] or state['Dead'] or state['Error']
            or host['NanoCpus']!=16*10**9 or host['Memory']!=128*1024**3
            or host['ShmSize']!=8*1024**3 or host['NetworkMode']!='host' or host['Privileged']
            or devices!=expected_devices or set(host['Binds'])!=binds
            or not env.issubset(set(item['Config']['Env']))
            or item['Config']['Cmd']!=argv[argv.index(IMAGE)+1:] or '--preflight-only' in argv):
        raise ValueError('exact complete registered full-inference container required')


def audit_ledger(ledger,rows,budgets,predictions,headers,expected_prompt):
    """expected_prompt returns independently rendered prompt and expanded IDs.

    Header/receipt arithmetic is rechecked, but raw waveforms are not decoded
    again here. Actual actor waveform identity relies on authenticated runtime
    source and its successful byte verification, not receipts alone.
    """
    ledger=Path(ledger)
    names={digest(r['id'])+'.json' for r in rows}
    if (not rows or len(names)!=len(rows) or len(rows)!=len(budgets) or len(rows)!=len(predictions)
            or {p.name for p in (ledger/'attempts').iterdir()}!=names
            or {p.name for p in (ledger/'records').iterdir()}!=names):
        raise ValueError('complete ledger and prediction coverage required')
    hashes={};limited=0;seconds=0.
    for row,budget,prediction in zip(rows,budgets,predictions,strict=True):
        name=digest(row['id'])+'.json';ap=ledger/'attempts'/name;rp=ledger/'records'/name
        attempt=json.loads(ap.read_text());record=json.loads(rp.read_text())
        if (budget['id']!=row['id'] or set(attempt)!={'id','tier','request_sha256','input_sha256',
                'prompt_sha256','audio_receipts','input_tokens','input_ids_sha256'}
                or attempt['id']!=row['id'] or attempt['tier']!=row['tier']
                or attempt['request_sha256']!=digest(row) or attempt['input_sha256']!=digest(row['input'])
                or any(record.get(k)!=v for k,v in attempt.items())
                or set(record)!=set(attempt)|{'raw_text','error','token_ids','finish_reason','length_limited','seconds'}):
            raise ValueError('original input/attempt/record identity mismatch')
        prompt,ids=expected_prompt(row,budget)
        if (attempt['prompt_sha256']!=hashlib.sha256(prompt.encode()).hexdigest()
                or attempt['input_tokens']!=budget['input_tokens'] or len(ids)!=budget['input_tokens']
                or attempt['input_ids_sha256']!=digest(ids)):
            raise ValueError('independent prompt and expanded tokens differ')
        receipts=attempt['audio_receipts'];audios=row['input']['audios']
        if len(receipts)!=len(audios) or len(receipts)!=len(budget['frames']):
            raise ValueError('complete ordered waveform receipts required')
        for receipt,audio,frames in zip(receipts,audios,budget['frames'],strict=True):
            header=headers[audio['path']];source_samples=header['frames']
            target_samples=(source_samples*16000+23999)//24000;aligned=(target_samples+159)//160*160
            if (header['manifest_sha256']!=audio['sha256'] or header['sample_rate']!=24000
                    or header['channels']!=1 or header['sample_width_bytes']!=2
                    or any(receipt[k]!=audio[k] for k in ('path','sha256','bytes'))
                    or receipt['conversion']!=dict(source_rate=24000,source_channels=1,source_samples=source_samples,
                        target_rate=16000,target_samples=target_samples,
                        conversion='channel mean then scipy.signal.resample_poly when needed')
                    or receipt['actor_samples']!=aligned or receipt['right_zero_padding_samples']!=aligned-target_samples
                    or receipt['expected_frames']!=aligned//160 or frames!=aligned//160
                    or not re.fullmatch('[a-f0-9]{64}',receipt['actor_waveform_sha256'])):
                raise ValueError('original waveform identity or conversion differs')
        tokens=record['token_ids'];reason=record['finish_reason']
        if (record['error'] is not None or type(record['raw_text']) is not str
                or type(tokens) is not list or not 1<=len(tokens)<=1024
                or any(type(t) is not int or t<0 for t in tokens)
                or reason not in ('stop','length') or record['length_limited'] is not (reason=='length')
                or (reason=='length' and len(tokens)!=1024)
                or not math.isfinite(record['seconds']) or record['seconds']<0
                or prediction!={k:record[k] for k in ('id','raw_text','error')}):
            raise ValueError('terminal output or scorer projection differs')
        limited+=record['length_limited'];seconds+=record['seconds']
        hashes[row['id']]=dict(attempt=hashlib.sha256(ap.read_bytes()).hexdigest(),
                              record=hashlib.sha256(rp.read_bytes()).hexdigest())
    return dict(records=len(rows),length_limited=limited,generation_seconds=seconds,
                ledger_hashes_sha256=digest(hashes))
