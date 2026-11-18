"""Exact single-waveform frame rounding and independent prefix-mask hashes.

The sealed v1 auditor used ceil(samples/160). The fixed Qwen processor drops
the last partial hop for single-waveform longest padding, yielding floor.
Keep v1 and its evidence intact; this version also reconstructs the full mask.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import types

from tools.reports.context_attribute_selection_processor import derive as selection_derive

VERIFY_SHA='98abacdf71cedaff5331c5729d57f73b983e97a6e2d4bec8ad4e62cb422f12e1'


def frame_geometry(samples):
    if type(samples) is not int or samples<1:
        raise ValueError('Positive complete waveform length required')
    frames=max(samples,8000)//160
    width=((frames+99)//100)*100
    return frames,width,13*(frames//100)+(frames%100+7)//8


def validate_audio_masks(records):
    for record in records:
        frames,width,tokens=frame_geometry(record['samples'])
        raw=struct.pack('<i',1)*frames+struct.pack('<i',0)*(width-frames)
        expected=dict(shape=[1,width],dtype='torch.int32',sha256=hashlib.sha256(raw).hexdigest())
        for row in record['records']:
            tensors=row['tensors']
            if (row['audio_tokens']!=tokens or tensors['input_features_mask']!=expected
                    or tensors['input_features']['shape']!=[1,128,width]
                    or tensors['input_features']['dtype']!='torch.float32'):
                raise ValueError('Exact single-waveform frames, padding or mask hash differs')
        yield record


def derive(source_root,role):
    p=Path(source_root)/'tools/reports/verify_context_attribute_population.py'
    if hashlib.sha256(p.read_bytes()).hexdigest()!=VERIFY_SHA:
        raise ValueError('Sealed independent auditor changed')
    if role=='learning':text=p.read_text()
    elif role=='selection':
        # Reuse precisely the old held-out adaptation, without changing it.
        module=selection_derive(source_root,'verify')
        # Its functions retain the original code filename; rebind inspect_records
        # below to the corrected source, retaining original verify population logic.
        text=p.read_text()
    else:raise ValueError('Known clean population required')
    old="frames=(meta['samples']+159)//160"
    if text.count(old)!=1:raise ValueError('Frame rounding anchor changed')
    text=text.replace(old,"frames=max(meta['samples'],8000)//160")
    fresh=types.ModuleType('attribute_population_v2')
    exec(compile(text,str(p)+'[single-waveform-v2]','exec'),fresh.__dict__)
    original_inspect=fresh.inspect_records
    def inspect_records(records,*args,**kwargs):
        return original_inspect(validate_audio_masks(records),*args,**kwargs)
    fresh.inspect_records=inspect_records
    if role=='selection':
        module.inspect_records=inspect_records
        return module
    return fresh


def verify(collection,package,transport,anchors,*,source_root,role):
    if role=='selection':
        from tools.reports.context_attribute_selection_processor import validate_package
        validate_package(package)
    result=derive(source_root,role).verify(collection,package,transport,anchors)
    result.update(schema='context-attribute-population-independent-v2',role=role,
        frame_rounding='floor(max(samples,8000)/160); single waveform longest padding',
        mask_verification='Every int32 prefix mask independently reconstructed and byte-hash matched',
        old_verifier_sha256=VERIFY_SHA,verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('collection','package','transport','anchors','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--source-root',default='.');p.add_argument('--role',choices=['learning','selection'],required=True)
    a=vars(p.parse_args());out=Path(a.pop('output'))
    if out.exists():raise ValueError('New independent receipt required')
    result=verify(**a)
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='histogram'},indent=2))
