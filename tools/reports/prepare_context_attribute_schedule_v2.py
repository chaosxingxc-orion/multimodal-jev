"""Use the corrected full-input auditor with the unchanged schedule builder."""
import argparse
import json
from pathlib import Path
import types

from multimodal_jev.baseline import file_hash
from tools.reports import context_attribute_training_contract as original
from tools.reports.verify_context_attribute_population_v2 import verify

SOURCE_SHA='04dd13d6d5663333f82da8f2b14ee5f98ecdac4f2e12e77bc1495c8134645ff6'


def prepare(**kwargs):
    if file_hash(original.__file__)!=SOURCE_SHA:
        raise ValueError('Original candidate schedule builder changed')
    evidence={}
    def corrected(collection,package,transport,anchors):
        result=verify(collection,package,transport,anchors,source_root=kwargs['source_root'],role='learning')
        evidence.update(result)
        return result
    # Separate function globals; never mutate the imported or sealed module.
    context=dict(original.prepare_schedule.__globals__,verify=corrected)
    builder=types.FunctionType(original.prepare_schedule.__code__,context)
    result=builder(**kwargs)
    root=Path(kwargs['output'])
    (root/'processor-independent-v2.json').write_text(json.dumps(evidence,indent=2)+'\n')
    result['processor_independent_sha256']=file_hash(root/'processor-independent-v2.json')
    result['schedule_builder_sha256']=SOURCE_SHA
    result['auditor_version']='single-waveform-floor-and-prefix-mask-v2'
    (root/'MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('source-root','original-schedule','original-sha256','collection','package','transport','anchors','output'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--arm',required=True,choices=['direct','q_input','q_credit'])
    print(json.dumps(prepare(**vars(p.parse_args())),indent=2))
