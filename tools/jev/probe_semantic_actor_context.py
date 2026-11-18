#!/usr/bin/env python3
"""Measure frozen candidate requests with a real CPU processor; no model load."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

from multimodal_jev.span_contract import SpanRequest, SpanQuestion, encode_span_request, span_labels
from multimodal_jev.task_schema import DecisionSchema


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import soundfile as sf
    import torch
    from transformers import Qwen3ASRProcessor

    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    config=json.loads(args.config.read_text())
    assert not args.output.exists()
    assert not list(Path('/dev').glob('davinci[0-9]*'))
    for path,digest in config['pins'].items():
        assert sha(path)==digest,path
    templates=[json.loads(s) for s in Path(config['templates']).read_text().splitlines()]
    assert len(templates)==34
    torch.set_num_threads(4)
    processor=Qwen3ASRProcessor.from_pretrained(config['model'],local_files_only=True)
    args.output.mkdir(parents=True)
    records=[]
    with (args.output/'records.jsonl').open('x') as stream:
        for asset in config['audio']:
            assert sha(asset['audio'])==asset['sha256']
            wave,rate=sf.read(asset['audio'],dtype='float32')
            assert rate==16000 and wave.ndim==1
            for template in templates:
                d=template['request']
                request=SpanRequest(asset['audio'],tuple(SpanQuestion(q['id'],DecisionSchema.from_dict(q['schema'])) for q in d['questions']),d['final_id'],tuple(d['requirements']),d['version'])
                mappings=span_labels(processor.tokenizer,request.questions)
                inputs,positions=encode_span_request(processor,request,wave)
                ids=inputs['input_ids'][0].tolist()
                audio_id=processor.tokenizer.convert_tokens_to_ids('<|audio_pad|>')
                row=dict(audio_id=asset['id'],audio_sha256=asset['sha256'],audio_samples=len(wave),
                    panel_id=template['panel_id'],arm=template['arm'],representation=template['representation'],
                    native_layout=template['native_layout'],template_sha256=template['template_sha256'],
                    expanded_contract_sha256=template['expanded_contract_sha256'],tokens=len(ids),
                    audio_tokens=ids.count(audio_id),positions=positions.tolist(),
                    mappings=mappings,input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    overflow={str(limit):len(ids)>limit for limit in (8192,16384,32768,65536)})
                stream.write(json.dumps(row)+'\n');stream.flush();records.append(row)
    processor_file=inspect.getfile(type(processor))
    result=dict(schema='semantic-actor-context-observed-v1',requests=len(records),
        records_sha256=sha(args.output/'records.jsonl'),config_sha256=sha(args.config),
        implementation_sha256=sha(__file__),processor_source_sha256=sha(processor_file),
        versions={n:importlib.metadata.version(n) for n in ('transformers','tokenizers','torch')},
        per_arm={arm:dict(requests=sum(r['arm']==arm for r in records),
            min_tokens=min(r['tokens'] for r in records if r['arm']==arm),
            max_tokens=max(r['tokens'] for r in records if r['arm']==arm),
            overflow={str(limit):sum(r['tokens']>limit for r in records if r['arm']==arm) for limit in (8192,16384,32768,65536)}) for arm in ('Direct','Q_input')},
        image=config['image'],fixed_training_image=config['fixed_training_image'],
        fixed_training_image_used=config['image']==config['fixed_training_image'],
        context_accepted=False,training_dispatch_allowed=False,
        representations=sorted({r['representation'] for r in records}),
        audio_assets=len(config['audio']),
        scope='CPU diagnostic in the identified image, current17 frozen semantic panels and Direct/Q_input with configured existing learning waveforms. Image identity requires external container inspection. Q_credit request identity checked offline. No model forward, all-population guarantee, native-layout intervention execution or training-runtime acceptance.')
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
