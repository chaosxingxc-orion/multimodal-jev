"""Reuse the exact acoustic diagnostic cohort with read-only language hooks."""
import argparse
import hashlib
import json
from pathlib import Path

PARENT_SHA='d68eef5df5ace22c79e2253a1c1a2eb27a9c798ff64b76c6dbeea68f5ddc5f7b'


def namespace():
    path=Path(__file__).with_name('context_audio_stage_probe.py');raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PARENT_SHA:
        raise ValueError('shared read-only capture lifecycle changed')
    source=raw.decode()
    changes={
        'from tools.reports.context_audio_stage_capture import attach':
            'from tools.reports.context_language_stage_capture import attach',
        'attach(base.model.audio_tower, base.model.multi_modal_projector)':
            "attach(base.model.language_model, backend='hf')",
        "'tools.reports.context_audio_stage_worker.'+worker":
            "'tools.reports.context_language_stage_worker.'+worker",
    }
    for old,new in changes.items():
        if source.count(old)!=1:raise ValueError('unique capture binding required: '+old)
        source=source.replace(old,new)
    ns={'__name__':__name__+'.shared','__file__':__file__}
    exec(compile(source,'<language-stage-probe>','exec'),ns)
    ns['compare']=compare
    return ns


def compare(output,config):
    import torch
    from tools.reports.context_language_stage_capture import compare as compare_stages
    from tools.reports.context_acoustic_numeric_probe import difference
    output=Path(output)
    refs=[torch.load(output/'hf'/f'row-{i:02d}.pt',map_location='cpu',weights_only=True)
          for i in range(len(config['rows']))]
    lengths=[r['tensors']['input_ids']['shape'][1] for r in config['rows']]
    def read(mode,name):return torch.load(output/mode/name,map_location='cpu',weights_only=True)
    rows=[]
    for i,row in enumerate(config['rows']):
        rows.append(dict(id=row['id'],comparisons={mode:compare_stages([refs[i]],
            read(mode,f'row-{i:02d}.pt'),lengths=[lengths[i]]) for mode in ('baseline','acoustic')}))
    batch={mode:compare_stages(refs,read(mode,'batch.pt'),lengths=lengths)
           for mode in ('baseline','acoustic')}
    expected=json.loads((output/'hf/records.json').read_text());scores={}
    for mode in ('baseline','acoustic'):
        scores[mode]={}
        for part in ('records','batch-records'):
            actual=json.loads((output/mode/(part+'.json')).read_text())
            scores[mode][part]=[dict(id=row['id'],logits=difference(torch.tensor(a['logits']),torch.tensor(b['logits'])),
                prob=difference(torch.tensor(a['prob']),torch.tensor(b['prob'])))
                for row,a,b in zip(config['rows'],expected,actual,strict=True)]
    return dict(status='actual_language_stage_comparison',rows=rows,batch=batch,scores=scores,
        full_model_parity_verified=False,training_dispatch_allowed=False,
        scope='Original ordered 32-request cohort, no gold, no numerical tolerance changes; diagnostic only')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['hf','baseline','acoustic'])
    a=p.parse_args();ns=namespace()
    if a.phase:ns['phase'](a.package,a.output,a.phase)
    else:ns['run'](a.package,a.output)
