"""Same ordered language diagnostic, with first attention boundaries exposed."""
import argparse
from pathlib import Path
import hashlib

from tools.reports.context_language_stage_probe import compare


def namespace():
    path=Path(__file__).with_name('context_audio_stage_probe.py');raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='d68eef5df5ace22c79e2253a1c1a2eb27a9c798ff64b76c6dbeea68f5ddc5f7b':
        raise ValueError('shared capture lifecycle changed')
    source=raw.decode()
    changes={
        'from tools.reports.context_audio_stage_capture import attach':
            'from tools.reports.context_attention_stage_capture import attach',
        'attach(base.model.audio_tower, base.model.multi_modal_projector)':
            "attach(base.model.language_model, backend='hf')",
        "'tools.reports.context_audio_stage_worker.'+worker":
            "'tools.reports.context_attention_stage_worker.'+worker",
    }
    for before,after in changes.items():
        if source.count(before)!=1:raise ValueError('capture integration anchor changed')
        source=source.replace(before,after)
    ns={'__name__':__name__+'.shared','__file__':__file__}
    exec(compile(source,'<attention-stage-probe>','exec'),ns);ns['compare']=compare
    return ns


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['hf','baseline','acoustic'])
    a=p.parse_args();ns=namespace()
    if a.phase:ns['phase'](a.package,a.output,a.phase)
    else:ns['run'](a.package,a.output)
