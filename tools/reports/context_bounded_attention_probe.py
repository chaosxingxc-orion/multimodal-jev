"""Full model on the original32 cohort with a bounded attention kernel only."""
import argparse
from pathlib import Path

from tools.reports.context_language_stage_probe import namespace as language_namespace, compare


def namespace():
    ns=language_namespace()
    # Recompile the pinned existing lifecycle, changing only the worker binding.
    path=Path(__file__).with_name('context_audio_stage_probe.py')
    source=path.read_text()
    changes={
        'from tools.reports.context_audio_stage_capture import attach':
            'from tools.reports.context_language_stage_capture import attach',
        'attach(base.model.audio_tower, base.model.multi_modal_projector)':
            "attach(base.model.language_model, backend='hf')",
        "'tools.reports.context_audio_stage_worker.'+worker":
            "'tools.reports.context_bounded_attention_worker.'+worker",
    }
    for before,after in changes.items():
        if source.count(before)!=1:raise ValueError('shared full model lifecycle changed')
        source=source.replace(before,after)
    ns={'__name__':__name__+'.shared','__file__':__file__}
    exec(compile(source,'<bounded-attention-probe>','exec'),ns);ns['compare']=compare
    return ns


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['hf','baseline','acoustic'])
    a=p.parse_args();ns=namespace()
    if a.phase:ns['phase'](a.package,a.output,a.phase)
    else:ns['run'](a.package,a.output)
