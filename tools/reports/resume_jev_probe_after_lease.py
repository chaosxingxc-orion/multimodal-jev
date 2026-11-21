"""Resume only an unlaunched HF phase after a verified lease conflict; reuse all verified inputs."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.native_seal_operations import run
from tools.reports.score_jev_boundary_panel import score


def resume(c, *, first_attempt=False):
    root=Path(c['output']);prior=root/'hf/run';out=root/'hf-queued'
    read=lambda p:json.loads(p.read_bytes())
    fresh_or_conflict=(not prior.exists() if first_attempt else
                       prior.is_dir() and set(x.name for x in prior.iterdir())=={'terminal.json'}
                       and read(prior/'terminal.json')['error'].startswith('BlockingIOError(11,'))
    if (not fresh_or_conflict or
            read(root/'prep/run/terminal.json')!={'exit_code':0,'status':'declared_stages_finished'}):
        raise ValueError('completed prep and lease conflict before any HF launch required')
    receipt=read(root/'receipt.json')
    if file_hash(Path(c['seal'])/'MANIFEST.json')!=receipt['seal_sha256']:raise ValueError('seal changed')
    manifest=read(Path(c['seal'])/'MANIFEST.json')
    for arm,dataset in manifest['datasets'].items():
        v=read(root/'prep/run'/f'verify-{arm}-collected.json')
        if v['recordings']!=dataset['recordings'] or v['status']!='full_stable_prefix_cache_verified':
            raise ValueError('complete verified input required')
    for key,filename in [('local_sha256','local.json'),('archive_sha256','entry.tar.gz')]:
        if file_hash(root/'hf'/filename)!=receipt['packages']['hf'][key]:raise ValueError('HF payload changed')
    ops=read(root/'hf/local.json')['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    out.mkdir(exist_ok=False)
    try:
        run(dict(ops,operations=str(out/'run'),wait_for_lease=True))
        result=score(c,operations=out/'run')
        commit_json(out/'paired-score.json',result)
        commit_json(out/'terminal.json',dict(exit_code=0,score_sha256=file_hash(out/'paired-score.json')))
    except BaseException as exc:
        commit_json(out/'terminal.json',dict(exit_code=1,error=repr(exc)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--first-attempt',action='store_true');a=p.parse_args();resume(load_config(a.config,a.local)[0],first_attempt=a.first_attempt)
