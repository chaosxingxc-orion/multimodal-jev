#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from multimodal_jev.baseline import file_hash


def run(c,out):
    import torch
    from multimodal_jev.context_policy_admission import admit_initial_policy
    from multimodal_jev.local_checkpoint import commit_json
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.policy_identity import checkpoint_identity
    torch.set_num_threads(1);require_training_storage(out)
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    for p,h in c['pins'].items():
        if file_hash(p)!=h:raise ValueError('pinned evidence changed: '+p)
    if checkpoint_identity(c['model'])[0]!=c['base_id']:raise ValueError('base changed')
    evidence={k:json.loads(Path(c[k]).read_text()) for k in ('parity','prefix','replay','token_assets')}
    result=admit_initial_policy(c['source_policy'],c['verified_policy'],out/'policy',base=c['model'],**evidence)
    result['evidence_sha256']={k:file_hash(c[k]) for k in evidence}
    commit_json(out/'result.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False);run(json.loads(a.config.read_text()),a.output)
