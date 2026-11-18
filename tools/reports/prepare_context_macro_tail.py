"""Seal a fresh two-update epoch-tail diagnostic; never dispatch from prepare."""
import argparse
import json
from pathlib import Path
import shutil

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_macro_training_operations import validate as validate_parent
from tools.reports.context_macro_tail_operations import validate
from tools.reports.context_macro_tail_candidate import contract
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.prepare_context_macro_replay import write
from tools.reports.run_context_batch_probe_candidate import verify

ADDITIONS = ('tools/reports/context_macro_formal_candidate.py',
             'tools/reports/context_macro_tail_candidate.py')


def prepare(c,provenance):
    parent=json.loads(Path(c['parent_operations']).read_text());receipt=validate_parent(parent)
    source=Path(parent['prepared'])/'package';old=json.loads((source/'training.json').read_text())
    if (old['start_update']!=0 or old['stop_update']!=100 or old['microbatch']!=8
            or old.get('resume_root') or old['experiment_kind']!='efficiency'):
        raise ValueError('fresh micro8 efficiency parent required')
    output=Path(c['output']);remote=Path(c['remote_root']);project=Path(c['project'])
    check_paths({'prepared':output,'operations':c['operations_root']})
    if (output.exists() or output.resolve().is_relative_to(source.resolve())
            or not remote.is_absolute() or '..' in remote.parts or ':' in str(remote)
            or remote.is_relative_to(parent['remote_root']) or Path(parent['remote_root']).is_relative_to(remote)
            or len(str(remote/'ipc').encode())+8>107):
        raise ValueError('new disjoint bounded tail paths required')
    dest=output/'package';shutil.copytree(source,dest);(dest/'FILES.json').unlink()
    for rel in ADDITIONS:
        (dest/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(project/rel,dest/rel)
    # Preserve all parent runtime bytes. Add only the explicit diagnostic entry
    # and its full-schedule validator; keep a new identity for the resulting tree.
    write(dest/'RUNTIME-SOURCES.json',{str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*.py')})
    t=relocate(old,parent['remote_root'],str(remote))
    t.update(experiment_kind='tail_admission',stop_update=2,engineering_prefix=True,
        training_dispatch_allowed=False,checkpoint_interval=1)
    t['dual_runtime_identity']['source_manifest_sha256']=file_hash(dest/'RUNTIME-SOURCES.json')
    t['macro_client']['runtime_identity']=t['dual_runtime_identity']
    t['macro_client']['attempt_id']='tail-admission-v1'
    internal={str(remote/'package'/p.relative_to(dest)):file_hash(p) for p in dest.rglob('*')
        if p.is_file() and p.name not in ('training.json','service-0.json','service-1.json')}
    t['pins']={p:h for p,h in t['pins'].items() if not Path(p).is_relative_to(remote/'package')}
    t['pins'].update(internal);tail=contract(t);write(dest/'training.json',t)
    for i in range(2):
        s=relocate(json.loads((source/f'service-{i}.json').read_text()),parent['remote_root'],str(remote))
        s['dual_runtime_identity']=t['dual_runtime_identity'];s['pins']=internal
        write(dest/f'service-{i}.json',s)
    write(dest/'FILES.json',{str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()})
    r=relocate(receipt,parent['remote_root'],str(remote))
    r.update(status='macro_tail_prepared_only',files_sha256=verify(dest),runtime_identity=t['dual_runtime_identity'],
        tail_admission=tail,config_provenance=provenance,parent_manifest_sha256=receipt['files_sha256'],
        pending=['fixed-image preflight','actual tail NPU updates','independent output and transport audit'])
    write(output/'PREPARED.json',r)
    ops=dict(parent,prepared=str(output),remote_root=str(remote),operations_root=c['operations_root'],
        tag=c['tag'],manifest_sha256=r['files_sha256'],finite_training_dispatch_allowed=False,
        tail_diagnostic_dispatch_allowed=False)
    validate(ops);write(output/'operations.json',ops)
    return dict(prepared=str(output),manifest_sha256=r['files_sha256'],tail_admission=tail,launched=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(prepare(c,provenance),indent=2))
