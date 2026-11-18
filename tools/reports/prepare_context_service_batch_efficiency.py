"""Clone an audited100 DP2 run, changing only service query/token capacity."""
import argparse
import json
from pathlib import Path
import shutil

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports.context_macro_training_operations import validate
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.prepare_context_macro_replay import write
from tools.reports.run_context_batch_probe_candidate import verify


def prepare(c,provenance):
    parent,_=load_config(c['baseline_config'],c['baseline_local']);receipt=validate(parent)
    base=Path(parent['operations_root'])
    terminal=json.loads((base/'finalization-terminal.json').read_text())
    if terminal.get('audit_status')!='finite_dp2_efficiency_audited':
        raise ValueError('completed audited100 baseline required; runner reaudits before launch')
    source=Path(parent['prepared'])/'package';old=json.loads((source/'training.json').read_text())
    if old['microbatch']!=8 or old['stop_update']!=100 or old['start_update']!=0:
        raise ValueError('same initial micro8 hundred-step baseline required')
    output=Path(c['output']);remote=c['remote_root']
    if (output.exists() or output.resolve().is_relative_to(source.resolve())
            or Path(remote).is_relative_to(parent['remote_root'])
            or Path(parent['remote_root']).is_relative_to(remote)):
        raise ValueError('new disjoint comparison paths required')
    if c['batch_queries']!=32 or c['batch_tokens']!=32*4096:
        raise ValueError('declared service32 complete prompt budget required')
    dest=output/'package';shutil.copytree(source,dest);(dest/'FILES.json').unlink()
    t=relocate(old,parent['remote_root'],remote)
    # Policy/runtime bytes remain identical. Separate cold reference cache and
    # IPC roots arise solely through relocation.
    write(dest/'training.json',t)
    for i in range(2):
        s=relocate(json.loads((source/f'service-{i}.json').read_text()),parent['remote_root'],remote)
        s['policy'].update(max_batch_queries=c['batch_queries'],max_batch_tokens=c['batch_tokens'])
        write(dest/f'service-{i}.json',s)
    write(dest/'FILES.json',{str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()})
    r=relocate(receipt,parent['remote_root'],remote)
    r.update(files_sha256=verify(dest),service_batch_comparison=dict(
        parent_manifest_sha256=receipt['files_sha256'],baseline_operations=parent,
        changed=['service.max_batch_queries:4->32','service.max_batch_tokens:16384->131072'],
        unchanged=['runtime_source_bytes','training_micro8','global128','DP2_dualTP1',
            'initial_policy','source_schedule','GRPO_objective','checkpoint_cadence'],
        config_provenance=provenance))
    write(output/'PREPARED.json',r)
    ops=dict(parent,prepared=str(output),remote_root=remote,operations_root=c['operations_root'],
        tag=c['tag'],manifest_sha256=r['files_sha256'])
    validate(ops);write(output/'operations.json',ops)
    return dict(prepared=str(output),manifest_sha256=r['files_sha256'],launched=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(prepare(c,provenance),indent=2))
