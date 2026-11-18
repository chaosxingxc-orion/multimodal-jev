"""Seal batch32/48 probes from the previously authenticated TRAIN128 package.

Reuses the same serial reference method, actual worker telemetry, source bytes,
model, inputs and dual-wire measurement. No remote call or launch.
"""
import argparse
import json
from pathlib import Path
import shutil

from multimodal_jev.local_config import load_config
from tools.reports.context_dual_probe_operations import validate as validate_parent
from tools.reports.context_capacity_probe_operations import namespace
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.prepare_context_batch_probe_candidate import sha,ROOT,command
from tools.reports.prepare_context_dual_probe_candidate import rebind_argv
from tools.reports.launch_context_batch_probe_candidate import write_new
from tools.reports.run_context_batch_probe_candidate import verify


def prepare(c,provenance):
    parent,_=load_config(c['parent_config'],c['parent_local'])
    receipt,old=validate_parent(parent)
    source=Path(parent['prepared'])/'package';output=Path(c['output'])
    if output.exists() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError('new independent capacity package root required')
    if set(c['arms'])!={'batch32','batch48'}:raise ValueError('declared32/48 pair required')
    output.mkdir(parents=True);results={}
    for name in ('batch32','batch48'):
        arm=c['arms'][name];batch=int(name[5:]);remote=arm['remote_root']
        dest=output/name;dest.mkdir();package=dest/'package'
        shutil.copytree(source,package);(package/'FILES.json').unlink()
        config=relocate(old,parent['remote_root'],remote)
        config['policy'].update(max_batch_queries=batch,max_batch_tokens=arm['batch_tokens'])
        config['pins']={remote+'/package/'+str(p.relative_to(package)):sha(p)
                       for p in package.rglob('*') if p.is_file() and p.name!='config.json'}
        (package/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        write_new(package/'FILES.json',{str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file()})
        record=dict(receipt,remote_root=remote,tag=arm['tag'],files_sha256=verify(package),
            parent_files_sha256=receipt['files_sha256'],config_provenance=provenance,
            argv=rebind_argv(command(parent['model']),ROOT,remote,parent['model'],parent['model'],
                             parent['driver_root'],7,arm['tag']),
            scope='TRAIN128 serial vs batch32/48 capacity, exact logits/prob/G16 and measured worker forwards; no macro192 or training claim')
        write_new(dest/'PREPARED.json',record)
        operations=dict(parent,prepared=str(dest),remote_root=remote,tag=arm['tag'],
            operations_root=arm['operations_root'],package_manifest_sha256=record['files_sha256'],
            batch_queries=batch,batch_tokens=arm['batch_tokens'],priority_mode='idle_resource_window',
            finite_inference_dispatch_allowed=True)
        namespace(operations)['validate'](operations)
        write_new(dest/'operations.json',operations)
        results[name]=dict(prepared=str(dest),manifest_sha256=record['files_sha256'])
    if verify(source)!=receipt['files_sha256']:raise ValueError('parent changed')
    result=dict(schema='jev-service-capacity-probe-pair-v1',arms=results,
        input_sha256=old['input_sha256'],policy_id=old['policy']['policy_id'],
        planned_query_capacity=[32,48],training_dispatch_allowed=False,launched=False)
    write_new(output/'PAIR.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(prepare(c,provenance),indent=2))
