"""Record terminal actor device OOM from collected exact-container evidence.

No remote observation, retries, cleanup, or inference of untested capacities.
Docker OOMKilled and accelerator allocation failure are separate facts.
"""
import argparse
import json
from pathlib import Path
import re

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports.context_macro_probe_operations import inspect_runtime
from tools.reports.launch_context_batch_probe_candidate import write_new


def report(c):
    root=Path(c['operations_root']); hashes={}
    def read(name):
        path=root/name
        if path.is_symlink():raise ValueError('regular collected evidence required')
        hashes[name]=file_hash(path)
        return path.read_text()
    terminal=json.loads(read('terminal.json'))
    if terminal['exit_code']!=1:
        raise ValueError('failed terminal attempt required')
    states={}
    for role in ('preflight','client','service-0','service-1'):
        handle=json.loads(read(role+'-launch.json'))
        observed=json.loads(read(role+'-inspect.json'))
        inspect_runtime(observed,handle)
        state=observed['State']
        if state['Running'] or state['Status']!='exited':
            raise ValueError('all exact attempt containers must be terminal')
        if role=='client' and state['ExitCode']==0:
            raise ValueError('failed actor client required')
        states[role]=dict(container_id=observed['Id'],state=state)
    log=read('client.log')
    pattern=r'\[rank(\d+)\]: torch\.OutOfMemoryError: NPU out of memory\. Tried to allocate ([\d.]+) MiB \(NPU (\d+); ([\d.]+) GiB total capacity; ([\d.]+) GiB already allocated; ([\d.]+) GiB current active; ([\d.]+) (KiB|MiB) free; ([\d.]+) GiB reserved'
    matches=re.findall(pattern,log)
    if {r[0] for r in matches}!={'0','1'} or len(matches)!=2:
        raise ValueError('one explicit device OOM record per DP2 rank required')
    failures=[]
    for rank,requested,device,total,allocated,active,free,unit,reserved in matches:
        if rank!=device:raise ValueError('local rank/device OOM identity differs')
        failures.append(dict(rank=int(rank),local_device=int(device),requested_mib=float(requested),
            capacity_gib=float(total),allocated_gib=float(allocated),active_gib=float(active),
            free_bytes=float(free)*(1024 if unit=='KiB' else 1024**2),reserved_gib=float(reserved)))
    return dict(schema='jev-terminal-actor-device-oom-v1',classification='actor_device_allocation_failed',
        terminal=terminal,containers=states,rank_failures=failures,evidence_sha256=hashes,
        retry=False,untested_larger_microbatch='unmeasured',
        scope='Collected terminal container identity and log evidence. No live memory attribution, '
              'failed-update count, core-dump diagnosis, inference-capacity or speed claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    result=report(c);result['config_provenance']=provenance
    write_new(Path(c['output']),result)
    print(json.dumps(result,indent=2))
