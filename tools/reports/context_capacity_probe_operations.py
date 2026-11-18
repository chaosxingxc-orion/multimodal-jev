"""Reuse finite probe launch/audit with explicit batch32/48 capacity controls."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config

PARENT_SHA='eeff28dc8c7f408ad511a1fe72ee50ad1497faad4f4c3df301441cf9a83af2bb'


def namespace(c):
    if (type(c['batch_queries']) is not int or c['batch_queries'] not in (32,48)
            or type(c['batch_tokens']) is not int or c['batch_tokens'] < 4096):
        raise ValueError('explicit batch32/48 and token budget required')
    path=Path(__file__).with_name('context_dual_probe_operations.py')
    if file_hash(path)!=PARENT_SHA:raise ValueError('finite operations source changed')
    source=path.read_text()
    before="policy['max_batch_queries']!=4 or policy['max_batch_tokens']!=16384"
    if source.count(before)!=1:raise ValueError('capacity guard anchor changed')
    source=source.replace(before,"policy['max_batch_queries']!=c['batch_queries'] or policy['max_batch_tokens']!=c['batch_tokens']")
    anchor="        checked(['mkdir',remote_root])"
    if source.count(anchor)!=1:raise ValueError('isolated directory creation anchor changed')
    source=source.replace(anchor,
        "        checked(['mkdir','-p',str(Path(remote_root).parent)])\n"+anchor)
    ns={'__name__':__name__+'.derived'}
    exec(compile(source,'<capacity-probe-operations>','exec'),ns)
    return ns


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','launch','audit'],default='validate')
    a=p.parse_args();c,provenance=load_config(a.config,a.local);ns=namespace(c)
    if a.action=='launch':result=ns['launch'](c,provenance)
    elif a.action=='audit':
        base=Path(c['operations_root']);tag=c['tag']
        result=ns['audit'](c,base/(tag+'-collected'),base/(tag+'-launch.json'))
        ns['write_new'](base/(tag+'-audit.json'),result)
    else:
        receipt,_=ns['validate'](c)
        result=dict(status='validated_only',files_sha256=receipt['files_sha256'])
    print(json.dumps(result,indent=2))
