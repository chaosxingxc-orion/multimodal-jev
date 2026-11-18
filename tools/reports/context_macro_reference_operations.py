"""Finite retained fixed-reference precompute over the isolated two-service lifecycle.

Cards 6/7 serve the sealed Stage1 reference; the client only streams label-free
encoded inputs and writes fixed-reference probabilities. No actor, optimizer,
label or reward path is involved.
"""
import argparse
import json
from pathlib import Path
import re

from multimodal_jev.local_config import load_config
from tools.reports.audit_context_macro_probe import collect_and_audit
from tools.reports.context_macro_probe_operations import command as probe_command, run as run_services
from tools.reports.context_macro_reference_fill import SCHEMA as FILL_SCHEMA, audit_fill
from tools.reports.context_macro_training_operations import reusable_mounts
from tools.reports.prepare_context_batch_probe_candidate import IMAGE
from tools.reports.run_context_batch_probe_candidate import verify

RESULT_DIRS=('run','journal','service-0','service-1')
ENTRY_COUNT=40513


def validate(c):
    prepared=Path(c['prepared']);package=prepared/'package'
    receipt=json.loads((prepared/'PREPARED.json').read_text())
    if (receipt['status']!='reference_fill_prepared_only' or receipt['image']!=IMAGE
            or receipt['files_sha256']!=c['manifest_sha256'] or verify(package)!=c['manifest_sha256']
            or receipt['remote_root']!=c['remote_root'] or receipt['training_dispatch_allowed'] is not False
            or receipt['device_scope']!=[[6],[7]]):
        raise ValueError('authenticated finite reference fill payload required')
    for key in ('host','tag'):
        if not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]*',c[key]):raise ValueError('safe host/tag required')
    remote=Path(c['remote_root'])
    if (not remote.is_absolute() or '..' in remote.parts or str(remote) in ('/','/data','/mnt')
            or not Path(receipt['ipc_root']).is_relative_to(remote)):
        raise ValueError('isolated local root and contained IPC required')
    fill=json.loads((package/'fill.json').read_text());client=fill['macro_client']
    if (fill['schema']!=FILL_SCHEMA or fill['source_root']!=str(remote/'package')
            or fill['output']!=str(remote/'run') or fill['expected_entries']!=ENTRY_COUNT
            or fill['dual_runtime_identity']!=receipt['runtime_identity']
            or client['runtime_identity']!=receipt['runtime_identity']
            or client['attempt_id']!='reference-fill-v1' or client['journal_root']!=str(remote/'journal')
            or client['max_queries']!=128 or client['rollout_queries']!=32):
        raise ValueError('macro128 fill output/runtime contract differs')
    cache=Path(fill['reference_cache'])
    if (not cache.is_absolute() or '..' in cache.parts or ':' in str(cache)
            or cache.is_relative_to(remote) or remote.is_relative_to(cache)):
        raise ValueError('retained reference cache must be disjoint from the fill root')
    if fill['reference_cache']==str(remote/'reference-cache'):
        raise ValueError('fill must target the retained cache, not an arm-local one')
    for i in range(2):
        service=json.loads((package/f'service-{i}.json').read_text());endpoint=client['replicas'][i]
        if (service['policy']['physical_devices']!=[6+i]
                or service['policy']['model']!=c['model']
                or service['policy']['policy_id']!=fill['source_policy_id']
                or service['policy']['snapshot']!=fill['source_policy']
                or service['policy']['adapter']!=fill['reference_adapter']
                or service['policy']['contract']!=fill['source_contract']
                or service['policy'].get('process_device_scope') is not None
                or service['dual_runtime_identity']!=fill['dual_runtime_identity']
                or service['source_root']!=fill['source_root']
                or service['socket_path']!=endpoint['socket'] or service['exchange']!=endpoint['exchange']
                or not all(Path(v).is_relative_to(receipt['ipc_root']) for v in endpoint.values())):
            raise ValueError('isolated reference service/runtime/IPC differs')
    pins=json.loads((package/'FILES.json').read_text())
    for path,digest in fill['pins'].items():
        if Path(path).is_relative_to(remote):
            if pins.get(str(Path(path).relative_to(remote/'package')))!=digest:
                raise ValueError('internal dependency differs')
    return receipt


def command(c,role):
    argv=probe_command(c,role);root=c['remote_root'];package=root+'/package'
    if role in ('client','preflight'):
        fill=json.loads((Path(c['prepared'])/'package/fill.json').read_text())
        mounts=reusable_mounts(c,fill)
        for path,_ in mounts:
            if (not Path(path).is_absolute() or ':' in path or '..' in Path(path).parts
                    or Path(path).is_relative_to(root) or Path(root).is_relative_to(path)):
                raise ValueError('reusable inputs and runtime must be disjoint')
        before=argv.index('--entrypoint');extra=[]
        for path,readonly in mounts:extra+=['-v',path+':'+path+(':ro' if readonly else '')]
        argv[before:before]=extra
        mapped=[root,c['model'],c['driver_root'],*[path for path,_ in mounts]]
        if any(not any(Path(p).is_relative_to(m) for m in mapped) for p in fill['pins']):
            raise ValueError('dependency outside explicit mounts')
        if role=='client':
            code=("import json;from tools.reports.context_macro_reference_fill import run;"
                  "run(json.load(open("+repr(package+'/fill.json')+")),{'source':'sealed_reference_fill'})")
        else:
            code="""import json
from pathlib import Path
from tools.reports.context_macro_reference_fill import load_entry, entry_paths
from tools.reports.context_dual_runtime_candidate import verify_runtime
from tools.reports.run_context_batch_probe_candidate import verify
from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
root=Path(ROOT);verify(root)
for i in range(2):verify_runtime(json.loads((root/'service-'+str(i)+'.json').read_text()))
c=json.loads((root/'fill.json').read_text());verify_runtime(c)
b=read_execution_bundle(c['source_policy'],expected_policy_id=c['source_policy_id'],expected_contract=c['source_contract'],expected_base_id=c['base_id'])
verify_native_adapter(c['reference_adapter'],b)
paths=entry_paths(c['encoded_audio_cache'],c['expected_entries'])
print('fixed-image fill imports, sealed runtime, reference policy and '+str(len(paths))+' encoded entries passed')
""".replace('ROOT',repr(package))
        argv=argv[:argv.index(IMAGE)+1]+['python','-B','-c',code]
    return argv


def run(c,provenance):
    if c.get('finite_reference_fill_dispatch_allowed') is not True:
        raise ValueError('finite reference fill disabled')
    return run_services(dict(c,finite_inference_dispatch_allowed=True),provenance,
        validator=validate,command_builder=command,required_devices=(6,7))


def collect(c):
    return collect_and_audit(c,validator=validate,command_builder=command,
        output_auditor=audit_fill,result_dirs=RESULT_DIRS)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','run','collect'],default='validate');a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    if a.action=='run':
        print(json.dumps(run(c,provenance),indent=2))
    elif a.action=='collect':
        from tools.reports.launch_context_batch_probe_candidate import write_new
        result=collect(c);write_new(Path(c['operations_root'])/'audit.json',result)
        print(json.dumps(result,indent=2))
    else:
        print(json.dumps(validate(c),indent=2))
