"""Bind the bulk connection to verified service sources/processor and runtime.

Image identity is checked by the launcher against Docker inspect; a process
cannot independently attest its own image. No training admission is implied.
"""
import copy
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_preparation import processor_cache_identity
from multimodal_jev.context_logits_wire import WIRE
from multimodal_jev.span_rollout_rpc import send,receive
from tools.reports.context_dual_rollout_candidate import DualContextEngineClient,RUNTIME


def verify_runtime(config):
    identity=config['dual_runtime_identity'];root=Path(config['source_root']).resolve()
    manifest=Path(config['dual_runtime_manifest'])
    if (set(identity)!={'image_sha256','source_manifest_sha256','processor_sha256','profile','output_protocol'}
            or identity['profile']!=RUNTIME or identity['output_protocol']!=WIRE
            or identity['image_sha256']!='52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
            or file_hash(manifest)!=identity['source_manifest_sha256']):
        raise ValueError('explicit pinned dual runtime identity required')
    files=json.loads(manifest.read_text())
    if not files or not isinstance(files,dict):raise ValueError('complete runtime source manifest required')
    actual={str(p.relative_to(root)) for p in root.rglob('*.py') if p.is_file()}
    if set(files)!=actual:raise ValueError('complete Python source coverage required')
    for name,digest in files.items():
        p=root/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink()
                or file_hash(p)!=digest):raise ValueError('runtime dependency changed: '+name)
    if processor_cache_identity(config['source_policy'])!=identity['processor_sha256']:
        raise ValueError('runtime processor changed')
    return copy.deepcopy(identity)


class BoundDualClient(DualContextEngineClient):
    def __init__(self,*args,runtime_identity,**kwargs):
        super().__init__(*args,**kwargs)
        try:
            hello=receive(self.socket)
            if (hello!=dict(protocol='dual-runtime-hello-v1',runtime_identity=runtime_identity,
                           policy_id=self.bundle['policy_id'],contract=self.bundle['contract'])):
                raise ValueError('actual connection runtime or policy differs')
            self.runtime_identity=copy.deepcopy(runtime_identity)
        except BaseException:
            self.phase='failed';self.socket.close();raise


def service_namespace(config):
    from tools.reports.serve_context_dual_candidate import service_namespace as original
    identity=verify_runtime(config)
    if processor_cache_identity(config['policy']['snapshot'])!=identity['processor_sha256']:
        raise ValueError('service startup processor source differs')
    namespace=original();serve=namespace['serve_connection']
    if config['policy'].get('process_device_scope') is not None:
        from tools.reports.context_dual_rollout_candidate import owner_class
        namespace['owner_class']=lambda root:owner_class(root,shared_scope=True)
    def bound(connection,engine,exchange):
        send(connection,dict(protocol='dual-runtime-hello-v1',runtime_identity=identity,
            policy_id=engine.bundle['policy_id'],contract=engine.bundle['contract']))
        return serve(connection,engine,exchange)
    namespace['serve_connection']=bound
    return namespace


def main():
    import argparse,time
    from multimodal_jev.baseline import atomic_json
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();config=json.loads(a.config.read_text());a.output.mkdir(exist_ok=False)
    try:service_namespace(config)['run'](config,a.output)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))


if __name__=='__main__':main()
