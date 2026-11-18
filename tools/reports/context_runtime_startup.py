"""Pinned launch preflight and protocol-aware shutdown for the shared runtime.

Executed before allocating inference engines. No model load or optimizer work.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import socket
from types import FunctionType


def preflight(config):
    root=Path(config['objective_extension']['root'])
    spec=importlib.util.spec_from_file_location('pinned_bootstrap',root/'bootstrap.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    trainer=module.install(config)
    from multimodal_jev.objective_plugin import ObjectivePlugin
    plugin=ObjectivePlugin(config['objective_plugin'])
    kind=config['experiment_kind']
    if kind=='cached_reference_tail_admission':
        from tools.reports.context_cached_tail_trainer import generate
        generator=FunctionType(trainer.generate.__code__,dict(trainer.generate.__globals__,original_generate=lambda source_root,formal=False:generate(source_root)))
        generator.__kwdefaults__=trainer.generate.__kwdefaults__
    else:generator=trainer.generate
    compile(generator(config['source_root'],formal=kind=='formal_source'),'<startup-generated-trainer>','exec')
    return dict(status='actor_import_and_generation_preflight_passed',objective_plugin=plugin.identity,npu_executed=False)


def close_replica(address,timeout):
    from multimodal_jev.span_rollout_rpc import send,receive
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout);connection.connect(str(address))
        greeting=receive(connection)
        if (not isinstance(greeting,dict)
                or set(greeting)!={'protocol','max_request_queries','runtime_identity','policy_id','contract'}
                or greeting['protocol']!='replica-macro-runtime-hello-v1'
                or type(greeting['max_request_queries']) is not int
                or greeting['max_request_queries']<=0
                or not isinstance(greeting['runtime_identity'],dict)
                or not isinstance(greeting['contract'],dict)
                or not isinstance(greeting['policy_id'],str) or not greeting['policy_id']):
            raise ValueError('owned replica greeting differs')
        send(connection,dict(operation='close',sequence=0))
        reply=receive(connection)
        if (not isinstance(reply,dict) or set(reply)!={'sequence','ok','result'}
                or type(reply['sequence']) is not int or reply['sequence']!=0
                or reply['ok'] is not True or reply['result'] is not None):
            raise ValueError('replica close rejected')
        return reply


def supervise_services(**kwargs):
    from context_process_supervisor import supervise_services as original
    fn=FunctionType(original.__code__,dict(original.__globals__,close_unconnected_service=close_replica),
                    argdefs=original.__defaults__)
    fn.__kwdefaults__=original.__kwdefaults__
    return fn(**kwargs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    print(json.dumps(preflight(json.loads(Path(a.config).read_text()))))
