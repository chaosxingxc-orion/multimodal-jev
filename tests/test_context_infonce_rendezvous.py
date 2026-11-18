"""Real torchrun startup with the exact candidate argv in a network=none image."""
import ast
from pathlib import Path
import subprocess
import sys


def test_static_loopback_rendezvous_without_hostname_resolution(tmp_path):
    # Read the runner literal without loading all deployment dependencies.
    tree=ast.parse((Path(__file__).resolve().parents[1]/'tools/reports/context_instruction_operations.py').read_text())
    ns={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('LEGACY_RUNNER','RUNNER'):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<runner>','exec'),ns)
    runner=ast.parse(ns['RUNNER'])
    assignment=next(n for n in runner.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='argv')
    script=tmp_path/'source/tools/reports/context_infonce_trainer.py';script.parent.mkdir(parents=True)
    script.write_text('''import os,socket
import torch.distributed as dist
from datetime import timedelta
assert os.environ['MASTER_ADDR']=='127.0.0.1'
original=socket.getaddrinfo
def reject_hostname(host,*args,**kwargs):
 if host not in ('127.0.0.1','localhost'):raise RuntimeError('hostname forbidden')
 return original(host,*args,**kwargs)
socket.getaddrinfo=reject_hostname
dist.init_process_group('gloo',timeout=timedelta(seconds=60))
assert dist.get_world_size()==2
dist.barrier();dist.destroy_process_group()
''')
    args=dict(sys=sys,c={'physical_devices':[0,1]},r=tmp_path)
    exec(compile(ast.Module(body=[assignment],type_ignores=[]),'<argv>','exec'),args)
    assert '--standalone' not in args['argv']
    result=subprocess.run(args['argv'],capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
