"""Collect the first three committed updates of a packaged dynamic TPO run."""
import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.watch_stage2_first_losses import Events


def read(path):
    return json.loads(path.read_bytes())


def collect(package):
    config = read(package/'operations-local.json')['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    binding = read(package/'entry/tpo.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    ops = Path(config['operations'])
    event = Events()
    try:
        while True:
            event.add(ops)
            launch = ops/'tpo-launch.json'
            if launch.exists():
                break
            if (ops/'terminal.json').exists():
                raise RuntimeError(read(ops/'terminal.json'))
            event.wait()
        handle = read(launch)
        receipt = read(ops/'finite-collected.json')
        training = binding['training']
        joint=binding['phase']=='joint_tpo'
        if joint:
            plan=read(package/'entry/joint.json')
            expected=dict(joint_sha=file_hash(package/'entry/joint.json'),source=binding['source']['sha256'],
                target_sha=plan['target_sha'],parent_sha=plan['parent_sha'],
                world_size=len(plan['training_devices']),initial_policy=plan['source_policy'])
            fixed=read(ops/'anchor-collected.json')
            if (expected['joint_sha']!=training['joint_sha'] or expected!=fixed['binding'] or
                    not fixed['passed'] or fixed['window']!='anchor' or
                    receipt['anchor_sha256']!=file_hash(ops/'anchor-collected.json')):
                raise ValueError('fixed anchor is not bound to this formal config')
        else:
            expected=dict(source=binding['source']['sha256'],
                **{k:training[k] for k in ('target_sha','parent_sha','world_size','next_macro_sha')})
        if (training['window'] != 'formal' or receipt['passed'] is not True or receipt['window'] != 'finite'
                or receipt['binding'] != expected):
            raise ValueError('finite and formal bindings differ')
        path = Path(handle['remote_root'])/'tpo'
        if handle['host'] != config['host'] or str(path.parent) != config['remote_root']:
            raise ValueError('unexpected formal location')
        observer = Path(__file__).with_name('watch_stage2_first_losses.py').read_text()
        remote = dict(format='tpo', remote_root=str(path), container_id=handle['container_id'],
                      name=config['tag']+'-tpo', image=config['image'])
        if joint:
            remote.update(config_files={name:file_hash(package/'entry'/name) for name in
                ('tpo.json','joint.json','targets/MANIFEST.json','source/FILES.json')},
                devices=plan['training_devices']+plan['inference_devices'])
        command = shlex.join(['python3','-','--remote-json',json.dumps(remote)])
        result = subprocess.run(['ssh','-o','BatchMode=yes',handle['host'],command],input=observer,
                                text=True,capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        data = json.loads(result.stdout)
        if len(data['updates'])!=3:
            raise ValueError('exactly three committed optimizer updates required')
        previous=expected.get('initial_policy')
        for i,row in enumerate(data['updates'],1):
            if joint:
                from tools.reports.joint_tpo_window import check_update
                check_update(row,i,plan,expected,previous)
                previous=row['reference']['next_policy']
                continue
            ref = row['reference']
            metrics = row['metrics']
            if (row['step'] != i or ref['policy_update'] != i-1 or ref['mode'] != 'next_macro'
                    or ref['config_sha256'] != training['next_macro_sha'] or not 0 < row['real'] <= ref['macro_slots']
                    or not all(math.isfinite(x) and x>0 for x in
                               (metrics['jev/tpo_loss'],metrics['actor/grad_norm'],ref['seconds_max']))):
                raise ValueError('invalid formal dynamic reference update')
        commit_json(ops/'first-three.json', dict(data,container_id=handle['container_id'],binding=receipt['binding']))
        return data
    finally:
        event.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('package',type=Path)
    print(json.dumps(collect(parser.parse_args().package)))
