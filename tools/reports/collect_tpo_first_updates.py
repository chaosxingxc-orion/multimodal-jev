"""Collect the first three committed updates of a packaged TPO run."""
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


def check_fixed_update(row, step, batch_real):
    metrics = row['metrics']
    loss = metrics['jev/tpo_loss']
    if (row['step'] != step or row['real'] != batch_real[step-1] or 'reference' in row
            or metrics['jev/reward_projection_kl'] != loss
            or not all(math.isfinite(v) and v > 0 for v in (loss, metrics['actor/grad_norm'], row['seconds']))):
        raise ValueError('invalid formal fixed reference update')


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
        dynamic='next_macro_sha' in training
        fixed_view = None
        if not (joint or dynamic):
            path = package/'entry/targets/MANIFEST.json'
            fixed_view = read(path)
            if (file_hash(path) != training['target_sha'] or fixed_view['schema'] != 'jev-tpo-reward-view-v1'
                    or not fixed_view.get('bidirectional_contrast')):
                raise ValueError('fixed bidirectional target manifest is not bound')
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
            keys = ('target_sha','parent_sha','world_size') + (('next_macro_sha',) if dynamic else ())
            expected=dict(source=binding['source']['sha256'], **{k:training[k] for k in keys})
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
        elif fixed_view:
            groups = config['chains'][0]
            formal, = [job for group in groups for job in group if job['name']=='tpo']
            remote.update(config_files={name: file_hash(package/'entry'/name) for name in
                ('tpo.json', 'targets/MANIFEST.json', 'source/FILES.json')}, devices=formal['devices'])
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
            if fixed_view:
                check_fixed_update(row, i, fixed_view['batch_real'])
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
