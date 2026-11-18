"""Prepare two recovery updates from a real committed original-run snapshot."""
import argparse
import ast
import json
from pathlib import Path
import shutil

from multimodal_jev.local_config import load_config
from tools.jev.prepare_context_formal_package import verify
from tools.reports.audit_committed_context_checkpoint import audit
from tools.reports.context_arithmetic_resume_runtime import expected_config
from tools.reports.prepare_context_dual_throughput_pair import sha,write

TRAINER_SHA='bfefc772ca82410801ff4ac0bb41ce3e4bdad73ee8cf24a3978b80e79443bcb9'
RUNNER_SHA='c508acffee7c7c4bd7e183cfb3a34a2ddff19702227265dcb648da65a67b6bd5'


def transform(raw, expected_sha):
    import hashlib
    if hashlib.sha256(raw.encode()).hexdigest()!=expected_sha:
        raise ValueError('exact original arithmetic runtime required')
    before='from context_arithmetic_formal_runtime import validate'
    if raw.count(before)!=1:raise ValueError('single original admission import required')
    changed=raw.replace(before,'from context_arithmetic_resume_runtime import validate')
    ast.parse(changed)
    return changed


def prepare(c,provenance):
    source=Path(c['source']);output=Path(c['output'])
    manifest=verify(source,c['source_manifest_sha256'])
    proof=audit(dict(snapshot=c['snapshot']))
    original=json.loads((source/'config.json').read_text())
    if (proof['files']['config.json']!=sha(source/'config.json') or
            proof['files']['PREPARED.json']!=c['source_manifest_sha256'] or
            proof['algorithm']!=original['algorithm']):
        raise ValueError('checkpoint must descend from exact original package')
    if output.exists() or output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
        raise ValueError('new independent recovery package required')
    old=manifest['remote_root'];remote=c['remote_root'];step=proof['step']
    config=expected_config(original,origin_config=old+'/config.json',remote_root=remote,step=step)
    trainer='source/tools/jev/train_context_formal.py'
    changed={trainer:transform((source/trainer).read_text(),TRAINER_SHA),
             'run.py':transform((source/'run.py').read_text(),RUNNER_SHA),
             'source/tools/jev/context_arithmetic_resume_runtime.py':Path('tools/reports/context_arithmetic_resume_runtime.py').read_text()}
    output.mkdir(parents=True);shutil.copytree(source/'source',output/'source')
    for name,content in changed.items():(output/name).write_text(content)
    # Reuse the original read-only Stage1/base and already committed delta state.
    # Source copies contain only code and frozen schedule, no weights.
    binding=dict(schema='context-arithmetic-recovery-prefix-v1',runtime_root=remote,step=step,
        original_config_sha256=sha(source/'config.json'),formal_release_id=manifest['release_id'],
        original_container=proof['container_id'],original_manifest_sha256=c['source_manifest_sha256'],
        original_files=manifest['files'],changed_files={n:sha(output/n) for n in changed},
        checkpoint_files={n:h for n,h in proof['files'].items() if n.startswith('run/')},
        behavior_policy_id=proof['policy_id'],training_dispatch_allowed=False)
    write(output/'RESUME.json',binding)
    pins={remote+'/'+n if n.startswith('source/') else old+'/'+n:h for n,h in manifest['files'].items()}
    pins.update({n:h for n,h in original['pins'].items() if not n.startswith(old+'/source/')})
    pins.update({remote+'/'+n:sha(output/n) for n in changed})
    pins.update({old+'/'+n:h for n,h in binding['checkpoint_files'].items()})
    pins.update({old+'/config.json':sha(source/'config.json'),remote+'/RESUME.json':sha(output/'RESUME.json')})
    config['pins']=pins
    service=json.loads((source/'service-config.json').read_text());service['pins']=dict(pins)
    service['policy'].update(snapshot=old+f'/run/policy-{step:04d}',adapter=config['initial_adapter'],policy_id=proof['policy_id'])
    write(output/'config.json',config);write(output/'service-config.json',service)
    write(output/'CPU-RECOVERY.json',proof)
    result=dict(status='arithmetic_recovery_prefix_prepared_only',remote_root=remote,image=manifest['image'],
        config_provenance=provenance,training_dispatch_allowed=False,finite_dispatch_allowed=False,
        files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()},
        pending=['fixed-image runtime validation','native adapter tensor reload','controlled resource/driver handoff',
                 'actual NPU recovery and continuous-step comparison','independent resumed-run audit'])
    write(output/'PREPARED.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local);result=prepare(c,provenance)
    print(json.dumps(dict(status=result['status'],files=len(result['files']),training_dispatch_allowed=False)))
