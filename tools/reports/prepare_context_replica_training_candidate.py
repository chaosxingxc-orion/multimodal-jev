"""Prepare a configuration-driven replica training package for any DP/topology.

The topology (training ranks and cards, global batch, microbatch, inference
services and cards, rollout width) is data in the resolved configuration. The
package reuses the existing dual runtime, replica schedule authenticator,
replica trainer transform and macro rollout client; only the derived copies and
pins are new. Prepared only: no dispatch is granted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from multimodal_jev.local_config import load_config
from multimodal_jev.cached_context_preparation import processor_cache_identity
from multimodal_jev.context_topology_batch_plan import derive
from multimodal_jev.context_logits_wire import WIRE
from tools.reports.context_dual_rollout_candidate import RUNTIME
from tools.reports.context_replica_trainer_candidate import generate, referenced_modules
from tools.reports.context_replica_formal_candidate import generate as formal_generate, window as formal_window
from tools.reports.context_batched_owner_candidate import transformed_source
from tools.reports.prepare_context_batch_probe_candidate import TOOLS

IMAGE = 'sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
REPORTS = [
    'context_dual_trainer_candidate.py', 'context_macro_trainer_candidate.py',
    'context_replica_trainer_candidate.py', 'context_dual_runtime_candidate.py',
    'context_replica_runtime.py', 'context_replica_bulk_rpc.py',
    'context_replica_batched_service.py', 'context_dual_actor_candidate.py',
    'context_dual_session_candidate.py', 'context_probability_policy_v2.py',
    'context_dual_rollout_candidate.py', 'context_dual_worker_candidate.py',
    'serve_context_dual_candidate.py', 'serve_context_batched_candidate.py',
    'context_replica_macro_client.py',
]
FORMAL = 'context_replica_formal_candidate.py'
SOURCES = [
    'context_dual_credit.py', 'context_logits_wire.py', 'context_native_actor.py',
    'context_lora_checkpoint_scope.py', 'context_replica_macro_rollout.py',
    'context_replica_schedule.py', 'context_training_schedule.py',
    'context_topology_batch_plan.py', 'span_length_groups.py',
    'vllm_context_pooler.py', 'slurp_portable_release.py', 'local_config.py',
]
MODULE = re.compile(r'multimodal_jev\.([A-Za-z0-9_]+)')
TOOL_MODULE = re.compile(r'^\s*(?:from|import)\s+tools\.(reports|jev)\.([A-Za-z0-9_]+)', re.M)
SYNTHETIC = re.compile(r"ModuleType\('multimodal_jev\.([A-Za-z0-9_]+)'")


def required_modules(project):
    """Modules every packaged file imports, minus those synthesized at runtime.
    The parent package supplies most; the rest must ship with the candidate."""
    project = Path(project)
    files = [project / 'tools/reports'/n for n in sorted(set(TOOLS) | set(REPORTS))]
    files += [project / 'tools/jev'/n for n in JEV]
    files += [project / 'src/multimodal_jev'/n for n in SOURCES]
    names, synthetic = set(referenced_modules(project)), set()
    for path in files:
        text = path.read_text()
        names |= set(MODULE.findall(text)); synthetic |= set(SYNTHETIC.findall(text))
    return sorted(names - synthetic)


def required_tools(project):
    """Tool modules imported by packaged files; the packager copies only its
    declared lists, so an undeclared runtime import fails at first use."""
    project = Path(project)
    files = [project / 'tools/reports'/n for n in sorted(set(TOOLS) | set(REPORTS))]
    files += [project / 'tools/jev'/n for n in JEV]
    files += [project / 'src/multimodal_jev'/n for n in SOURCES]
    names = set()
    for path in files:
        names |= {f'{package}/{name}' for package, name in TOOL_MODULE.findall(path.read_text())}
    return sorted(names)
JEV = ['train_context_online_candidate.py', 'context_switch_rollout.py',
       'context_process_supervisor.py', 'context_configurable_device_scope.py']

RUN_PY = '''import json,os,sys
from pathlib import Path
from context_process_supervisor import supervise_services
from multimodal_jev.local_runtime_storage import require_training_storage
r=Path(__file__).resolve().parent
require_training_storage(r)
c=json.loads((r/'config.json').read_text())
{GATE}
def env(role,visible):
 e=dict(os.environ);e.pop('ASCEND_RT_VISIBLE_DEVICES',None)
 cache=Path('/jev-tmp')/role
 cache.mkdir(exist_ok=True)
 e['VLLM_RPC_BASE_PATH']=str(cache)
 for key in ('TMPDIR','HF_HOME','XDG_CACHE_HOME','TORCHINDUCTOR_CACHE_DIR',
             'TRITON_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH'):
  e[key]=str(cache)
 e['ASCEND_RT_VISIBLE_DEVICES']=visible
 return e
services=[]
for name,visible in {SERVICES}:
 config=json.loads((r/('service-'+name+'.json')).read_text())
 services.append(dict(name=name,
  command=[sys.executable,str(r/'source/tools/reports/context_replica_runtime.py'),
   '--config',str(r/('service-'+name+'.json')),'--output',str(r/('service-'+name))],
  env=env('service-'+name,visible),
  ready_path=r/('service-'+name)/'ready.json',
  socket_path=Path(config['socket_path'])))
actor_command=['torchrun','--nnodes=1','--node-rank=0','--master-addr=127.0.0.1',
 '--master-port=29617','--nproc-per-node={world}',str(r/'source/tools/reports/{entry}'),
 '--config',str(r/'config.json'),'--output',str(r/'run')]
report=supervise_services(services=services,actor_command=actor_command,
 actor_env=env('actor',{train_visible}),output=r/'supervision')
raise SystemExit(0 if report['passed'] else 1)
'''
ENGINEERING_GATE = '''if c['engineering_prefix'] is not True or c['training_dispatch_allowed'] is not False:
 raise ValueError('two-update engineering candidate only')
if c['start_update']!=0 or c['stop_update']!=2:
 raise ValueError('initial bounded candidate only; resume requires separate binding')'''
FORMAL_GATE = '''from multimodal_jev.context_topology_batch_plan import derive
from tools.reports.context_replica_formal_candidate import contract as formal_contract
formal_contract(c)
if (c['engineering_prefix'] is not False or c['training_dispatch_allowed'] is not True
        or c['start_update']!=0
        or c['stop_update']!=derive(c['batch_config'])['total_updates']):
 raise ValueError('declared formal source interval only')'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rebase(value, old, new):
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [rebase(v, old, new) for v in value]
    if isinstance(value, dict):
        return {rebase(k, old, new): rebase(v, old, new) for k, v in value.items()}
    return value


def topology(c):
    batch_config = c['batch_config']; plan = derive(batch_config)
    world = plan['dp_replicas']; global_batch = plan['global_batch']
    devices = c['training_devices']
    if (not isinstance(devices, list) or len(devices) != world
            or any(type(d) is not int or d < 0 for d in devices)
            or len(set(devices)) != world):
        raise ValueError('one distinct physical training card per DP replica required')
    services = c['inference']; names = set()
    if not isinstance(services, list) or len(services) != plan['configuration']['inference_instances']:
        raise ValueError('inference service count must match the declared topology')
    for service in services:
        if not isinstance(service, dict):
            raise ValueError('explicit inference service mapping required')
        if set(service) != {'name', 'device', 'max_batch_queries', 'max_batch_tokens'}:
            raise ValueError('service name/device/model limits required')
        if (not isinstance(service['name'], str) or not service['name'].isidentifier()
                or service['name'] in names):
            raise ValueError('distinct identifier service names required')
        if (type(service['device']) is not int or service['device'] < 0
                or service['device'] in devices):
            raise ValueError('service card must be a distinct physical NPU')
        if (service['max_batch_queries'] != plan['configuration']['inference_batch_queries']
                or type(service['max_batch_tokens']) is not int
                or service['max_batch_tokens'] < 1):
            raise ValueError('service batch limits must match the declared topology')
        names.add(service['name'])
    rollout = c['rollout']
    if set(rollout) != {'rollout_queries', 'timeout_seconds'}:
        raise ValueError('rollout group and timeout required')
    if (type(rollout['rollout_queries']) is not int
            or not 1 <= rollout['rollout_queries'] <= global_batch):
        raise ValueError('bounded rollout group required')
    if (type(rollout['timeout_seconds']) not in (int, float)
            or not rollout['timeout_seconds'] > 0):
        raise ValueError('positive rollout timeout required')
    if (not isinstance(c.get('attempt_id'), str)
            or not c['attempt_id'].replace('-', '').replace('_', '').isalnum()):
        raise ValueError('safe unique attempt id required')
    return plan, world, global_batch, services, rollout


def prepare(c, provenance):
    parent = Path(c['parent']); project = Path(c['project']); output = Path(c['output'])
    schedule = Path(c['schedule']); parent_schedule = Path(c['parent_schedule'])
    receipt = json.loads((parent / 'PREPARED.json').read_text())
    if sha(parent / 'PREPARED.json') != c['parent_sha256']:
        raise ValueError('parent identity changed')
    actual = {str(p.relative_to(parent)): sha(p) for p in parent.rglob('*')
              if p.is_file() and p != parent / 'PREPARED.json'}
    if actual != receipt['files']:
        raise ValueError('parent payload changed')
    remote = c['remote_root']; old = receipt['remote_root']
    if (not Path(remote).is_absolute() or '..' in Path(remote).parts or remote == old
            or output.exists() or output.resolve().is_relative_to(parent.resolve())
            or Path(remote).is_relative_to(old) or Path(old).is_relative_to(remote)):
        raise ValueError('new isolated candidate paths required')
    plan, world, global_batch, services, rollout = topology(c)
    experiment = c.get('experiment')
    if experiment is None:
        formal = None
    else:
        if (not isinstance(experiment, dict)
                or set(experiment) != {'kind', 'evaluation_steps', 'seed'}
                or experiment['kind'] != 'formal_source' or experiment['seed'] != 317):
            raise ValueError('declared formal source kind, curve and seed required')
        formal = dict(kind='formal_source', seed=317,
                      evaluation_steps=formal_window(plan, experiment['evaluation_steps']))
        formal['recovery_checkpoint_steps'] = formal['evaluation_steps'][1:]
    if sha(schedule / 'MANIFEST.json') != c['schedule_sha256']:
        raise ValueError('replica schedule identity changed')
    manifest = json.loads((schedule / 'MANIFEST.json').read_text())
    if (manifest['schema'] != 'jev-replica-source-schedule-v1'
            or manifest['batch_plan'] != plan or manifest['total_updates'] != plan['total_updates']
            or manifest['total_real_exposures'] != c['batch_config']['source_queries'] * c['batch_config']['epochs']):
        raise ValueError('authenticated replica schedule for this topology required')
    if sha(parent_schedule / 'MANIFEST.json') != c['parent_schedule_sha256']:
        raise ValueError('parent schedule identity changed')
    parent_manifest = json.loads((parent_schedule / 'MANIFEST.json').read_text())
    if (parent_manifest['schema'] not in ('context-full-learning-schedule-v1',
                                          'context-full-learning-schedule-v2')
            or parent_manifest['release_sha256'] != c['schedule_release_sha256']):
        raise ValueError('parent schedule release differs')
    original = json.loads((parent / 'config.json').read_text())
    if original['source_policy_id'] != c['source_policy_id']:
        raise ValueError('fixed Stage1 source differs')
    policy = Path(c['local_policy']); policy_manifest = json.loads((policy / 'MANIFEST.json').read_text())
    if policy_manifest['policy_id'] != c['source_policy_id']:
        raise ValueError('local source policy differs')
    processor_id = processor_cache_identity(policy)
    transform = formal_generate if formal else generate
    transform(project)  # validate the trainer transformation before copying
    output.mkdir()
    shutil.copytree(parent / 'source', output / 'source')
    stale = output / 'source/schedule'
    if stale.exists():
        shutil.rmtree(stale)
    shutil.copytree(schedule, output / 'source/schedule')
    shutil.copytree(parent_schedule, output / 'source/parent-schedule')
    for name in sorted(set(TOOLS) | set(REPORTS)):
        dest = output / 'source/tools/reports'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / 'tools/reports'/name, dest)
    if formal:
        shutil.copy2(project / 'tools/reports'/FORMAL, output / 'source/tools/reports'/FORMAL)
    for name in SOURCES:
        shutil.copy2(project / 'src/multimodal_jev'/name, output / 'source/src/multimodal_jev'/name)
    for name in JEV:
        shutil.copy2(project / 'tools/jev'/name, output / 'source/tools/jev'/name)
    transform(output / 'source')  # validate the assembled declared trainer before sealing
    # Close the import graph over the assembled tree: anything a packaged file
    # imports but the parent does not provide ships from this project. v3/v5
    # both failed at first import on modules the declared lists missed.
    for name in required_modules(project):
        dest = output / 'source/src/multimodal_jev'/f'{name}.py'
        source_file = project / 'src/multimodal_jev'/f'{name}.py'
        if not dest.is_file() and source_file.is_file():
            shutil.copy2(source_file, dest)
    for name in required_tools(project):
        dest = output / 'source/tools'/f'{name}.py'
        source_file = project / 'tools'/f'{name}.py'
        if not dest.is_file() and source_file.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, dest)
    missing = [name for name in required_modules(project)
               if not (output / 'source/src/multimodal_jev'/f'{name}.py').is_file()]
    missing += [name for name in required_tools(project)
                if not (output / 'source/tools'/f'{name}.py').is_file()]
    if formal and not (output / 'source/tools/reports'/FORMAL).is_file():
        missing.append('tools/reports/' + FORMAL)
    if missing:
        raise ValueError('packaged sources import unpackaged modules: ' + ', '.join(missing))
    transform(output / 'source')  # regenerate the assembled tree before sealing
    if formal and FORMAL not in {p.name for p in (output / 'source/tools/reports').glob('*.py')}:
        raise ValueError('formal trainer entry missing from the assembled source')
    transformed_source(output / 'source')  # reject incompatible inherited owner before sealing
    files = {str(p.relative_to(output / 'source')): sha(p)
             for p in (output / 'source').rglob('*.py') if p.is_file()}
    (output / 'RUNTIME-SOURCES.json').write_text(json.dumps(files, sort_keys=True, indent=2) + '\n')
    identity = dict(image_sha256=IMAGE.removeprefix('sha256:'),
                    source_manifest_sha256=sha(output / 'RUNTIME-SOURCES.json'),
                    processor_sha256=processor_id, profile=RUNTIME, output_protocol=WIRE)
    replicas = [dict(socket=remote + '/service-' + s['name'] + '/socket',
                     exchange=remote + '/service-' + s['name'] + '/exchange') for s in services]
    config = rebase(original, old, remote)
    config.pop('binary_policy_arithmetic', None)
    config.update(source_root=remote + '/source',
                  dual_runtime_manifest=remote + '/RUNTIME-SOURCES.json',
                  dual_runtime_identity=identity, algorithm='dual_credit_grpo_floor_clipped_v1',
                  dual_arm='joint', reference_cache=remote + '/reference-cache',
                  physical_devices=list(c['training_devices']),
                  batch_config=dict(c['batch_config']),
                  global_batch=global_batch, microbatch=c['batch_config']['microbatch'],
                  schedule=remote + '/source/schedule', schedule_sha256=c['schedule_sha256'],
                  parent_schedule=remote + '/source/parent-schedule',
                  parent_schedule_sha256=c['parent_schedule_sha256'],
                  schedule_release_sha256=c['schedule_release_sha256'],
                  macro_client=dict(schema='jev-replica-macro-client-v1',
                                    attempt_id=c['attempt_id'],
                                    journal_root=remote + '/run/pool-journal',
                                    runtime_identity=identity, replicas=replicas,
                                    timeout_seconds=float(rollout['timeout_seconds']),
                                    rollout_queries=rollout['rollout_queries'],
                                    max_queries=plan['macro_queries']),
                  start_update=0,
                  stop_update=plan['total_updates'] if formal else 2,
                  engineering_prefix=formal is None,
                  training_dispatch_allowed=formal is not None)
    if formal:
        config.update(experiment_kind='formal_source', stage='task',
                      evaluation_steps=list(formal['evaluation_steps']), seed=formal['seed'])
        # The recovery cadence is the declared curve, not the inherited prefix flag.
        config.pop('checkpoint_interval', None)
    mounted = sorted([*c['training_devices'], *[s['device'] for s in services]])
    service_values = {}
    for service in services:
        name = service['name']; device = service['device']
        value = rebase(json.loads((parent / 'service-config.json').read_text()), old, remote)
        value.update(source_root=config['source_root'], source_policy=config['source_policy'],
                     dual_runtime_manifest=config['dual_runtime_manifest'],
                     dual_runtime_identity=identity,
                     transport_protocol='bulk_dual_logits_candidate_v1',
                     socket_path=remote + '/service-' + name + '/socket',
                     exchange=remote + '/service-' + name + '/exchange')
        value['policy'].update(candidate_runtime=RUNTIME, output_protocol=WIRE,
                               physical_devices=[device],
                               max_batch_queries=service['max_batch_queries'],
                               max_batch_tokens=service['max_batch_tokens'])
        if len(mounted) > 1:
            value['policy']['process_device_scope'] = dict(
                kind='same_container_runtime_visibility_v1', mounted_physical=mounted,
                visible_ordinal=str(mounted.index(device)))
            value['policy']['device_scope_module'] = dict(
                name='context_configurable_device_scope',
                sha256=sha(output / 'source/tools/jev/context_configurable_device_scope.py'))
        else:
            value['policy'].pop('process_device_scope', None)
            value['policy'].pop('device_scope_module', None)
        service_values[name] = value
    external = {p: h for p, h in original['pins'].items() if not p.startswith(old + '/')}
    localpins = {remote + '/' + str(p.relative_to(output)): sha(p)
                 for p in output.rglob('*') if p.is_file()}
    config['pins'] = dict(external, **localpins)
    for value in service_values.values():
        value['pins'] = dict(config['pins'])
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    for name, value in service_values.items():
        (output / ('service-' + name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
    service_bindings = ', '.join(
        f"({s['name']!r}, {str(s['device'])!r})" for s in services)
    train_visible = repr(','.join(str(d) for d in c['training_devices']))
    (output / 'run.py').write_text(RUN_PY.replace('{SERVICES}', '[' + service_bindings + ']')
                                   .replace('{world}', str(world))
                                   .replace('{train_visible}', train_visible)
                                   .replace('{entry}', FORMAL if formal else
                                            'context_replica_trainer_candidate.py')
                                   .replace('{GATE}', FORMAL_GATE if formal else ENGINEERING_GATE))
    result = dict(status=('replica_formal_prepared_only' if formal
                          else 'replica_integrated_prepared_only'), remote_root=remote, image=IMAGE,
                  parent_sha256=c['parent_sha256'], config_provenance=provenance,
                  dual_runtime_identity=identity, schedule_sha256=c['schedule_sha256'],
                  batch_plan=plan, training_devices=list(c['training_devices']),
                  experiment=formal or dict(kind='engineering_prefix', updates=2),
                  inference_services=[dict(name=s['name'], device=s['device']) for s in services],
                  training_dispatch_allowed=False, finite_npu_dispatch_allowed=False,
                  layout=dict(configs=['config.json', 'run.py', 'RUNTIME-SOURCES.json', 'PREPARED.json'],
                              services={s['name']: 'service-' + s['name'] + '.json'
                                        for s in services},
                              directories=['run', 'supervision', 'reference-cache']
                              + ['service-' + s['name'] for s in services]),
                  files={str(p.relative_to(output)): sha(p)
                         for p in output.rglob('*') if p.is_file()},
                  pending=['fixed-image replica source import/check',
                           'real-audio inference passes across every declared service',
                           'NPU updates with global reduction and tail denominator',
                           'nonzero adapter head activation on every service',
                           'fresh-process resume from the committed checkpoint'])
    (output / 'PREPARED.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--local', required=True)
    args = parser.parse_args()
    cfg, prov = load_config(args.config, args.local)
    result = prepare(cfg, prov)
    print(json.dumps(dict(status=result['status'], files=len(result['files']),
                          training_dispatch_allowed=False), indent=2))
