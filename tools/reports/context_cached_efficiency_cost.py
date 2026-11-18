"""Account for cached GRPO loop time and one full reference-build container.

Measurement only, not a training/parity audit or a matched speedup claim.
Inputs are immutable collected artifacts; no remote requests or model calls.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.context_macro_efficiency_summary import summarize
from multimodal_jev.context_topology_batch_plan import derive


def seconds(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
        raise ValueError('finite nonnegative measured duration required')
    return value


def container_window(observed,handle):
    state=observed['State']
    if (observed['Id']!=handle['container_id'] or observed['Image']!=handle['image']
            or state['Status']!='exited' or state['Running'] is not False
            or state['ExitCode']!=0 or state['OOMKilled'] is not False):
        raise ValueError('exact successful terminal container required')
    start=datetime.fromisoformat(state['StartedAt'].replace('Z','+00:00'))
    end=datetime.fromisoformat(state['FinishedAt'].replace('Z','+00:00'))
    if start.utcoffset() is None or end.utcoffset() is None or end<=start:
        raise ValueError('ordered timezone-aware container timestamps required')
    return start,end,(end-start).total_seconds()


def compose(training,reference_build,reference_result,training_inspect,training_handle,
            reference_inspect,reference_handle,statistics):
    plan=derive(training['batch_config'])
    if (training['experiment_kind']!='efficiency' or training['start_update']!=0
            or training['stop_update']!=100 or training.get('resume_root')
            or plan['dp_replicas']!=2 or plan['global_batch']!=128
            or plan['configuration']['inference_instances']!=2
            or plan['configuration']['source_queries']!=40513
            or statistics['optimizer_steps']!=100 or statistics['warmup_updates']!=5
            or statistics['all_updates']['real_query_exposures']!=12800
            or statistics['measured']['real_query_exposures']!=12160
            or statistics['all_updates']['reference_misses']!=0):
        raise ValueError('complete 100-step DP2 cached efficiency run and fixed warmup required')
    dataset=training['reference_dataset'];identity=dataset['identity']
    if (reference_result['manifest_sha256']!=dataset['manifest_sha256']
            or reference_result['identity']!=identity
            or reference_result['recordings']!=40513
            or reference_result['population']!=dataset['population']
            or len(dataset['population'])!=40513
            or dataset['root']!=reference_build['output']+'/dataset'
            or identity['policy_id']!=training['source_policy_id']
            or identity['runtime_identity']!=training['dual_runtime_identity']):
        raise ValueError('same complete Stage1 reference required')
    for key in ('source_policy_id','base_id','source_contract','release_sha256','dual_runtime_identity'):
        if training[key]!=reference_build[key]:raise ValueError('reference identity differs: '+key)
    train_start,train_end,train_seconds=container_window(training_inspect,training_handle)
    ref_start,ref_end,ref_seconds=container_window(reference_inspect,reference_handle)
    if (reference_handle['container_id']==training_handle['container_id']
            or reference_handle['image']!=training_handle['image'] or ref_end>train_start):
        raise ValueError('separate completed reference build before same-image training required')
    loop=seconds(statistics['all_updates']['complete_seconds'])
    build=seconds(reference_result['total_seconds'])
    if loop<=0 or loop>train_seconds+1e-6 or build>ref_seconds+1e-6:
        raise ValueError('nested loop/build time exceeds container wall time')
    # Both lifecycle containers reserve the same four-card lease. Only the
    # reference services actively infer on two cards; allocation is not usage.
    for observed in (training_inspect,reference_inspect):
        devices={d['PathOnHost'] for d in observed['HostConfig']['Devices']
            if Path(d['PathOnHost']).name.startswith('davinci')
            and Path(d['PathOnHost']).name[7:].isdigit()}
        if len(devices)!=4:raise ValueError('four mapped NPU devices required')
    train_devices={d['PathOnHost'] for d in training_inspect['HostConfig']['Devices']}
    ref_devices={d['PathOnHost'] for d in reference_inspect['HostConfig']['Devices']}
    if train_devices!=ref_devices:raise ValueError('same declared device mappings required')
    return dict(schema='jev-cached-efficiency-cost-v1',status='measurements_only_independent_admission_required',
        loop=statistics,reference=dict(recordings=40513,build_seconds=build,
            container_seconds=ref_seconds,container_non_build_seconds=ref_seconds-build),
        training=dict(container_seconds=train_seconds,loop_seconds=loop,
            container_non_step_seconds=train_seconds-loop),
        combined=dict(real_training_exposures=12800,
            serial_container_seconds=ref_seconds+train_seconds,
            training_exposures_per_container_second=12800/(ref_seconds+train_seconds),
            allocated_device_hours=4*(ref_seconds+train_seconds)/3600,
            gap_between_containers_seconds=(train_start-ref_end).total_seconds(),
            elapsed_first_start_to_last_finish_seconds=(train_end-ref_start).total_seconds()),
        unavailable=['active_NPU_device_hours','cold_start_subphases','controller_audit_seconds',
            'package_transfer_collection_seconds','full_formal_run_cost'],
        reference_cost_charged_once=True,reference_reuse_amortization=False,
        correctness_audit_required=True,matched_speedup_claim=False,efficacy_claim=False)


def run(c,provenance):
    keys=('training_config','reference_build_config','reference_result','training_inspect',
          'training_handle','reference_inspect','reference_handle','training_transport','reference_transport')
    values={};hashes={}
    for key in keys:
        entry=c[key];path=Path(entry['path'])
        if path.is_symlink() or file_hash(path)!=entry['sha256']:
            raise ValueError('bound cost input changed: '+key)
        hashes[key]=entry['sha256'];values[key]=json.loads(path.read_text())
    if values['training_handle']['config_sha256']!=hashes['training_config']:
        raise ValueError('training launch/config binding differs')
    stats=summarize(c['run_root'],start_update=0,stop_update=100,warmup_updates=5)
    for prefix in ('training','reference'):
        transport=values[prefix+'_transport'];handle=values[prefix+'_handle'];observed=values[prefix+'_inspect']
        if (transport['status']!='transport_complete_independent_audit_required'
                or transport['container_id']!=handle['container_id'] or transport['image']!=handle['image']
                or transport['state']!=observed['State'] or transport['remote_root']!=handle['remote_root']
                or any(transport[k] for k in ('bad_pins','bad_package','launch_binding_mismatches'))):
            raise ValueError('bound complete transport required: '+prefix)
    train_files=values['training_transport']['files'];ref_files=values['reference_transport']['files']
    if (train_files.get('config.json')!=hashes['training_config']
            or ref_files.get('config.json')!=values['reference_handle']['config_sha256']
            or ref_files.get('reference-build-config.json')!=hashes['reference_build_config']
            or ref_files.get('reference-build/result.json')!=hashes['reference_result']
            or any(train_files.get('run/'+name)!=digest for name,digest in stats['files'].items())):
        raise ValueError('measured files differ from exact container collection')
    proof=compose(values['training_config'],values['reference_build_config'],values['reference_result'],
        values['training_inspect'],values['training_handle'],values['reference_inspect'],values['reference_handle'],stats)
    proof.update(input_sha256=hashes,config_provenance=provenance,
        implementation_sha256=file_hash(__file__))
    commit_json(c['output'],proof);return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(run(c,provenance),indent=2))
