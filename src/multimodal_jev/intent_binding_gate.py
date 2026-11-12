"""Full transformed learning-population custody before any batch selection.

Artifact checks do not establish actual process exits. A dispatch controller
must separately observe exact gate containers and authenticate frozen sources.
"""
import json
import math
from pathlib import Path
from .baseline import file_hash
from .intent_binding_inputs import load,select,prepare_item
from .intent_gate_identity import verify_population


def verify_records(populations,artifact,hf,vllm,policy_id):
    """Verify every D/Q view against independently rederived source requests."""
    expected={};sources=set()
    for arm in ('D','Q'):
        rows=populations[arm]
        if len(rows)!=356 or len({r['id'] for r in rows})!=356:
            raise ValueError('complete 89-source four-view arm required')
        # Reuse the source-view grouping validator without altering selection.
        select(rows,317,0)
        current={r['source_id'] for r in rows}
        if len(current)!=89 or (sources and current!=sources):
            raise ValueError('same complete source population required')
        sources=current
        for row in rows:
            if row['gate_id'] in expected:raise ValueError('duplicate gate input')
            expected[row['gate_id']]=row['request'].model_input()
    if len(expected)!=712 or artifact.get('recordings')!=89:
        raise ValueError('complete transformed population required')
    cpu={r['id']:r for r in artifact['records']}
    for arm in ('D','Q'):
        for row in populations[arm]:
            source=cpu.get(row['gate_id'],{})
            if (source.get('recording_id')!=row['source_id'] or source.get('arm')!=arm
                    or source.get('variant')!=row['variant']
                    or source.get('audio_sha256')!=row['audio_sha256']
                    or [m['values'] for m in source.get('candidate_mapping',[])]!=[list(q.values) for q in row['request'].questions]):
                raise ValueError('processor source/view/semantic support differs')
    indices=[]
    for name,data in (('hf',hf),('vllm',vllm)):
        if (data.get('passed') is not True or data.get('backend')!=name
                or data['base_id']+':'+data['adapter_id']!=policy_id):
            raise ValueError('matching complete backend policy required')
        verify_population(artifact,expected,data['records'],policy_id)
        indices.append({r['id']:r for r in data['records']})
    worst=0.;low=high=1.
    for uid in expected:
        left,right=(indexed[uid] for indexed in indices)
        if left['request_identity']!=right['request_identity'] or left['request_id']!=right['request_id']:
            raise ValueError('backend full input identity differs')
        lower=upper=0.
        for a,b in zip(left['candidate_logprobs'],right['candidate_logprobs'],strict=True):
            delta=[x-y for x,y in zip(a,b,strict=True)]
            worst=max(worst,max(map(abs,delta)));lower+=min(delta);upper+=max(delta)
        if worst>.02 or lower<math.log(.98) or upper>math.log(1.02):
            raise ValueError('complete transformed policy parity failed')
        low=min(low,math.exp(lower));high=max(high,math.exp(upper))
    return indices,dict(passed=True,requests=712,source_recordings=89,policy_id=policy_id,
        maximum_logprob_error=worst,joint_ratio_range=[low,high])


def authenticate_artifacts(config,policy_id):
    """Authenticate pinned full evidence; no Docker observation or dispatch."""
    def pinned(path):
        path=Path(path)
        if config['pins'].get(str(path))!=file_hash(path):
            raise ValueError('binding gate evidence unpinned or changed: '+str(path))
        return json.loads(path.read_text())
    root=Path(config['acceptance_root']);gate=pinned(root/'config.json')
    launch=pinned(root/'launch.json');manifest=pinned(root/'source-manifest.json')
    if (launch['image']!=config['image'] or launch['physical_devices']!=[4]
            or launch['config_sha256']!=file_hash(root/'config.json')
            or launch['source_manifest_sha256']!=file_hash(root/'source-manifest.json')):
        raise ValueError('fixed TP1 gate launch required')
    for path,digest in gate['pins'].items():
        if file_hash(path)!=digest:raise ValueError('gate pinned asset changed')
    for path,digest in manifest.items():
        if file_hash(root/'source'/path)!=digest:raise ValueError('gate source changed')
    if (gate['model']!=config['model_path'] or gate['adapter']!=config['initial_adapter']
            or gate['runtime_profile']!=config['runtime_profile']
            or gate['population']!=dict(recordings=89,requests=712)
            or gate['base_id']+':'+gate['adapter_id']!=policy_id):
        raise ValueError('binding gate policy or population differs')
    from .policy_identity import checkpoint_identity
    if checkpoint_identity(config['model_path'])[0]+':'+checkpoint_identity(config['initial_adapter'])[0]!=policy_id:
        raise ValueError('actual checkpoint identity differs')
    for name in ('terminal.json','hf/terminal.json','vllm/terminal.json'):
        if pinned(root/'run'/name)['exit_code']!=0:raise ValueError('binding gate phase failed')
    ready=pinned(root/'run/READY.json');hf=pinned(root/'run/hf/result.json');vl=pinned(root/'run/vllm/result.json')
    if (ready['passed'] is not True or ready['requests']!=712
            or ready['base_id']+':'+ready['adapter_id']!=policy_id
            or ready['result_sha256']!=file_hash(root/'run/vllm/result.json')):
        raise ValueError('binding readiness differs')
    artifact=pinned(gate['inputs'])
    if gate['pins'].get(gate['inputs'])!=file_hash(gate['inputs']):raise ValueError('gate processor pin differs')
    populations={a:load(config['binding_release'],config['binding_manifest_sha256'],a)[0] for a in ('D','Q')}
    indices,audit=verify_records(populations,artifact,hf,vl,policy_id)
    return populations,indices,dict(audit,gate_container_id=launch['container_id'],
        gate_config_sha256=file_hash(root/'config.json'),
        hf_sha256=file_hash(root/'run/hf/result.json'),vllm_sha256=file_hash(root/'run/vllm/result.json'),
        actual_exit_observed=False)


def prepare(config,processor,policy_id,step):
    """Full-population authentication precedes the 8-source/32-view selection.

    The controller must observe the returned exact container exited zero before
    launching training. This function alone never grants device ownership.
    """
    populations,indices,audit=authenticate_artifacts(config,policy_id)
    arm=config['arm']
    if arm not in ('D','Q','P'):raise ValueError('registered binding arm required')
    rows=populations[arm] if arm!='P' else load(config['binding_release'],config['binding_manifest_sha256'],'P')[0]
    chosen,selection=select(rows,config['seed'],step)
    prepared=[]
    for row in chosen:
        item,_=prepare_item(row,processor,policy_id,indices[0][row['gate_id']],indices[1][row['gate_id']])
        prepared.append(item)
    return prepared,dict(selection,step=step,seed=config['seed'],arm=arm,gate=audit,
        binding_manifest_sha256=config['binding_manifest_sha256'])
