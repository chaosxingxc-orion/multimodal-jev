"""Exercise the complete archive auditor with synthetic CPU evidence.

Only dataset/schedule loading and storage-device checks are substituted. Model
inference, Docker execution and FSDP are NOT performed: their saved records are
fixtures. Actual policy exports, Adam steps/shards, credit sampling, archive
hashes, runtime inspection checks and recursive replay audits run normally.
"""
import copy
import json
from pathlib import Path
import shutil
import tarfile

import pytest
import torch

from multimodal_jev.context_candidate_task import compile_candidates, bind_candidate_targets
from multimodal_jev.context_dual_credit import prepare
from multimodal_jev.context_logits_wire import WIRE
from multimodal_jev.context_policy_artifacts import read_execution_bundle, export_native_adapter
from multimodal_jev.context_policy_snapshot import export_policy, lora_contract
from multimodal_jev.context_training_checkpoint import capture
from multimodal_jev.context_lora_checkpoint_scope import validate_trainable_scope
from tools.reports.audit_context_dual_training import audit, expected_contract
from tools.reports.context_dual_group_audit import digest
from tools.reports.context_dual_training_operations import validate, command, IMAGE
from tools.reports.prepare_context_dual_replay import seal
from tools.reports.prepare_context_dual_throughput_pair import sha
from test_context_dual_training_operations import fixture as operations_fixture
from test_context_policy_artifacts import fixture as policy_fixture


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def reseal_package(ops):
    package = Path(ops['package'])
    manifest = json.loads((package/'PREPARED.json').read_text())
    manifest['files'] = {str(p.relative_to(package)): sha(p) for p in package.rglob('*')
                         if p.is_file() and p.name != 'PREPARED.json'}
    write(package/'PREPARED.json', manifest)
    ops['manifest_sha256'] = sha(package/'PREPARED.json')


def transport(c):
    root = Path(c['collection'])
    archive = root.with_suffix('.tar')
    with tarfile.open(archive, 'w') as tar:
        for p in sorted(root.rglob('*')):
            if p.is_file():
                tar.add(p, arcname=str(p.relative_to(root)), recursive=False)
    launch = json.loads(Path(c['launch']).read_text())
    inspected = json.loads(Path(c['inspect']).read_text())
    write(c['transport'], dict(status='transport_complete_independent_audit_required',
        bad_pins=[], bad_package=[], launch_binding_mismatches=[], state=inspected['State'],
        container_id=launch['container_id'], image=IMAGE, remote_root=launch['remote_root'],
        archive_sha256=sha(archive), files={str(p.relative_to(root)): sha(p)
            for p in root.rglob('*') if p.is_file()}))


def envelope(directory, ops, initial):
    """Create synthetic terminal Docker records, checked by real check_runtime."""
    directory.mkdir(parents=True, exist_ok=True)
    root = directory/'collection'
    root.mkdir(exist_ok=True)
    package = Path(ops['package'])
    for name in ('config.json', 'service-config.json', 'PREPARED.json'):
        shutil.copy2(package/name, root/name)
    m, config, _ = validate(ops)
    argv = command(ops, m, config)
    def values(flag):
        return [argv[i+1] for i, v in enumerate(argv) if v == flag]
    cid = ('b' if config['start_update'] else 'a') * 64
    launch = dict(container_id=cid, image=IMAGE, argv=argv, remote_root=m['remote_root'],
        manifest_sha256=sha(package/'PREPARED.json'), config_sha256=sha(package/'config.json'),
        service_config_sha256=sha(package/'service-config.json'))
    inspected = dict(Id=cid, Image=IMAGE, State=dict(Status='exited', ExitCode=0, OOMKilled=False),
        Config=dict(Entrypoint=values('--entrypoint'), Cmd=argv[argv.index(IMAGE)+1:], Env=values('-e')),
        HostConfig=dict(Privileged=False, NetworkMode='none', Binds=values('-v'),
            Devices=[dict(PathOnHost=v.split(':')[0], PathInContainer=v.split(':')[1],
                          CgroupPermissions='rwm') for v in values('--device')]))
    c = dict(collection=str(root), initial_policy=str(initial), release='synthetic-release',
             schedule='synthetic-schedule')
    for key in ('operations_config', 'operations_local', 'launch', 'inspect', 'transport'):
        c[key] = str(directory/(key+'.json'))
    write(c['operations_config'], ops)
    write(c['operations_local'], dict(schema='jev-local-macros-v1', macros={}))
    write(c['launch'], launch)
    write(c['inspect'], inspected)
    return c


@pytest.fixture(scope='module')
def archives(tmp_path_factory):
    root = tmp_path_factory.mktemp('dual-archive-synthetic')
    with pytest.MonkeyPatch.context() as mp:
        import multimodal_jev.local_runtime_storage as storage
        import multimodal_jev.slurp_official_training as dataset
        import multimodal_jev.context_training_schedule as scheduling
        mp.setattr(storage, 'check_paths', lambda _: [])
        model, _, _, contract = policy_fixture(root, mp)
        # Same number of trainable keys as Qwen's 28 q/v LoRA layers, tiny shapes.
        layer = model.backbone.language_model.layers[0]
        model.backbone.language_model.layers.extend([copy.deepcopy(layer) for _ in range(27)])
        contract['lora'] = lora_contract(model.backbone)
        initial = root/'initial'
        initial_id = export_policy(initial, model, root/'processor', contract)
        candidates = [dict(id=f'c{i:02d}', description=f'Synthetic event {i}') for i in range(60)]
        task = dict(schema=dict(candidates=candidates), thinking=dict(requirements=['Use audio.']))
        rows = [dict(id=f'row{i}', audio=f'synthetic/{i}.wav', intent=candidates[i % 60]['id'])
                for i in range(40513)]
        batches = [dict(epoch=0, index=i,
            slots=[dict(id=rows[(i*128+j) % len(rows)]['id'], padding=False, weight=1.) for j in range(128)],
            rank_indices=[list(range(rank, 128, 4)) for rank in range(4)]) for i in range(634)]
        # Only ingress is mocked; per-slot target compilation still executes.
        mp.setattr(dataset, 'load', lambda *a, **kw: (rows, task, {}))
        mp.setattr(scheduling, 'load_schedule', lambda *a, **kw: (batches, {}))
        ops = operations_fixture(root)
        package = Path(ops['package'])
        config = json.loads((package/'config.json').read_text())
        config.update(base_id=contract['base_id'], source_policy_id=initial_id, source_contract=contract,
            runtime_profile='synthetic-bf16', release_sha256='1'*64, schedule_sha256='2'*64,
            behavior_bounds={}, stage='task', initial_adapter='/inputs/initial/adapter',
            reference_adapter='/inputs/initial/adapter')
        write(package/'config.json', config)
        (package/'run.py').write_text("if c['start_update']!=0 or c['stop_update']!=2:\n raise ValueError('cursor')\n")
        reseal_package(ops)
        c = envelope(root/'prefix', ops, initial)
        collection = Path(c['collection']); run = collection/'run'
        write(collection/'supervision/terminal.json', dict(passed=True, actor_started=True,
            forced_cleanup=False, actor_exit_code=0, service_exit_code=0, child_exit_codes=[0, 0]))
        write(collection/'service/terminal.json', dict(exit_code=0))
        write(run/'result.json', dict(passed=True, start_update=0, stop_update=2,
            schedule_updates=634, learning_population=40513, engineering_prefix=True))
        ready = dict(policy_id=initial_id, contract=contract, max_batch_queries=4,
            max_batch_tokens=16384, physical_devices=[7], transport_protocol='bulk_dual_logits_candidate_v1')
        write(collection/'service/ready.json', ready)
        train_contract = expected_contract(config)
        # Deterministic CPU Adam on explicit rank slices, including empty owners.
        shards = [[] for _ in range(4)]
        for name, p in model.named_parameters():
            if p.requires_grad:
                for rank, part in enumerate(torch.tensor_split(p.detach().float().flatten(), 4)):
                    shards[rank].append(('_fsdp_wrapped_module.'+name, torch.nn.Parameter(part.clone())))
        optimizers = [torch.optim.AdamW([p for _, p in named], lr=1e-4, betas=(.9,.999),
            eps=1e-8, weight_decay=0., foreach=False, fused=False) for named in shards]
        current = initial_id
        for index in range(2):
            step = index+1; planned = batches[index]; records=[]; queries=[]; groups=[]
            for i, slot in enumerate(planned['slots']):
                row = rows[index*128+i]; uid=f'{row["id"]}::epoch0-batch{index}-slot{i}'
                request = compile_candidates(row['audio'], [dict(id=v['id'], definition=v['description'])
                    for v in candidates], task['thinking']['requirements'], semantics='exclusive_event')
                target = bind_candidate_targets(request, semantics='exclusive_event', exclusive_event=row['intent'])
                records.append(dict(id=row['id'], request_id=uid, weight=1., target=target))
                ident=dict(policy_id=current, inputs={'synthetic_audio': row['audio']}, positions=list(range(60)))
                z=torch.linspace(-.7,.7,60); probability=z.sigmoid()
                raw=dict(policy_id=current, request_identity=ident, request_id=digest(ident),
                    prob=probability.tolist(), logits=z.tolist(), output_protocol=WIRE)
                ref_ident=dict(ident, policy_id=initial_id)
                reference=dict(policy_id=initial_id, request_identity=ref_ident, request_id=digest(ref_ident), prob=[.5]*60)
                identity=dict(policy_id=current, input_id=raw['request_id'], query_id=uid, update=index)
                labels=torch.tensor(target['labels'])
                credit=prepare(z, probability, [v['id'] for v in candidates], labels,
                    **identity, algorithm=config['algorithm'])
                queries.append(dict(request_id=uid, behavior_record=raw, reference_record=reference,
                    credit_binding_sha256=credit['binding_sha256']))
                groups.append(dict(request_id=uid, policy_id=current, candidate_ids=[v['id'] for v in candidates],
                    labels=labels[None], observed=torch.ones_like(labels)[None], real_request=True, request_weight=1.,
                    global_rows=128, global_real_requests=128, dual_options=train_contract['dual_objective'],
                    reference_policy_id=initial_id, reference_probability=torch.full((60,),.5),
                    dual_credit=credit, dual_identity=identity))
            receipt=dict(runtime_identity=config['dual_runtime_identity'], behavior_policy_id=current,
                fixed_reference_policy_id=initial_id, update=index, queries=queries)
            write(run/f'rollout-{index:04d}.json', receipt)
            write(run/f'batch-{index:04d}.json', dict(policy_id=current, rank_indices=planned['rank_indices'], records=records))
            save(run/f'groups-{index:04d}.pt', [groups[i] for rank in planned['rank_indices'] for i in rank])
            shard_hashes={}
            for rank, (named, opt) in enumerate(zip(shards, optimizers)):
                for _, p in named:
                    if p.numel(): p.grad=torch.full_like(p,.1)
                opt.step();opt.zero_grad(set_to_none=True)
                state=capture(named,opt,rank=rank,world_size=4,completed_updates=step,contract=train_contract)
                name=f'rank-{rank}/checkpoint-{step:04d}.pt';save(run/name,state);shard_hashes[name]=sha(run/name)
                write(run/f'rank-{rank}/checkpoint-scope.json',validate_trainable_scope(named))
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        p.copy_(torch.cat([dict(s)['_fsdp_wrapped_module.'+name].detach() for s in shards]).reshape(p.shape))
            execution={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
            assert len(execution)==128
            save(run/f'execution-{step:04d}.pt',execution)
            current=export_policy(run/f'policy-{step:04d}',model,root/'processor',contract)
            bundle=read_execution_bundle(run/f'policy-{step:04d}',expected_policy_id=current,
                expected_contract=contract,expected_base_id=contract['base_id'])
            export_native_adapter(run/f'adapter-{step:04d}',bundle)
            write(run/f'checkpoint-{step:04d}-committed.json',dict(updates=step,contract=train_contract,
                rank_shards=shard_hashes,execution_file=f'execution-{step:04d}.pt',execution_sha256=sha(run/f'execution-{step:04d}.pt'),
                joint_policy_directory=f'policy-{step:04d}',joint_policy=dict(policy_id=current),
                joint_policy_manifest_sha256=sha(run/f'policy-{step:04d}/MANIFEST.json')))
            write(run/f'activation-{step:04d}.json',dict(error=None,policy_id=current))
            for rank in range(4):
                write(run/f'rank-{rank}/update-{step:04d}.json',dict(update=step,real_queries=128,
                    next_policy_id=current,rollout=receipt,checkpoint=True,total_seconds=10.+rank,
                    behavior_parity_history=[[dict(passed=True,error=None) for _ in range(4)] for _ in range(4)]))
        for rank in range(4):
            write(run/f'rank-{rank}/result.json',dict(passed=True,start_update=0,stop_update=2,
                executions={str(i):'bfloat16' for i in range(113)}))
        transport(c)
        yield root, c, ops


def test_complete_prefix_and_recursive_replay_archive(archives):
    root,c,ops=archives
    prefix=audit(c)
    assert prefix['passed'] and not prefix['fresh_process_replay_verified']
    assert [u['checked_slots'] for u in prefix['updates']]==[128,128]
    assert all(u['master_export']['parameters']==128 for u in prefix['updates'])
    output=root/'replay-package'
    seal(dict(output=str(output),remote_root='/run/replay'),{},prefix,c)
    resumed_ops=dict(ops,package=str(output),manifest_sha256=sha(output/'PREPARED.json'))
    replay=envelope(root/'replay',resumed_ops,c['initial_policy'])
    collection=Path(replay['collection'])
    for directory in ('run','service','supervision'):
        shutil.copytree(Path(c['collection'])/directory,collection/directory)
    ready=json.loads((collection/'service/ready.json').read_text())
    ready['policy_id']=json.loads((output/'PREPARED.json').read_text())['replay']['initial_behavior_policy_id']
    write(collection/'service/ready.json',ready)
    for name in ['run/result.json',*[f'run/rank-{r}/result.json' for r in range(4)]]:
        value=json.loads((collection/name).read_text());value['start_update']=1;write(collection/name,value)
    for rank in range(4):
        write(collection/f'run/rank-{rank}/restored.json',dict(exact_masters=True,exact_adam=True,completed_updates=1))
    # New replay saves only step2; remove original step0/step1 artifacts.
    for p in sorted((collection/'run').rglob('*'),key=lambda p:len(p.parts),reverse=True):
        if p.name in ('rollout-0000.json','batch-0000.json','groups-0000.pt','execution-0001.pt',
                      'checkpoint-0001-committed.json','activation-0001.json','update-0001.json','checkpoint-0001.pt'):
            p.unlink()
    for name in ('policy-0001','adapter-0001'):shutil.rmtree(collection/'run'/name)
    write(root/'parent-audit.json',c)
    replay.update(parent_audit_config=str(root/'parent-audit.json'),parent_audit_local=c['operations_local'])
    transport(replay)
    result=audit(replay)
    assert result['schema']=='context-dual-replay-audit-v1' and result['fresh_process_replay_verified']
    assert not result['training_dispatch_allowed']


@pytest.mark.parametrize('fault',['reward','master','adam','marker','activation','parity','runtime'])
def test_rehashed_but_contradictory_archive_rejected(archives,tmp_path,fault):
    _,original,_=archives
    c=dict(original,collection=str(tmp_path/'collection'),transport=str(tmp_path/'transport.json'))
    shutil.copytree(original['collection'],c['collection']);run=Path(c['collection'])/'run'
    if fault=='reward':
        p=run/'groups-0000.pt';groups=torch.load(p,weights_only=True)
        reward=groups[0]['dual_credit']['branches']['ranking']['rewards'];reward[0,0]=1-reward[0,0];save(p,groups)
    elif fault in ('master','adam'):
        p=run/'rank-0/checkpoint-0001.pt';state=torch.load(p,weights_only=True)
        if fault=='master':next(v for v in state['parameters'].values() if v.numel())[0]+=.5
        else:next(iter(state['optimizer']['state'].values()))['step']+=1
        save(p,state)
        p=run/'checkpoint-0001-committed.json';value=json.loads(p.read_text())
        value['rank_shards']['rank-0/checkpoint-0001.pt']=sha(run/'rank-0/checkpoint-0001.pt');write(p,value)
    elif fault=='marker':
        p=run/'policy-0001/execution.pt';value=torch.load(p,weights_only=True);value['decision_token_row'][0]+=.5;save(p,value)
    else:
        p=run/('activation-0001.json' if fault=='activation' else 'rank-0/update-0001.json' if fault=='parity' else 'rollout-0000.json')
        value=json.loads(p.read_text())
        if fault=='activation':value['policy_id']='0'*64
        elif fault=='parity':value['behavior_parity_history'][0][0]['passed']=False
        else:value['runtime_identity']={'changed':True}
        write(p,value)
    transport(c)
    messages=dict(reward='ranking/rewards',master='FP32 master shards',adam='Adam update count',
        marker='joint asset integrity',activation='not activated',parity='actual DP4 parity',
        runtime='rollout runtime')
    with pytest.raises(ValueError,match=messages[fault]):audit(c)
