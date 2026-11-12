#!/usr/bin/env python3
"""Resident native verl actor versus mid-run model/Adam reload, four CPU ranks.

Synthetic FP32 control only. No NPU, actual audio, BF16, vLLM or Ray claim.
"""
import argparse
import copy
from datetime import timedelta
import json
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from multimodal_jev.baseline import atomic_json, file_hash
from probe_span_native_actor_dp_cpu import Model, config, optimizer, snapshot
from probe_span_finite_actor_dp_cpu import make_items, backward_reference


def worker(rank, output):
    from multimodal_jev.span_finite_actor import make_global_batch, build_actor, admit_batch
    from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_device
    from multimodal_jev.span_training_session import SpanTrainingSession
    output = Path(output); torch.set_num_threads(1); torch.manual_seed(317)
    if list(Path('/dev').glob('davinci[0-9]*')): raise ValueError('CPU only')
    dist.init_process_group('gloo', init_method=(output/'gloo-init').resolve().as_uri(),
        rank=rank, world_size=4, timeout=timedelta(seconds=90))
    try:
        def gather(value):
            peers=[None]*4;dist.all_gather_object(peers,value);return peers
        initial = copy.deepcopy(Model().state_dict())
        reference = Model(); reference.load_state_dict(initial); refopt = optimizer(reference)
        branches = {}
        for key in ('resident', 'resumed'):
            module = Model(); module.load_state_dict(initial); model = DDP(module); opt = optimizer(model)
            cfg = config(1, 'policy0'); cfg.ppo_mini_batch_size = 2
            actor=build_actor(cfg,model,opt)
            branches[key] = dict(model=model, optimizer=opt, actor=actor,
                session=SpanTrainingSession(actor,policy_id='policy0',completed_updates=0,gather=gather))
        cases = []; restored = False
        for step in range(8):
            if step == 4:
                prior = branches['resumed']; path = output/f'rank-{rank}-step4.pt'
                torch.save(dict(model=prior['model'].module.state_dict(), optimizer=prior['optimizer'].state_dict(),
                                completed_updates=4, next_policy='policy4', rank=rank, world_size=4), path)
                import hashlib
                checkpoint_id=hashlib.sha256(json.dumps(gather(file_hash(path))).encode()).hexdigest()
                cursor=prior['session'].checkpoint_cursor(schedule_sha256='synthetic-eight-update-seed918-v1',checkpoint_id=checkpoint_id)
                atomic_json(output/f'rank-{rank}-session-step4.json',cursor)
                saved = torch.load(path, map_location='cpu', weights_only=True)
                assert saved['rank'] == rank and saved['world_size'] == 4 and saved['completed_updates'] == step
                module = Model(); module.load_state_dict(saved['model']); model = DDP(module); opt = optimizer(model)
                opt.load_state_dict(saved['optimizer']); cfg = config(1, saved['next_policy']); cfg.ppo_mini_batch_size = 2
                actor=build_actor(cfg,model,opt)
                branches['resumed'] = dict(model=model, optimizer=opt, actor=actor,
                    session=SpanTrainingSession.restore(actor,cursor,schedule_sha256='synthetic-eight-update-seed918-v1',checkpoint_id=checkpoint_id,gather=gather))
                for old, new in zip(prior['model'].parameters(), model.parameters(), strict=True):
                    assert torch.equal(old, new)
                    for field in ('step', 'exp_avg', 'exp_avg_sq'):
                        assert torch.equal(prior['optimizer'].state[old][field], opt.state[new][field])
                del prior; restored = True
            items = make_items(reference, step)
            chunks = list(make_global_batch(items).split(2)) if rank == 0 else None
            incoming = [None]; dist.scatter_object_list(incoming, chunks, src=0); batch = incoming[0]
            refopt.zero_grad(); backward_reference(reference, items)
            gradients = [p.grad.clone() for p in reference.parameters()]
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.); refopt.step()
            expected = snapshot(reference, refopt, gradients); measured = {}; errors = {}; stale_rejections = []
            for key, branch in branches.items():
                model, opt, actor = (branch[k] for k in ('model', 'optimizer', 'actor'))
                session=branch['session']
                assert session.policy_id==f'policy{step}' and session.completed_updates==step
                if step:
                    stale=copy.deepcopy(batch)
                    for payload in stale.non_tensor_batch['multi_modal_inputs']:payload['policy_id']=f'policy{step-1}'
                    try:admit_batch(stale,actor.config)
                    except ValueError:stale_rejections.append(key)
                    else:raise AssertionError('stale behavior accepted')
                captured = {}; original = actor._optimizer_step
                def capture(original=original, captured=captured, model=model):
                    captured['gradients'] = [p.grad.detach().clone() for p in model.parameters()]
                    return original()
                actor._optimizer_step = capture
                old = batch.batch['old_log_probs'].clone()
                try:
                    with native_actor_cpu_device(actor):
                        computed = actor.compute_log_prob(batch)['log_probs']
                        torch.testing.assert_close(computed, old, atol=2e-7, rtol=2e-5)
                        metrics = session.update(batch)
                finally: actor._optimizer_step = original
                assert torch.equal(old, batch.batch['old_log_probs'])
                assert all(opt.state[p]['step'].item() == step+1 for p in model.parameters())
                actual = snapshot(model, opt, captured['gradients']); measured[key] = actual; errors[key] = {}
                for field, values in expected.items():
                    errors[key][field] = max(float((a-b).abs().max()) for a, b in zip(values, actual[field], strict=True))
                    for a, b in zip(values, actual[field], strict=True):
                        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-5)
                try:session.update(batch)
                except ValueError:pass
                else:raise AssertionError('unaccepted second update permitted')
                session.publish(f'policy{step+1}')
                # Independent numerical reference check; synthetic CPU only.
                def verify(policy,updates):
                    for field,values in expected.items():
                        for a,b in zip(values,actual[field],strict=True):torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-5)
                    assert policy==f'policy{step+1}' and updates==step+1
                    return dict(passed=True,policy_id=policy,updates=updates)
                session.accept(verify)
            exact = {}
            for field in expected:
                exact[field] = all(torch.equal(a, b) for a, b in zip(measured['resident'][field], measured['resumed'][field], strict=True))
                if not exact[field]: raise AssertionError('resume diverged from resident: '+field)
            cases.append(dict(update=step+1, errors_against_reference=errors, resident_resume_exact=exact,
                              stale_policy_rejected_branches=stale_rejections))
        # Real collective negative control: each rank's receipt names different
        # evidence despite agreeing on policy and cursor. No optimizer involved.
        from types import SimpleNamespace
        dummy = SimpleNamespace(config=SimpleNamespace(jev_policy_id='receipt0'), update_policy=lambda batch: {})
        receipt_session = SpanTrainingSession(dummy, policy_id='receipt0', completed_updates=0, gather=gather)
        receipt_session.update(None); receipt_session.publish('receipt1')
        try:
            receipt_session.accept(lambda p,n:dict(passed=True,policy_id=p,updates=n,gate_hash='rank-'+str(rank)))
        except ValueError: pass
        else: raise AssertionError('different rank evidence accepted')
        assert receipt_session.phase == 'failed' and dummy.config.jev_policy_id == 'receipt0'
        atomic_json(output/f'rank-{rank}.json', dict(passed=True, rank=rank, restored_step4=restored,
            divergent_receipt_rejected=True, cases=cases))
    except BaseException as exc:
        atomic_json(output/f'rank-{rank}-failure.json', dict(error=repr(exc))); raise
    finally: dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    try:
        mp.spawn(worker, args=(str(a.output),), nprocs=4, join=True)
        ranks = [json.loads((a.output/f'rank-{i}.json').read_text()) for i in range(4)]
        if not all(r['passed'] for r in ranks): raise ValueError('incomplete ranks')
        atomic_json(a.output/'result.json', dict(passed=True, ranks=ranks, world_size=4, updates=8,
            scope='Reusable session with synthetic FP32 four-Gloo-rank resident native verl actor and exact model/Adam/cursor reload. '
                  'Unequal request weights and fully masked rank retained. No audio/NPU/BF16/FSDP/vLLM/Ray validation.'))
        atomic_json(a.output/'terminal.json', {'exit_code': 0})
    except BaseException as exc:
        atomic_json(a.output/'terminal.json', dict(exit_code=1, error=repr(exc))); raise
