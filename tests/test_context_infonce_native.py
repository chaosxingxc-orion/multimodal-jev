"""Real native verl DP8 update and fresh-process warmup/Adam recovery on CPU."""
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from multimodal_jev.context_training_checkpoint import capture, restore
from multimodal_jev.context_warmup import learning_rate, set_learning_rate
from tools.reports.context_infonce_actor import load


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.z = torch.nn.Parameter(torch.tensor([.3, -.4, .1]))

    def forward(self, n):
        return self.z[None].expand(n, -1)


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): same(x, y)
    else:
        assert a == b


def worker(rank, root, replay, world, microbatch=8, local_batch=64, margin=None):
    from probe_span_native_actor_dp_cpu import config
    from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_device
    root = Path(root)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(root / ('replay' if replay else 'first')).as_uri(),
                            rank=rank, world_size=world, timeout=timedelta(seconds=240))
    try:
        module = load(Path.cwd(), temperature=1.,probability_margin=margin)
        spec = dict(kind='linear_warmup_constant', peak_lr=1e-4, warmup_updates=8)
        model = Model(); wrapped = DDP(model)
        opt = torch.optim.AdamW(wrapped.parameters(), lr=learning_rate(spec, 0, 160),
                               weight_decay=0., foreach=False, fused=False)
        named = list(wrapped.named_parameters())
        contract = dict(scope='synthetic_cpu_native_declared_dp',world_size=world, lr_schedule=spec)
        identity = dict(rank=rank, world_size=world, contract=contract)
        if replay:
            restore(named, opt, torch.load(root / f'{rank}-first.pt', weights_only=True),
                    completed_updates=1, **identity)
        for step in ([1] if replay else [0, 1]):
            # The actual 512-slot tail has 65 real rows and six all-padding ranks.
            real = local_batch*world if step == 0 else 65
            rows = [dict(request_id=f'q{i}', policy_id=f'p{step}',
                         inputs={'input_ids': torch.tensor([[i, 99, 99, 99]])},
                         positions=torch.tensor([[1, 2, 3]]),
                         labels=torch.nn.functional.one_hot(torch.tensor([i % 3]), 3).bool(),
                         observed=torch.ones(1, 3, dtype=torch.bool),
                         real_request=i < real, request_weight=float(i < real)) for i in range(local_batch*world)]
            options = dict(stage='instruction', ranking_coefficient=1., probability_coefficient=0.,
                           algorithm='grpo_v1', kl_coefficient=0., reference_policy_id=None)
            full = module.make_batch(rows, **options)
            batch = list(full.split(local_batch))[rank]
            cfg = config(microbatch, f'p{step}'); cfg.ppo_mini_batch_size = local_batch
            cfg.policy_loss.loss_mode = module.MODE; cfg.jev_context_objective = options
            cfg.jev_context_parity = dict(probability_max_abs=0., binary_logprob_max_abs=0.)
            calls = []
            def forward(actor, inputs, **kw):
                assert all(set(p) == {'request_id', 'policy_id', 'inputs', 'positions'} for p in inputs)
                calls.append(len(inputs)); return actor(len(inputs))
            module.forward_context_audio_batch = forward
            actor = module.build_actor(cfg, wrapped, opt, collation={})
            ref = model.z.detach().clone().requires_grad_()
            expected_loss = torch.nn.functional.cross_entropy(ref[None].expand(real, -1), torch.arange(real) % 3)
            if margin is not None:
                from multimodal_jev.context_infonce import loss
                labels=torch.nn.functional.one_hot(torch.arange(real)%3,3).bool()
                expected_loss,_=loss(ref[None].expand(real,-1),labels,torch.ones_like(labels),torch.ones(real),torch.ones(real,dtype=torch.bool),global_real_requests=real,probability_margin=margin)
            expected, = torch.autograd.grad(expected_loss, ref)
            grads = []; original = actor._optimizer_step
            def update():
                grads.append(model.z.grad.detach().clone()); return original()
            actor._optimizer_step = update
            set_learning_rate(opt, spec, step, 160)
            with native_actor_cpu_device(actor): actor.update_policy(batch)
            assert calls == [microbatch] * (local_batch // microbatch) and len(grads) == 1
            torch.testing.assert_close(grads[0], expected, atol=2e-7, rtol=2e-6)
            state = capture(named, opt, completed_updates=step + 1, **identity)
            if step == 0: torch.save(state, root / f'{rank}-first.pt')
            elif not replay: torch.save(state, root / f'{rank}-expected.pt')
            else: same(state, torch.load(root / f'{rank}-expected.pt', weights_only=True))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('world',[4,8])
def test_native_full_and_tail_update_with_fresh_resume(tmp_path,world):
    pytest.importorskip('verl'); pytest.importorskip('omegaconf')
    assert not list(Path('/dev').glob('davinci[0-9]*'))
    mp.spawn(worker, args=(str(tmp_path), False,world), nprocs=world, join=True)
    mp.spawn(worker, args=(str(tmp_path), True,world), nprocs=world, join=True)


def test_readonly_input_cache_rejects_miss_and_write(tmp_path):
    from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache
    with pytest.raises(ValueError, match='prebuilt'): ReadOnlyEncodedAudioCache(tmp_path / 'missing')
    cache = ReadOnlyEncodedAudioCache(tmp_path)
    with pytest.raises(ValueError, match='miss'): cache.read({'request': 'absent'})
    with pytest.raises(ValueError, match='read-only'): cache.write(None)


def test_native_micro64_full_and_tail_with_fresh_resume(tmp_path):
    pytest.importorskip('verl'); pytest.importorskip('omegaconf')
    assert not list(Path('/dev').glob('davinci[0-9]*'))
    mp.spawn(worker, args=(str(tmp_path), False,4,64), nprocs=4, join=True)
    mp.spawn(worker, args=(str(tmp_path), True,4,64), nprocs=4, join=True)


def test_native_local128_micro32_full_tail_resume(tmp_path):
    pytest.importorskip('verl'); pytest.importorskip('omegaconf')
    mp.spawn(worker,args=(str(tmp_path),False,4,32,128),nprocs=4,join=True)
    mp.spawn(worker,args=(str(tmp_path),True,4,32,128),nprocs=4,join=True)


def test_native_margin_full_tail_and_fresh_resume(tmp_path):
    pytest.importorskip('verl'); pytest.importorskip('omegaconf')
    margin=dict(margin=.1,decision_threshold=.5,gap_weight=1.,anchor_weight=.25)
    mp.spawn(worker,args=(str(tmp_path),False,4,8,64,margin),nprocs=4,join=True)
    mp.spawn(worker,args=(str(tmp_path),True,4,8,64,margin),nprocs=4,join=True)
