import contextlib
import copy
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

import pytest
import torch

from tools.jev.prepare_context_checkpoint_runtime import prepare, sha, transform

SOURCE = Path('tools/jev/train_context_online_candidate.py')


def test_source_identity_and_original_guard(tmp_path):
    original = SOURCE.read_bytes()
    result = prepare(source=SOURCE, source_sha256=sha(SOURCE), output=tmp_path / 'candidate')
    changed = (tmp_path / 'candidate/train_context_online_candidate.py').read_text()
    assert SOURCE.read_bytes() == original
    assert result['status'] == 'bounded_checkpoint_candidate_no_dispatch'
    assert "if c['engineering_prefix'] is not True or c['training_dispatch_allowed'] is not False:" in changed
    for a, b in [('                actor=build_actor', '                save_now='),
                 ('                activation=[None]', '                record=dict')]:
        assert changed[changed.index(a):changed.index(b)] == original.decode()[original.decode().index(a):original.decode().index(b)]


def test_changed_original_rejected_without_output(tmp_path):
    with pytest.raises(ValueError):prepare(source=SOURCE, source_sha256='wrong', output=tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
    with pytest.raises(ValueError):transform(SOURCE.read_text().replace('save_now=True', 'save_now=False'))


def publication():
    code = transform(SOURCE.read_text())
    a = code.index('                save_now=')
    b = code.index('                activation=[None]', a)
    return compile(textwrap.dedent(code[a:b]), '<checkpoint-candidate-publication>', 'exec')


def execute_segment(root, start, stop, interval=64, initial=None):
    root.mkdir()
    code = publication()
    models, optimizers = [], []
    for rank in range(4):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():model.weight.fill_(.1)
        opt = torch.optim.AdamW(model.parameters(), lr=.01, foreach=False, fused=False)
        if initial:
            model.load_state_dict(initial['model']);opt.load_state_dict(copy.deepcopy(initial['optimizer']))
        models.append(model);optimizers.append(opt);(root / f'rank-{rank}').mkdir()
    exports, captures = [], []
    def capture(named, opt, **kw):
        rank = kw['rank'];captures.append((kw['completed_updates'], rank))
        return dict(model=copy.deepcopy(models[rank].state_dict()), optimizer=copy.deepcopy(opt.state_dict()), **kw)
    def commit_file(path, writer):
        with path.open('wb') as stream:writer(stream)
    def commit_json(path, value):path.write_text(json.dumps(value))
    def export(source, target, tensors, **kw):
        target.mkdir();torch.save(tensors, target / 'weights.pt')
        pid = sha(target / 'weights.pt')
        (target / 'MANIFEST.json').write_text(json.dumps(dict(policy_id=pid)))
        exports.append(target.name)
        return dict(policy_id=pid)
    def gather(peers, obj):peers[:] = [obj] * 4
    for index in range(start, stop):
        for rank in (3, 2, 1, 0):
            model, opt = models[rank], optimizers[rank]
            opt.zero_grad();(model(torch.ones(1, 2)).sum() - .7).square().backward();opt.step()
            ns = dict(index=index, c=dict(checkpoint_interval=interval, source_policy='fixed-stage1',
                source_policy_id='reference', source_contract={}, base_id='base'), stop_update=stop,
                frozen={}, wrapped=model, model=model, named=list(model.named_parameters()), opt=opt,
                rank=rank, contract={}, root=root, out=root / f'rank-{rank}', torch=torch,
                tensor_identity=lambda t: dict(shape=list(t.shape), values=t.tolist()),
                capture=capture, commit_file=commit_file, commit_json=commit_json, file_hash=sha,
                canonical_execution=lambda t: t, export_checkpoint_policy=export,
                FSDP=SimpleNamespace(summon_full_params=lambda *a, **kw: contextlib.nullcontext()),
                dist=SimpleNamespace(all_gather_object=gather, barrier=lambda: None))
            exec(code, ns)
    return models[0], optimizers[0], exports, captures


def test_every_update_exports_but_only_interval_and_terminal_save(tmp_path):
    _, _, exports, captures = execute_segment(tmp_path / 'run', 0, 66)
    root = tmp_path / 'run'
    assert len(exports) == 66
    assert sorted(set(step for step, rank in captures)) == [64, 66]
    assert len(captures) == 8
    assert not (root / 'execution-0001.pt').exists()
    assert not (root / 'checkpoint-0001-committed.json').exists()
    assert (root / 'policy-0001/MANIFEST.json').exists()
    for step in (1, 63, 64, 65, 66):
        receipt = json.loads((root / f'policy-export-{step:04d}.json').read_text())
        assert receipt['recovery_checkpoint'] == (step in (64, 66))
        assert receipt['policy_manifest_sha256'] == sha(root / f'policy-{step:04d}/MANIFEST.json')
    for step in (64, 66):
        commit = json.loads((root / f'checkpoint-{step:04d}-committed.json').read_text())
        assert len(commit['rank_shards']) == 4
        assert all(sha(root / p) == h for p, h in commit['rank_shards'].items())


def test_saved_optimizer_replay_matches_continuation(tmp_path):
    model, opt, _, _ = execute_segment(tmp_path / 'full', 0, 66)
    saved = torch.load(tmp_path / 'full/rank-0/checkpoint-0064.pt', weights_only=True)
    resumed, resumed_opt, exports, captures = execute_segment(tmp_path / 'resumed', 64, 66, initial=saved)
    assert torch.equal(model.weight, resumed.weight)
    for key, value in opt.state_dict()['state'][0].items():
        assert torch.equal(value, resumed_opt.state_dict()['state'][0][key])
    assert len(exports) == 2 and {step for step, _ in captures} == {66}


def test_last_partial_epoch_saved(tmp_path):
    _, _, exports, captures = execute_segment(tmp_path / 'tail', 632, 634)
    assert exports == ['policy-0633', 'policy-0634']
    assert {step for step, _ in captures} == {634}


def test_interval_change_preserves_every_step_weights_and_terminal_adam(tmp_path):
    left, right = tmp_path / 'every-step', tmp_path / 'interval'
    model_a, opt_a, _, _ = execute_segment(left, 0, 2, interval=1)
    model_b, opt_b, _, _ = execute_segment(right, 0, 2, interval=64)
    assert torch.equal(model_a.weight, model_b.weight)
    for step in (1, 2):
        a = torch.load(left / f'policy-{step:04d}/weights.pt', weights_only=True)
        b = torch.load(right / f'policy-{step:04d}/weights.pt', weights_only=True)
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    for rank in range(4):
        a = torch.load(left / f'rank-{rank}/checkpoint-0002.pt', weights_only=True)
        b = torch.load(right / f'rank-{rank}/checkpoint-0002.pt', weights_only=True)
        assert a['completed_updates'] == b['completed_updates'] == 2
        assert torch.equal(a['model']['weight'], b['model']['weight'])
        assert a['optimizer']['param_groups'] == b['optimizer']['param_groups']
        for key, value in a['optimizer']['state'][0].items():
            assert torch.equal(value, b['optimizer']['state'][0][key])
    assert (left / 'checkpoint-0001-committed.json').exists()
    assert not (right / 'checkpoint-0001-committed.json').exists()
