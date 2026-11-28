"""Replay the Stage1 FSDP scoring forward on the frozen admission panel."""
from collections import Counter
from datetime import timedelta
import os
from pathlib import Path
import time

from multimodal_jev.card_scoring_settings import scoring_settings
from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.card_scoring_benchmark import load_panel, pinned, render


def reference(c, output):
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from multimodal_jev.context_audio_collate import collate_context_audio
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache
    from multimodal_jev.span_runtime_profile import (
        SHORT_AUDIO_PROFILE, configure_actor_profile, initialize_verl_preserving_qwen3,
    )
    from tools.reports.context_transfer_macro_parity import load_native_reference
    from tools.reports.stage2_view_actor import wrap_actor
    from multimodal_jev.context_policy_artifacts import read_execution_bundle
    from multimodal_jev.model_provider import resolve_provider

    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != len(c['devices']) or rank != local or sorted(
            p.name for p in Path('/dev').glob('davinci[0-9]*')) != sorted(
            f'davinci{i}' for i in c['devices']):
        raise ValueError('FSDP reference requires the frozen device topology')
    train = pinned(c['training'])
    panel, catalog = load_panel(c)
    policy = c['formal']['policy_id']
    if (file_hash(Path(c['policy'])/'MANIFEST.json') != c['policy_sha256']
            or checkpoint_identity(train['model'])[0] != train['base_id']):
        raise ValueError('FSDP reference weights changed')
    device = torch.device('npu', local)
    torch.npu.set_device(device)
    torch.manual_seed(c['seed'])
    torch.npu.manual_seed_all(c['seed'])
    initialize_verl_preserving_qwen3()
    configure_actor_profile(SHORT_AUDIO_PROFILE)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(c['torch_threads'])
    dist.init_process_group('hccl', timeout=timedelta(seconds=c['formal']['fsdp_timeout_seconds']))
    try:
        out = output/f'fsdp-{rank}'
        out.mkdir()
        config = dict(model=train['model'], snapshot=c['policy'], policy_id=policy,
                      base_id=train['base_id'], contract=train['source_contract'])
        bundle = read_execution_bundle(c['policy'], expected_policy_id=policy,
                                       expected_contract=config['contract'], expected_base_id=config['base_id'])
        # Same model and native BF16 execution bundle as the trained policy. Use
        # the Stage1 FULL_SHARD forward, including its mixed precision casts.
        model, _, base, _runtime_handle = load_native_reference(config, bundle, out, device)
        provider = resolve_provider(config)
        collate = provider.collation(base, provider.processor(bundle['processor']), model)
        wrapped, _, _ = wrap_actor(model, device, {}, build_optimizer=False)
        wrapped.eval()
        cache = {False: ReadOnlyEncodedAudioCache(train['encoded_audio_cache']),
                 True: ReadOnlyEncodedAudioCache(c['synthetic_cache'])}
        cards = [tuple(torch.tensor(card[k]) for k in ('prefix', 'suffix')) for card in catalog]
        width = scoring_settings(train, c.get('scoring_reference'))['forward_rows']
        rows = panel['rows']
        records, calls = [], Counter()
        start = time.monotonic()
        torch.npu.reset_peak_memory_stats()
        with torch.no_grad():
            for offset in range(0, len(rows), world):
                index = offset+rank
                row = rows[min(index, len(rows)-1)]
                features, _, _ = cache[row['synthetic']].read(row['binding'])
                items = list(render(features, cards, collate, row['id'], policy))
                values = []
                for k in range(0, len(items), width):
                    group = items[k:k+width]
                    inputs, positions = collate_context_audio(group, policy_id=policy, **collate)
                    inputs = {key: val.to(device=device, dtype=torch.bfloat16 if val.is_floating_point()
                                         else val.dtype) for key, val in inputs.items()}
                    values.extend(wrapped(inputs, positions.to(device)).flatten().float().cpu().tolist())
                    if index < len(rows):
                        calls[len(group)] += 1
                if index < len(rows):
                    records.append(dict(id=row['id'], prob=values))
        commit_json(out/'reference.json', dict(policy_id=policy, rank=rank, records=records,
                    batch_sizes=dict(calls), seconds=time.monotonic()-start,
                    peak_bytes=torch.npu.max_memory_allocated()))
    finally:
        dist.destroy_process_group()
