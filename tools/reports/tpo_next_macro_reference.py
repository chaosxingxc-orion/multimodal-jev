"""Produce just the frozen reward difference, and check against the paired arm."""
import json
from collections import Counter
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_tpo_rewards import RewardSpec, compile_targets
from multimodal_jev.local_checkpoint import commit_file, commit_json
from tools.reports.build_tpo_view import verified


def prepare(c, entry, data, m, parent_sha, world, repo):
    read = lambda path: json.loads(path.read_bytes())
    design = read(repo / c['design'])
    verified(data/'MANIFEST.json', design['parent_view_sha256'])
    if (m['schema'] != 'jev-tpo-choices-v1' or m['files']['batches.json'] != design['schedule_file_sha256']
            or m['updates'] != design['endpoint_update'] or design['scope'] != 'KL_prior_only'
            or design['reference_interval_optimizer_updates'] != 1 or design['reference_cache_scope'] != 'one_next_macro_batch'):
        raise ValueError('not the registered paired schedule')
    for name, h in m['files'].items():
        verified(data/name, h)
    ref = Path(c['native_reference'])
    native = read(verified(ref/'MANIFEST.json', m['reference_sha256']))
    records = read(verified(ref/'records.json', native['files']['records.json']))
    rows = read(data/'rows.json')
    ordered = sorted(rows.values(), key=lambda r:r['index'])
    if [r['index'] for r in ordered] != list(range(len(rows))):
        raise ValueError('ambiguous row indices')
    ps = torch.tensor([records[r['request_key']]['prob'] for r in ordered])
    gold = torch.tensor([r['gold'] for r in ordered])
    counts = Counter(gold.tolist())
    utility = torch.tensor([len(rows)/(len(counts)*counts[g]) for g in gold.tolist()])
    targets = torch.load(data/'targets.pt', weights_only=True, map_location='cpu')
    shift = torch.empty_like(targets, dtype=torch.float64)
    torch.set_num_threads(c['cpu_threads'])
    reconstructed, audit = compile_targets(ps, rows, gold, utility, read(data/'batches.json'),
                                          RewardSpec(**m['rewards']), reward_shift=shift)
    if (not torch.equal(reconstructed, targets) or audit != m['reward_audit']
            or not torch.equal((ps.double().logit()+shift).float(), targets)):
        raise ValueError('frozen rewards do not reproduce fixed-reference targets')
    receipt = read(Path(c['control_receipt']))
    if (receipt['target_sha'] != design['parent_view_sha256'] or receipt['parent_sha'] != parent_sha
            or receipt['world_size'] != world or receipt['passed'] is not True or receipt['window'] != 'reference'):
        raise ValueError('unbound control reference')
    commit_file(entry/'reward-shift.pt', lambda f:torch.save(shift, f))
    commit_json(entry/'next-macro.json', dict(mode='next_macro',reward_shift=file_hash(entry/'reward-shift.pt'),
        control_reference=receipt, target_sha=design['parent_view_sha256'],parent_sha=parent_sha,
        exact_target_elements=targets.numel(),real_exposures=audit['real_exposures']))
