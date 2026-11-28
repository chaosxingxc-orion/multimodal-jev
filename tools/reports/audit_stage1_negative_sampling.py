"""Compare committed EMA distributions with the sealed source TRAIN scores.

The result measures snapshot-based selection probabilities, not historical exposure.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from multimodal_jev.speech_alignment_pairs import Partners


def read(path):
    return json.loads(path.read_bytes())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def hit(mass, draws):
    return 1.0 - (1.0 - mass) ** draws


def export_ema(run_root, steps):
    # Run on the checkpoint host with the frozen training source on PYTHONPATH.
    import torch
    from multimodal_jev.class_logit_ema import ClassLogitEMA

    snapshots = {}
    for step in steps:
        commit_raw = (run_root / f'checkpoint-{step:04d}-committed.json').read_bytes()
        commit = json.loads(commit_raw)
        witness = commit['sampler_state']
        relative = f'rank-0/checkpoint-{step:04d}.pt'
        path = run_root / relative
        if commit['updates'] != step or sha(path.read_bytes()) != commit['rank_shards'][relative]:
            raise ValueError('uncommitted EMA checkpoint')
        state = torch.load(path, map_location='cpu', weights_only=True)
        ema = ClassLogitEMA(witness['labels'], **witness['config'])
        ema.load_state_dict(state['auxiliary']['class_logit_ema'])
        if ema.updates != step or ema.receipt(witness['config']) != witness:
            raise ValueError('EMA state differs from committed witness')
        snapshots[step] = dict(witness=witness, q=ema.probabilities().tolist(), commit_sha256=sha(commit_raw))
    return snapshots


def load_panel(root, endpoint):
    audit_raw = (root / 'audit.json').read_bytes()
    audit = json.loads(audit_raw)
    release = read(endpoint / 'receipt.json')
    score = root / 'collected' / 'score'
    receipt = read(score / 'receipt.json')
    raw = (root / 'entry' / 'full.json').read_bytes()
    if (not audit['passed'] or not release['passed']
            or (release['updates'], release['checkpoint_sha256'], release['policy_id'], release['manifest_sha256'])
                != (audit['updates'], audit['checkpoint_sha256'], audit['policy_id'], audit['policy_sha256'])
            or sha(raw) != audit['full_sha256']
            or any(audit[k] != receipt[k] for k in ('full_sha256', 'policy_id', 'policy_sha256', 'admission_sha256'))):
        raise ValueError('scores are not bound to the audited Stage1 endpoint')
    full = json.loads(raw)
    labels = full['labels']
    population = {r['id']: r for r in sorted(full['rows'], key=lambda r: r['id'])}
    ids = {uid: i for i, uid in enumerate(population)}
    y = np.asarray([labels.index(row['intent']) for row in population.values()])
    p = np.full((len(ids), len(labels)), np.nan)
    seen = set()
    for i, worker in enumerate(receipt['workers']):
        for block in worker['blocks']:
            data = (score / f'worker-{i}' / block['name']).read_bytes()
            if sha(data) != block['sha256']:
                raise ValueError('score block changed')
            records = json.loads(data)['payload']['records']
            rows = {r['id']: r['prob'] for r in records}
            if len(rows) != len(records) or seen.intersection(rows):
                raise ValueError('duplicate scored ID')
            idx = [ids[uid] for uid in rows]
            p[idx] = list(rows.values())
            seen.update(rows)
    if (not (0 < p).all() or not (p < 1).all() or len(ids) != audit['recordings']
            or p.size != audit['pairs'] or len(ids) != len(full['rows']) or labels != audit['labels']):
        raise ValueError('incomplete or invalid TRAIN score panel')
    return audit, sha(audit_raw), release, labels, population, y, p


def audit_scores(score_root, stage1_root, ema_path, *, top_k):
    endpoint = stage1_root / 'endpoint'
    audit, audit_sha, release, labels, population, y, p = load_panel(score_root, endpoint)
    config = read(endpoint / 'training.json')
    objective = config['instruction_objective']
    if not objective['sample_with_replacement']:
        raise ValueError('coverage calculation assumes independent draws with replacement')
    cards, recordings = (objective['negative_cards_per_anchor'], objective['negative_recordings_per_anchor'])
    margin = objective['speech']['probability_margin']
    if not 1 <= top_k < len(labels):
        raise ValueError('top_k must fit the non-gold cards')

    # Resolve ties by frozen label order, and by sorted recording ID for audio.
    # Also calculate a tie-aware top-1 separately.
    i = np.arange(len(y))
    truth = p[i, y].copy()
    p[i, y] = -np.inf
    top = np.argsort(-p, axis=1, kind='stable')[:, :top_k]
    rival_score = p[i, top[:, 0]]
    tied_max = p == rival_score[:, None]
    violation = p > (truth[:, None] - margin)
    wrong = rival_score > truth
    ties = rival_score == truth
    gap_failed = rival_score > truth - margin
    audio_rank = [np.argsort(-p[:, c], kind='stable') for c in range(len(labels))]
    audio_top = np.stack([order[:top_k] for order in audio_rank])
    if (y[audio_top] == np.arange(len(labels))[:, None]).any() or not np.isfinite(
            p[audio_top, np.arange(len(labels))[:, None]]).all():
        raise ValueError('insufficient wrong-class recordings')

    view_path = stage1_root / 'package' / 'entry' / 'speech-partners.json'
    view_raw = view_path.read_bytes()
    if sha(view_raw) != config['speech_partners']['sha256']:
        raise ValueError('wrong semantic controls')
    partners = Partners(json.loads(view_raw), population)
    ids = {uid: j for j, uid in enumerate(population)}
    seed = config['mixed']['seed']
    # Draw these score-blind controls exactly as the current implementation.
    controls = [np.asarray([ids[n] if n is not None else -1 for _, _, n in
                (partners.draws(uid, epoch, seed) for uid in population)], dtype=np.int64)
                for epoch in range(len(release['epoch_end_updates']))]

    scopes = {'all': np.ones(len(y), dtype=bool), 'wrong': wrong, 'wrong_or_tie': wrong | ties,
              'margin_failed': gap_failed,
              'real': np.asarray([not row['synthetic'] for row in population.values()])}
    rank = p[i[:, None], top]
    sizes = np.bincount(y, minlength=len(labels))
    rows = {}
    uniform = {name: float(hit(mass, cards)) for name, mass in
               [('top1_card', 1 / (len(labels)-1)), ('top_k_cards', top_k / (len(labels)-1))]}
    snapshots = read(ema_path)
    if set(map(int, snapshots)) != set(release['epoch_end_updates']):
        raise ValueError('EMA snapshots must match the audited epoch boundaries')
    for epoch, step in enumerate(release['epoch_end_updates']):
        info = snapshots[str(step)]
        witness = info['witness']
        q = np.asarray(info['q'], dtype='<f8')
        if (witness['labels'] != labels or witness['updates'] != step
                or witness['config'] != config['class_logit_ema']
                or q.shape != (len(labels), len(labels)) or not np.isfinite(q).all()
                or (q < 0).any() or not np.allclose(q.sum(axis=1), 1.)
                or np.any(np.diag(q) != 0) or sha(q.tobytes()) != witness['distribution_sha256']
                or (step == release['updates'] and (witness != release['sampler_state']
                    or info['commit_sha256'] != release['checkpoint_sha256']))):
            raise ValueError('EMA distribution does not match its recorded identity')
        qy = q[y]
        audio_margin_mass = np.zeros(len(y))
        for c, order in enumerate(audio_rank):
            anchors = y == c
            negatives = order[:len(y) - sizes[c]]
            scores = p[negatives, c]
            weights = q[c, y[negatives]] / sizes[y[negatives]]
            prefix = np.concatenate(([0.], weights.cumsum()))
            # Strictly greater is required to violate the hinge margin.
            counts = np.searchsorted(-scores, -(truth[anchors] - margin), side='left')
            audio_margin_mass[anchors] = np.clip(prefix[counts], 0., 1.)
        masses = {
            'top1_card': qy[i, top[:, 0]],
            'top1_card_with_ties': (qy * tied_max).sum(1),
            'top_k_cards': qy[i[:, None], top].sum(1),
            'margin_violating_card': (qy * violation).sum(1),
            'top1_fixed_card_audio': (q[np.arange(len(labels)), y[audio_top[:, 0]]] / sizes[y[audio_top[:, 0]]])[y],
            'top_k_fixed_card_audio': (q[np.arange(len(labels))[:, None], y[audio_top]] / sizes[y[audio_top]]).sum(1)[y],
            'margin_violating_fixed_card_audio': audio_margin_mass,
        }
        control = controls[epoch]
        has = control >= 0
        safe = np.maximum(control, 0)
        card = y[safe]
        forced = {
            'top1_card': top[:, 0] == card,
            'top1_card_with_ties': tied_max[i, card],
            'top_k_cards': (top == card[:, None]).any(1),
            'margin_violating_card': violation[i, card],
            'top1_fixed_card_audio': audio_top[y, 0] == control,
            'top_k_fixed_card_audio': (audio_top[y] == control[:, None]).any(1),
            'margin_violating_fixed_card_audio': p[safe, y] > truth - margin,
        }
        metrics = {}
        for name, mass in masses.items():
            draws = recordings if name.endswith('_audio') else cards
            replaced = np.where(has, np.where(forced[name], 1., hit(mass, draws - 1)), hit(mass, draws))
            metrics[name] = {
                scope: {'k_base': float(hit(mass[mask], draws).mean()),
                        'with_semantic_control': float(replaced[mask].mean())}
                for scope, mask in scopes.items() if mask.any()
            }
        rows[str(step)] = dict(
            semantic_control_epoch_index=epoch, distribution_sha256=witness['distribution_sha256'],
            commit_sha256=info['commit_sha256'], controls=int(has.sum()),
            controls_on_wrong=int((has & wrong).sum()),
            median_class_top_sampling_probability=float(np.median(q.max(1))),
            maximum_class_sampling_probability=float(q.max()),
            repeated_card_probability=(3 * (q ** 2).sum(1) - 2 * (q ** 3).sum(1))[y].mean().item()
                if cards == 3 else None,
            metrics=metrics,
        )

    # Optimistic, in-sample oracle: the three most frequent Top-1 rivals per
    # gold class maximize coverage among fixed sets of the same size.
    def fixed_card_oracle(mask):
        counts = np.zeros((len(labels), len(labels)), dtype=np.int64)
        np.add.at(counts, (y[mask], top[mask, 0]), 1)
        return float(np.sort(counts, axis=1)[:, -cards:].sum() / mask.sum())

    return dict(
        score_audit_sha256=audit_sha,
        full_sha256=audit['full_sha256'], stage1_endpoint_sha256=sha((endpoint / 'receipt.json').read_bytes()),
        score_updates=audit['updates'], score_wall_seconds=audit['wall_seconds'], cards=cards, recordings=recordings,
        top_k=top_k, margin=margin, ties=int(ties.sum()),
        cardinalities={scope: int(mask.sum()) for scope, mask in scopes.items()},
        ambiguous_top1_wrong_card=int((tied_max.sum(1) > 1).sum()),
        gap_median=float(np.median(truth-rival_score)),
        top1_minus_kth_wrong_median=float(np.median(rank[:, 0]-rank[:, -1])),
        uniform_card_baseline=uniform,
        fixed_class_oracle={scope: fixed_card_oracle(scopes[scope]) for scope in ('all', 'wrong')},
        snapshots=rows,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest='command', required=True)
    collect = command.add_parser('export-ema')
    collect.add_argument('run_root', type=Path)
    collect.add_argument('steps', type=int, nargs='+')
    audit = command.add_parser('audit')
    audit.add_argument('score_root', type=Path)
    audit.add_argument('stage1_root', type=Path)
    audit.add_argument('ema_snapshots', type=Path, help='JSON containing verified checkpoint witnesses and q tables')
    audit.add_argument('--top-k', type=int, required=True)
    args = parser.parse_args()
    result = (export_ema(args.run_root, args.steps) if args.command == 'export-ema'
              else audit_scores(args.score_root, args.stage1_root, args.ema_snapshots, top_k=args.top_k))
    print(json.dumps(result, indent=2, allow_nan=False))
