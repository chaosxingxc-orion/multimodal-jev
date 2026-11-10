"""Static effective-length budgets; never estimates of FLOPs or realized RL work."""
from .training_budget import batch_budget, sum_budgets


def schedule_profile(counts, batches, *, samples=1, max_response_tokens=None,
                     no_grad_passes=0):
    if type(samples) is not int or samples < 1:
        raise ValueError('positive sample count required')
    if max_response_tokens is not None and (
            type(max_response_tokens) is not int or max_response_tokens < 1):
        raise ValueError('positive response cap required')
    if not batches:
        raise ValueError('nonempty schedule required')
    budgets = []
    for ids in batches:
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('nonempty unique IDs per batch required')
        rows = [counts[uid] for uid in ids for _ in range(samples)]
        budgets.append(batch_budget(
            [r['prompt_tokens'] for r in rows],
            [r['gold_tokens'] if max_response_tokens is None else max_response_tokens for r in rows],
            [r['audio_frames'] for r in rows], gradient_passes=1,
            no_grad_passes=no_grad_passes, optimizer_steps=1))
    return {'total': sum_budgets(budgets),
            'largest_batch_by_field': {k: max(b[k] for b in budgets) for k in budgets[0]},
            'batches': budgets,
            'interpretation': ('exact effective lengths if every scheduled SFT update executes'
                if max_response_tokens is None else
                'upper bounds assuming every response reaches cap and every batch updates; '
                'generation work excluded; zero-advantage batches can skip gradients')}
