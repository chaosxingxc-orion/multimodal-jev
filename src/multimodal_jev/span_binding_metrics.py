"""Label-free, paired semantic versus positional probability diagnostics."""
import math
from .span_candidate_intervention import canonicalize


def probabilities(logprobs):
    if len(logprobs) < 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in logprobs):
        raise ValueError('finite full categorical support required')
    if max(logprobs) > 1e-5: raise ValueError('normalized log probabilities required')
    p = [math.exp(v) for v in logprobs]
    total = sum(p)
    if abs(total - 1) > 1e-5: raise ValueError('normalized log probabilities required')
    return [v / total for v in p]


def compare(native, changed, mapping):
    a, b = probabilities(native), probabilities(changed)
    if len(a) != len(b): raise ValueError('same candidate support required')
    semantic = canonicalize(b, mapping)
    top = lambda p: max(range(len(p)), key=p.__getitem__)
    tv = lambda p, q: sum(abs(x-y) for x, y in zip(p, q, strict=True))/2
    # Exact ties follow the production first-index argmax, not an artificial
    # stable canonical tie-break. Report ties so invariance is not overstated.
    ai, bi = top(a), top(b)
    return dict(semantic_tv=tv(a, semantic), symbol_tv=tv(a, b),
        semantic_greedy_equal=ai == mapping['presented_to_canonical'][bi],
        symbol_greedy_equal=ai == bi,
        native_top_tied=a.count(max(a)) > 1, changed_top_tied=b.count(max(b)) > 1)


def summarize(pairs):
    if not pairs: raise ValueError('nonempty paired population required')
    fields = ('semantic_tv', 'symbol_tv', 'semantic_greedy_equal', 'symbol_greedy_equal',
              'native_top_tied', 'changed_top_tied')
    return dict(requests=len(pairs), **{key: sum(r[key] for r in pairs)/len(pairs) for key in fields})
