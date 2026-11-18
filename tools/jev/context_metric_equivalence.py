"""Strict result comparison with bounded rounding only for two aggregate scores."""
import math


ROUNDING_PATHS = {('metrics', 'candidate_binary_nll'), ('metrics', 'candidate_brier')}


def compare(recomputed, reported):
    """Check every recomputed field; caller authenticates extra runtime fields.

    No tolerance on probabilities, identities, counts, decisions or set metrics.
    Eight binary64 ULPs allow platform libm rounding in aggregate NLL/Brier.
    """
    differences = []

    def visit(a, b, path):
        if type(a) is not type(b):
            raise ValueError(f'metric type differs: {path}')
        if isinstance(a, dict):
            if set(a) != set(b):
                raise ValueError(f'metric keys differ: {path}')
            for key in a:
                visit(a[key], b[key], path + (key,))
        elif isinstance(a, list):
            if len(a) != len(b):
                raise ValueError(f'metric length differs: {path}')
            for index, (x, y) in enumerate(zip(a, b, strict=True)):
                visit(x, y, path + (index,))
        else:
            if isinstance(a, float) and (not math.isfinite(a) or not math.isfinite(b)):
                raise ValueError(f'nonfinite metric: {path}')
            if a == b:
                return
            if path not in ROUNDING_PATHS or type(a) is not float:
                raise ValueError(f'metric differs: {path}')
            bound = 8 * max(math.ulp(a), math.ulp(b))
            if abs(a - b) > bound:
                raise ValueError(f'metric exceeds binary64 rounding allowance: {path}')
            differences.append(dict(path='.'.join(path), recomputed=a, reported=b,
                                    absolute_difference=abs(a-b), bound=bound))

    if not isinstance(recomputed, dict) or not isinstance(reported, dict):
        raise ValueError('result mappings required')
    if not set(recomputed) <= set(reported):
        raise ValueError('reported result fields missing')
    visit(recomputed, {k: reported[k] for k in recomputed}, ())
    return dict(passed=True, rounding_differences=differences,
                rule='Only aggregate binary NLL/Brier permit at most 8 binary64 ULPs')
