#!/usr/bin/env python3
"""Exact selective-duplication examples; not an actor or new RL objective."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path


def mean_credit(correct, masks, cap=F(1, 10)):
    if len(correct) != len(masks) or not correct:
        raise ValueError('nonempty aligned configured questions required')
    if any(type(v) is not bool for v in [*correct, *masks]):
        raise ValueError('boolean correctness and admission masks required')
    return cap * sum(c and m for c, m in zip(correct, masks)) / len(correct)


def audit():
    # Two fixed candidate outcomes; adding an identical-label/identical-answer
    # coordinate introduces no information yet changes correctness weighting.
    a = [True, False, False]
    b = [False, True, True]
    copies = 3
    aa = [a[0]] * copies + a[1:]
    bb = [b[0]] * copies + b[1:]
    before = [mean_credit(x, [True] * len(x)) for x in (a, b)]
    after = [mean_credit(x, [True] * len(x)) for x in (aa, bb)]
    assert before[0] < before[1] and after[0] > after[1]
    complete_copy = mean_credit(a * 2, [True] * 6)
    assert complete_copy == before[0]
    # Masking an added coordinate must not renormalize remaining valid ones.
    masked = mean_credit([True, False], [True, False])
    assert masked == F(1, 20)
    return dict(
        schema='jev-semantic-reward-redundancy-audit-v1',
        original_correctness={'A': a, 'B': b},
        expanded_correctness={'A': aa, 'B': bb},
        before=list(map(str, before)), after=list(map(str, after)),
        uniform_copy_preserves_credit=str(complete_copy),
        masked_coordinate_fixed_denominator=str(masked),
        selective_duplication_reverses_ranking=True,
        process_cap='1/10',
        scope='Fixed-answer reward arithmetic only. Shared-label predicates need not have identical model answers. No proof of shared-backbone gradient impact or empirical model benefit.',
        decision='Keep current fixed-count baseline unchanged; freeze semantic menu and measure repeated/concentrated supervision before experiments. Do not dynamically remove overlapping Q or infer labels by hierarchy.',
        training_dispatch_allowed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
