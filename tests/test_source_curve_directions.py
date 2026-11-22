from tools.reports.source_curve_directions import summarize


def test_two_sided_changes_count_each_boundary_once():
    directions = {
        "a->b": dict(n=4, reverse_n=3, mid_correct=1, final_correct=3),
        "b->a": dict(n=3, reverse_n=4, mid_correct=2, final_correct=1),
        "a->c": dict(n=1, reverse_n=0, mid_correct=1, final_correct=1),
    }
    observed = summarize({"directions": directions, "error-q2": {"points": {"stage1": {"n": 8}}}})
    assert observed == {
        "source_errors": 8, "tradeoff_boundaries": 1, "tradeoff_rows": 7,
        "tradeoff_gains": 2, "tradeoff_losses": 1, "without_reverse_source_rows": 1,
    }
