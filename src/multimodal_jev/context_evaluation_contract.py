"""One metric vocabulary for every writer, reader and comparison.

Before this module roughly eight tool files each carried their own literal field
tuple, so a metric added in one place silently disappeared from a comparison in
another. The canonical order lives here; existing writers keep their own literals
so pinned packages stay byte-identical, and the smoke audit asserts those
literals still agree with the contract rather than trusting that they do.

Every field is either a finite number or an explicit ``None``. ``None`` means
the ratio is undefined (no denominator) or the supervision is missing; it is
never zero and never silently dropped.
"""
import math

# Byte-identical to the literal ContextEventMetrics already emits, so an
# existing frozen score file keeps its scope string verbatim.
SCOPE = ('Exclusive benchmark event identity; independent sigmoid sets at fixed '
         'threshold, not semantic intent presence.')

# Core comparable metrics, in canonical order. Every writer's tuple must be a
# subset and the union across writers must equal this plus DERIVED_FIELDS.
METRIC_FIELDS = ('query_macro_set_f1', 'exact_set', 'top1_event_accuracy',
                 'top1_macro_f1', 'candidate_binary_nll', 'candidate_brier',
                 'mean_selected_count')

# Reportable but not core: computed by the caller from raw selection counts.
DERIVED_FIELDS = ('empty_selection_rate', 'multiple_selection_rate')

# The selection rule names exactly these; changing either invalidates a
# selection decision that was already made under the published rule.
SELECTION_FIELDS = ('query_macro_set_f1', 'candidate_binary_nll')

# Emitted by ContextEventMetrics.result() beyond the comparable fields.
IDENTITY_FIELDS = ('queries', 'candidate_ids', 'threshold', 'top1_tie_rule',
                   'classes', 'confusion', 'scope', 'set_precision', 'set_recall',
                   'set_precision_defined_queries', 'binary_nll_infinite')


def validate_row(row, fields):
    """Assert one metric row carries exactly ``fields``, each finite or None."""
    if type(row) is not dict:
        raise ValueError('metric row object required')
    if set(row) != set(fields):
        raise ValueError('metric row fields differ from the declared contract')
    for name in fields:
        value = row[name]
        if value is None:
            continue
        if type(value) not in (int, float) or isinstance(value, bool):
            raise ValueError(f'metric {name} must be a number or null')
        if not math.isfinite(value):
            raise ValueError(f'metric {name} must be finite or null')
    return dict(row)


def from_result(result):
    """Project an evaluator result onto the comparable metric fields."""
    if type(result) is not dict:
        raise ValueError('evaluator result object required')
    missing = [name for name in METRIC_FIELDS if name not in result]
    if missing:
        raise ValueError('evaluator result lacks contracted metrics: ' + ', '.join(missing))
    return validate_row({name: result[name] for name in METRIC_FIELDS}, METRIC_FIELDS)


def ranked_key(metrics):
    """The published selection order: highest set F1, then lowest NLL.

    Only the two selection fields are read; a caller may pass a whole metric row,
    because the rule is applied to rows as they are reported, not to a projection
    of them.
    """
    if type(metrics) is not dict:
        raise ValueError('metric row object required')
    missing = [name for name in SELECTION_FIELDS if name not in metrics]
    if missing:
        raise ValueError('selection requires: ' + ', '.join(missing))
    row = validate_row({name: metrics[name] for name in SELECTION_FIELDS}, SELECTION_FIELDS)
    f1, nll = row['query_macro_set_f1'], row['candidate_binary_nll']
    if f1 is None:
        raise ValueError('selection requires a defined query_macro_set_f1')
    return (-f1, math.inf if nll is None else nll)


def delta(initial, later, fields=METRIC_FIELDS):
    """Paired descriptive difference; undefined on either side stays undefined."""
    a = validate_row(initial, fields)
    b = validate_row(later, fields)
    return {name: None if a[name] is None or b[name] is None else b[name] - a[name]
            for name in fields}
