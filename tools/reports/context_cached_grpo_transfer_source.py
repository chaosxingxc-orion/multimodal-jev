"""Reuse source-role admission for the actual cached-GRPO training layout."""
import inspect

from tools.reports import context_transfer_source_admission as shared
from tools.reports.collect_context_cached_grpo_curve import reconstruct


def admit(c):
    source = inspect.getsource(shared.admit)
    before = "Path(source_ops['prepared'])/'package/training.json'"
    after = "Path(source_ops['training_package']['output'])/'config.json'"
    if source.count(before) != 1:
        raise ValueError('shared source admission layout anchor changed')
    source = source.replace(before, after)
    namespace = dict(shared.admit.__globals__, reconstruct=reconstruct)
    exec(compile(source, '<cached-grpo-transfer-source>', 'exec'), namespace)
    return namespace['admit'](c)


seal = shared.seal
verify_bundle = shared.verify_bundle
