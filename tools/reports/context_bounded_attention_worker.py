"""Isolated full-model diagnostic with <=8 requests per attention kernel call."""
from tools.reports.context_language_stage_worker import CaptureMixin
from tools.reports.context_acoustic_worker_candidate import AcousticBaselineWorker, AcousticBatchWorker


class BoundedMixin:
    def install_jev_dual_candidate(self, expected_policy_id):
        result = super().install_jev_dual_candidate(expected_policy_id)
        import tools.reports.context_segmented_attention_candidate as attention
        from multimodal_jev.bounded_prefill_attention import bounded_prefill_kernel
        attention.masked_row_varlen_kernel = bounded_prefill_kernel
        result['attention_kernel_max_requests'] = 8
        return result


class StageBaselineWorker(CaptureMixin, BoundedMixin, AcousticBaselineWorker): pass
class StageAcousticWorker(CaptureMixin, BoundedMixin, AcousticBatchWorker): pass
