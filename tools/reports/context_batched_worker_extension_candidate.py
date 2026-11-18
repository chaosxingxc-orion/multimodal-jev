"""Separate worker registration; importing does not install a segmented profile."""
from context_switch_worker import SwitchingContextWorker
from tools.reports.context_segmented_worker_candidate import install_segmented_worker


class BatchedContextWorkerCandidate(SwitchingContextWorker):
    def install_jev_segmented_candidate(self, expected_policy_id):
        model = self.get_model()
        if model._jev_policy_id != expected_policy_id:
            raise ValueError('segmented install policy mismatch')
        result = install_segmented_worker(model)
        return dict(result, policy_id=expected_policy_id)
