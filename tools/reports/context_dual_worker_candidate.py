"""Worker opt-in retains the existing head/LoRA activation transaction."""
from tools.reports.context_batched_worker_extension_candidate import BatchedContextWorkerCandidate
from multimodal_jev.context_logits_wire import install_logits_readout


class DualContextWorkerCandidate(BatchedContextWorkerCandidate):
    def jev_dual_measurement(self, operation):
        # RPC carries only a bounded string, never serialized executable code.
        from tools.reports.context_dual_measurement_candidate import install_serial_wire
        from tools.reports.context_batch_measurement_candidate import install_counters, reset_counters, read_counters
        operations = dict(serial_wire=install_serial_wire, install=install_counters,
                          reset=reset_counters, read=read_counters)
        if operation not in operations:
            raise ValueError('unknown finite measurement operation')
        return operations[operation](self.get_model())

    def install_jev_dual_candidate(self,expected_policy_id):
        receipt=self.install_jev_segmented_candidate(expected_policy_id)
        wire=install_logits_readout(self.get_model().pooler.readout)
        return dict(receipt,**wire)
