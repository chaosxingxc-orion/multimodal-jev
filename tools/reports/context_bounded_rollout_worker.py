"""Production-shaped candidate worker; no diagnostic activation captures."""
from tools.reports.context_acoustic_worker_candidate import AcousticBatchWorker
from tools.reports.context_bounded_attention_worker import BoundedMixin


class BoundedAcousticWorker(BoundedMixin, AcousticBatchWorker):
    pass
