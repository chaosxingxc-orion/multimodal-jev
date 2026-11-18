"""Attention-boundary capture with the shared worker capture lifecycle."""
from types import FunctionType

from tools.reports.context_language_stage_worker import CaptureMixin as LanguageMixin
from tools.reports.context_acoustic_worker_candidate import AcousticBatchWorker,AcousticBaselineWorker
from tools.reports.context_attention_stage_capture import attach


class CaptureMixin:
    jev_stage_capture=FunctionType(LanguageMixin.jev_stage_capture.__code__,
        dict(LanguageMixin.jev_stage_capture.__globals__,attach=attach),'jev_stage_capture')


class StageAcousticWorker(CaptureMixin,AcousticBatchWorker):pass
class StageBaselineWorker(CaptureMixin,AcousticBaselineWorker):pass
