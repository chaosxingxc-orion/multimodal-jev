"""Read-only staged acoustic capture on the actual aligned worker."""
from pathlib import Path
import torch
from tools.reports.context_acoustic_worker_candidate import AcousticBatchWorker, AcousticBaselineWorker


from tools.reports.context_audio_stage_capture import attach


class CaptureMixin:
    def jev_stage_capture(self, operation, destination):
        from multimodal_jev.local_checkpoint import commit_file
        from multimodal_jev.local_runtime_storage import check_paths
        model = self.get_model()
        if operation == 'start':
            if getattr(model, '_jev_stage_handles', None): raise ValueError('capture already active')
            model._jev_stage_handles, model._jev_stage_values = attach(model.audio_tower, model.audio_tower)
            return {'capture_started': True}
        if operation != 'finish' or not model._jev_stage_handles: raise ValueError('active capture required')
        for hook in model._jev_stage_handles: hook.remove()
        model._jev_stage_handles = []
        path = Path(destination); check_paths({'capture': path})
        commit_file(path, lambda f: torch.save(model._jev_stage_values, f))
        names = list(model._jev_stage_values)
        model._jev_stage_values = None
        return {'stages': names}


class StageAcousticWorker(CaptureMixin, AcousticBatchWorker): pass
class StageBaselineWorker(CaptureMixin, AcousticBaselineWorker): pass
