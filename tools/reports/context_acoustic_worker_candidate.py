"""Opt-in acoustic candidate on the existing dual-logit worker."""
from tools.reports.context_dual_worker_candidate import DualContextWorkerCandidate
from tools.reports.context_acoustic_batch_candidate import install


class AcousticBatchWorker(DualContextWorkerCandidate):
    def install_jev_dual_candidate(self,expected_policy_id):
        result=super().install_jev_dual_candidate(expected_policy_id)
        result.update(install(self.get_model(),conv_batch_chunks=32))
        return result

    def jev_acoustic_counters(self,operation):
        model=self.get_model();tower=model.audio_tower
        if operation=='install':
            if hasattr(model,'_jev_acoustic_hooks'):raise ValueError('fresh acoustic counters required')
            model._jev_acoustic_counts={'conv':[],'encoder':[]}
            def conv(module,args):model._jev_acoustic_counts['conv'].append(list(args[0].shape))
            def encoder(module,args):model._jev_acoustic_counts['encoder'].append(args[1].detach().cpu().tolist())
            model._jev_acoustic_hooks=[tower.conv2d1.register_forward_pre_hook(conv),
                                      tower.layers[0].register_forward_pre_hook(encoder)]
            return dict(installed=True)
        if operation=='reset':
            model._jev_acoustic_counts={'conv':[],'encoder':[]};return dict(reset=True)
        if operation=='read':return model._jev_acoustic_counts
        raise ValueError('unknown acoustic telemetry operation')


class AcousticBaselineWorker(AcousticBatchWorker):
    def install_jev_dual_candidate(self,expected_policy_id):
        return DualContextWorkerCandidate.install_jev_dual_candidate(self,expected_policy_id)
