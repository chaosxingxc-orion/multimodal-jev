#!/usr/bin/env python3
"""Use the standard sampler with an explicitly diagnostic live HF audio tower."""
import json
from functools import partial
from pathlib import Path
import runpy
import sys

from multimodal_jev import rollout
from multimodal_jev.baseline import atomic_json
from multimodal_jev.hf_audio_worker_probe import install_hf_audio_probe


if __name__ == '__main__':
    config_path = Path(sys.argv[sys.argv.index('--config')+1])
    output = Path(sys.argv[sys.argv.index('--output')+1])
    config = json.loads(config_path.read_text())
    if config.get('diagnostic_only') is not True:
        raise ValueError('diagnostic_only must be explicit in saved sampling identity')
    original = rollout.VllmAudioRollout

    class DiagnosticRollout(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            try:
                receipt = self.engine.apply_model(partial(install_hf_audio_probe, checkpoint=config['checkpoint']))
                if len(receipt) != 1 or receipt[0]['policy_id'] != self.policy_id:
                    raise ValueError('worker checkpoint mismatch')
                atomic_json(output/'audio-backend.json', receipt)
            except BaseException:
                self.close()
                raise

    rollout.VllmAudioRollout = DiagnosticRollout
    runpy.run_path(str(Path(__file__).with_name('sample_audio_policy.py')), run_name='__main__')
