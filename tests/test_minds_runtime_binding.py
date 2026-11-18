"""Recheck actual local input evidence; mutations never touch frozen artifacts."""
import hashlib
import json
from pathlib import Path

import pytest
from tools.reports.minds_context_runtime import load_inputs


@pytest.mark.parametrize('fault', ['terminal', 'same_process', 'waveform', 'verifier', 'protocol'])
def test_forged_admission_rejected(tmp_path, fault):
    path=Path('.local/decision-objective-research/minds-context-runtime-inputs-v1.json')
    if not path.exists():pytest.skip('Full local MINDS evidence not installed')
    config=json.loads(path.read_text())
    key={'terminal':'verification_transport','same_process':'verification_transport',
         'waveform':'processor_audit','verifier':'actual_verification','protocol':'protocol'}[fault]
    value=json.loads(Path(config[key]['path']).read_text())
    if fault=='terminal':value['state']['ExitCode']=1
    elif fault=='same_process':
        value['container_id']=json.loads(Path(config['preparation_transport']['path']).read_text())['container_id']
    elif fault=='waveform':value['all_whole_audio_hops_present']=False
    elif fault=='verifier':value['recordings']=8167
    else:value['evaluation']['set_threshold']=.6
    changed=tmp_path/'changed.json';changed.write_text(json.dumps(value))
    config[key]=dict(path=str(changed),sha256=hashlib.sha256(changed.read_bytes()).hexdigest())
    with pytest.raises(ValueError):load_inputs(config)
