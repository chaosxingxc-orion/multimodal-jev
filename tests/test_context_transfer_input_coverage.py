import json
from pathlib import Path

import pytest

from test_context_transfer_portable_inputs import build
from tools.reports import audit_context_transfer_input_coverage as m


def specs(tmp_path,monkeypatch):
    _,full,_=build(tmp_path,monkeypatch)
    root=tmp_path/'bundle';manifest=json.loads((root/'MANIFEST.json').read_text())
    return full,dict(root=str(root),manifest_sha256=m.file_hash(root/'MANIFEST.json'),
        panel_sha256=manifest['panel_sha256'],cache_root='/new/cache')


def test_actual_full_and_panel_bundles_reconstruct_identical_inputs(tmp_path,monkeypatch):
    full,panel=specs(tmp_path,monkeypatch)
    result=m.audit(full,panel)
    assert result['recordings']==result['parity_recordings']==3 and result['decisions']==14
    assert result['npu_executed'] is False and result['cache_bytes_verified'] is False


@pytest.mark.parametrize('fault',['dataset','processor','population','indices','request'])
def test_individually_valid_panel_must_match_full_population(tmp_path,monkeypatch,fault):
    full,panel=specs(tmp_path,monkeypatch)
    selected=m.load_panel(panel)
    if fault=='dataset':selected['dataset']='FSC_native31'
    if fault=='processor':selected['processor_sha256']='different'
    if fault=='population':selected['selected_panel']['population_sha256']='different'
    if fault=='indices':selected['selected_panel']['indices']=[2,1,0]
    if fault=='request':selected['requests'][selected['population'][0]['id']]=None
    monkeypatch.setattr(m,'load_panel',lambda _:selected)
    with pytest.raises(ValueError):m.audit(full,panel)
