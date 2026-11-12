import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from verify_intent_development_review import verify
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.slurp_training import content_hash


def test_screening_cannot_silently_become_admission(tmp_path):
    packet=tmp_path/'packet.json';audit=tmp_path/'audit.json'
    source=dict(request={'id':'one','source_group':'source'},target={'native_target':{'scenario':'alarm','action':'set'}})
    atomic_json(packet,[source])
    row=dict(index=0,id='one',source_group='source',intent='alarm_set',record_sha256=content_hash(source),
        status='reference_consistent',reason='Explicit alarm request',audio_reviewed=False,evaluation_allowed=False,optimization_allowed=False)
    document=dict(packet_sha256=file_hash(packet),records=[row],reference_consistent=1,pending=0)
    atomic_json(audit,document);assert verify(packet,audit)['counts']=={'reference_consistent':1}
    for field in ('audio_reviewed','evaluation_allowed','optimization_allowed'):
        row[field]=True;atomic_json(audit,document)
        with pytest.raises(ValueError,match='cannot grant'):verify(packet,audit)
        row[field]=False
    row['record_sha256']='changed';atomic_json(audit,document)
    with pytest.raises(ValueError,match='source identity'):verify(packet,audit)
