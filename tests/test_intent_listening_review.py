from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from verify_intent_listening_review import validate
from multimodal_jev.slurp_training import content_hash


def fixture():
    packet=[dict(request=dict(id='a',source_group='g',audio_sha256='audio'),reference='spoken words')]
    row=dict(id='a',source_group='g',audio_sha256='audio',record_sha256=content_hash(packet[0]),
        alignment='matches',ontology='consistent',notes='reviewed',reviewer='human',recorded_at='2026-11-12T04:00:00Z',
        evaluation_allowed=False,optimization_allowed=False)
    ledger=dict(schema='slurp-intent-listening-review-v1',packet_sha256='packet',reference_audit_sha256='ref',records=[row],
        evaluation_allowed=False,optimization_allowed=False)
    return packet,ledger


def test_complete_listening_claim_does_not_grant_admission():
    p,l=fixture();r=validate(p,l,'packet','ref')
    assert r['both_fields_reviewed']==1 and not r['evaluation_allowed'] and not r['optimization_allowed']
    l['records'][0].update(alignment='pending',ontology='pending',notes='')
    assert validate(p,l,'packet','ref')['both_fields_reviewed']==0


@pytest.mark.parametrize('field,value',[('audio_sha256','other'),('record_sha256','other'),('notes',''),
    ('reviewer',''),('recorded_at','2026-11-12'),('evaluation_allowed',True),('ontology','approved')])
def test_rejects_rebinding_missing_attribution_and_premature_admission(field,value):
    p,l=fixture();l['records'][0][field]=value
    with pytest.raises(ValueError):validate(p,l,'packet','ref')
