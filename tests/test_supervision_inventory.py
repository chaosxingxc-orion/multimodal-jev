from pathlib import Path
import runpy

classify=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/build_slurp_supervision_inventory.py'))['classify']


def test_reference_mismatch_does_not_become_a_native_label_error():
    d={'native_membership':True,'definition_fit':'not_supported'}
    assert classify(d,'intent_fit')=='native_member_definition_not_supported'
    assert d=={'native_membership':True,'definition_fit':'not_supported'}


def test_cross_class_support_is_distinct_from_member_failure():
    assert classify({'native_membership':False,'definition_fit':'supported'},'intent_fit')=='other_native_class_definition_supported'
    assert classify({'native_membership':True,'definition_fit':'unresolved'},'intent_fit')=='native_member_definition_unresolved'


def test_slot_conflict_and_missing_are_not_negative():
    for status in ('positive','negative','unavailable','conflict'):
        assert classify({'status':status},'slot')=='slot_'+status
