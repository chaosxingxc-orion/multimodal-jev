from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from audit_semantic_prediction_ledger import validate_terminal_coverage, decode_spans


def row(uid='r1'):
    return dict(request_id=uid,recording_id='audio1',panel_id='p0',arm='Q_input',transform='canonical')


def decode(receipt):
    return decode_spans(receipt,final_position='1',native_symbols={'A':'native_0','B':'native_1'},
                        process_symbols={'0':{'A':False,'B':True}})


def test_explicit_failure_keeps_population_and_ignores_partial_answer():
    r=dict(row(),status='failed',error='backend failed',spans={'0':'B','1':'A'})
    assert validate_terminal_coverage([row()],[r])['failed']==1
    assert decode(r)==dict(final=None,process={'0':None},envelope_valid=False)


@pytest.mark.parametrize('receipts', [[],[dict(row(),status='running')],
    [dict(row(),status='failed')], [dict(row(),status='completed')]*2,
    [dict(row('unknown'),status='completed')],
    [dict(row(),status='completed',arm='Q_credit')]])
def test_incomplete_duplicate_mismatched_or_unfinished_receipts_rejected(receipts):
    with pytest.raises(ValueError):validate_terminal_coverage([row()],receipts)


def test_independent_symbol_namespaces_and_missing_node():
    r=decode(dict(status='completed',spans={'0':'A','1':'A'}))
    assert r['process']=={'0':False} and r['final']=='native_0'
    r=decode(dict(status='completed',spans={'1':'A'}))
    assert r['final']=='native_0' and r['process']=={'0':None}


@pytest.mark.parametrize('value',[False,0,'false','C',None])
def test_no_boolean_coercion_or_guessed_symbols(value):
    r=decode(dict(status='completed',spans={'0':value,'1':'B'}))
    assert r['final']=='native_1' and r['process']=={'0':None}


def test_foreign_coordinate_or_malformed_envelope_invalidates_all_nodes():
    for spans in ({'0':'B','1':'A','2':'B'},['B','A'],None):
        r=decode(dict(status='completed',spans=spans))
        assert r==dict(final=None,process={'0':None},envelope_valid=False)


def test_direct_has_no_synthetic_process_answers():
    r=decode_spans(dict(status='completed',spans={'0':'A'}),final_position='0',
                   native_symbols={'A':'native_0'},process_symbols={})
    assert r['final']=='native_0' and r['process']=={}


def test_native_rebinding_changes_inverse_map_only():
    r=decode_spans(dict(status='completed',spans={'0':'B','1':'A'}),final_position='1',
                   native_symbols={'A':'native_1','B':'native_0'},process_symbols={'0':{'A':False,'B':True}})
    assert r['final']=='native_1' and r['process']=={'0':True}
