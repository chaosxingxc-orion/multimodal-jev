import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from context_cache_receipts import parse_stream,DRIVER_TAIL

def wire():
    return '\n'.join(map(json.dumps,[{'kind':'terminal','value':{'exit_code':0}},
        {'kind':'input','value':{'id':'a'}},{'kind':'manifest','value':{'recordings':1}}]))+'\n'

def test_exact_trailing_diagnostic_preserved():
    assert parse_stream(wire())[1]==[]
    records,diagnostics=parse_stream(wire()+DRIVER_TAIL+'\n')
    assert len(records)==3 and diagnostics==[DRIVER_TAIL]

@pytest.mark.parametrize('text',[wire()+'other log\n',DRIVER_TAIL+'\n'+wire(),
    wire()+DRIVER_TAIL+'\n'+DRIVER_TAIL+'\n',wire().replace('"recordings": 1','"recordings": 2'),
    wire().replace('"exit_code": 0','"exit_code": 1')])
def test_unknown_noise_or_incomplete_evidence_rejected(text):
    with pytest.raises(ValueError):parse_stream(text)
