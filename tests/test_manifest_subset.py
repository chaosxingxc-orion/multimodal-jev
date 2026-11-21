import pytest
from tools.reports.collect_manifest_subset import selected_files

def test_explicit_scope_excludes_features_and_intermediate_versions():
    files={'run/policy-0064/execution.pt':'a','run/policy-0064/processor/tokenizer.json':'b',
           'run/policy-0063/execution.pt':'c','cache/feature.pt':'d','run/checkpoint-0064-committed.json':'e'}
    assert set(selected_files(files,['run/checkpoint-0064-committed.json'],['run/policy-0064']))==set(files)-{'run/policy-0063/execution.pt','cache/feature.pt'}

@pytest.mark.parametrize('name',['../outside','/absolute','run/missing','run//double'])
def test_missing_or_escaping_request_rejected(name):
    with pytest.raises(ValueError):selected_files({'run/a':'x'},[name],[])
