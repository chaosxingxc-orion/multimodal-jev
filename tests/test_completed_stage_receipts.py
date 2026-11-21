import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from multimodal_jev.baseline import file_hash
from tools.reports.context_completed_stage_receipts import check_tree,CompletedStageReceipts
from tools.reports.context_stage2_release_components import graph

@pytest.mark.parametrize('change',['modify','add','delete','symlink'])
def test_tree_invalidation(tmp_path,change):
    p=tmp_path/'data';p.write_text('accepted');expected={'data':file_hash(p)}
    check_tree(tmp_path,expected)
    if change=='modify':p.write_text('modified')
    elif change=='add':(tmp_path/'extra').write_text('extra')
    elif change=='delete':p.unlink()
    else:(tmp_path/'link').symlink_to(p)
    with pytest.raises((ValueError,FileNotFoundError)):check_tree(tmp_path,expected)


def test_reuse_exact_bound_and_no_shared_mutation():
    from tools.reports import audit_objective_training as original
    old=original.audit
    receipts=CompletedStageReceipts.__new__(CompletedStageReceipts)
    receipts.bounds={'prefix':{'identity':1}}
    receipts.reports={'prefix':{'status':'accepted','package_audit':{}}}
    v=graph(SimpleNamespace(reconstruct=lambda *a:None,replay=lambda *a:None),None,completed=receipts)
    audit=v['tools.reports.audit_objective_training'].audit
    assert audit({'identity':1},{})['status']=='accepted'
    with pytest.raises(ValueError):audit({'identity':2},{})
    assert original.audit is old
    first=audit({'identity':1},{});first['status']='changed'
    assert audit({'identity':1},{})['status']=='accepted'
