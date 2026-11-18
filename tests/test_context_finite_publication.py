import ast
import json
from pathlib import Path

import pytest

from tools.reports.audit_context_replica_training import published_policy_commit
from tools.reports.context_cached_reference_trainer import generate


def test_auditor_reads_actual_generated_finite_publication_without_rewriting(tmp_path):
    # Execute the rank-zero publication block from the real cached trainer.
    # FSDP/model execution is outside this focused publication-format test.
    tree=ast.parse(generate(Path(__file__).resolve().parents[1]))
    blocks=[node for node in ast.walk(tree) if isinstance(node,ast.If)
        and any(isinstance(child,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ckpt_published'
            for t in child.targets) for child in node.body)]
    assert len(blocks)==1
    def commit(path,value):path.write_text(json.dumps(value))
    scope=dict(rank=0,index=0,start_update=0,root=tmp_path,world=2,save_now=True,
        contract={'fixture':'contract'},joint={'policy_id':'new-policy'},
        file_hash=lambda path:'hash:'+str(path.relative_to(tmp_path)),commit_json=commit)
    exec(compile(ast.Module(body=blocks,type_ignores=[]),'<actual-finite-publication>','exec'),scope)
    paths=list(tmp_path.iterdir());assert [p.name for p in paths]==['checkpoint-0001-committed.json']
    before=paths[0].read_bytes();full=json.loads(before)
    actual=published_policy_commit(tmp_path,1,formal=False)
    assert actual=={k:v for k,v in full.items() if k!='rank_shards'}
    assert paths[0].read_bytes()==before and len(list(tmp_path.iterdir()))==1
    assert set(full['rank_shards'])=={'rank-0/checkpoint-0001.pt','rank-1/checkpoint-0001.pt'}


def test_formal_never_falls_back_to_finite_commit(tmp_path):
    (tmp_path/'checkpoint-0001-committed.json').write_text(json.dumps({'rank_shards':{'rank0':'hash'}}))
    with pytest.raises(FileNotFoundError):published_policy_commit(tmp_path,1,formal=True)
    expected={'updates':1,'joint_policy':{'policy_id':'formal'}}
    (tmp_path/'policy-0001-committed.json').write_text(json.dumps(expected))
    assert published_policy_commit(tmp_path,1,formal=True)==expected


@pytest.mark.parametrize('shards',[None,{},[],False])
def test_finite_missing_optimizer_commit_is_rejected(tmp_path,shards):
    (tmp_path/'checkpoint-0001-committed.json').write_text(json.dumps({'rank_shards':shards}))
    with pytest.raises(ValueError,match='optimizer shards'):
        published_policy_commit(tmp_path,1,formal=False)
