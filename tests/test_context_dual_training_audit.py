import ast
from pathlib import Path

import pytest

from tools.reports.audit_context_dual_training import expected_contract
from tools.reports.context_dual_trainer_candidate import generate


@pytest.mark.parametrize('algorithm',['dual_credit_rloo_clipped_v1','dual_credit_grpo_floor_clipped_v1'])
def test_independent_contract_matches_actual_trainer_contract_expression(algorithm):
    # Extract actual bounded trainer contract statements, without importing
    # models/devices or executing the training loop.
    tree=ast.parse(generate(Path.cwd()))
    c={k:k for k in ('base_id','source_policy_id','source_contract','runtime_profile',
        'release_sha256','schedule_sha256','behavior_bounds')}
    c.update(global_batch=128,microbatch=8,stage='task',algorithm=algorithm,
             dual_arm='joint',dual_runtime_identity={'source':'fixed'})
    statements=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='contract' for v in node.targets):
            statements.append(node)
        if (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and
                isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and
                node.value.func.value.id=='contract' and node.value.func.attr=='update'):
            statements.append(node)
    assert len(statements)==2
    ns=dict(c=c,runtime_identity=c['dual_runtime_identity'],options=dict(algorithm=algorithm,arm='joint',
        ranking_coefficient=.1,probability_coefficient=.3,kl_coefficient=.01,reference_policy_id=c['source_policy_id']))
    exec(compile(ast.Module(body=sorted(statements,key=lambda n:n.lineno),type_ignores=[]),'<contract>','exec'),ns)
    assert ns['contract']==expected_contract(c)
