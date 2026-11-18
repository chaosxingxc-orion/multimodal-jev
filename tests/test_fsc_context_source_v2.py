"""New source auditor must retain original policy/package semantics."""
from tools.jev import prepare_fsc_context_source as original
from tools.jev import prepare_fsc_context_source_v2 as current
from test_fsc_context_source import source


def test_new_curve_auditor_keeps_real_policy_assets(source,monkeypatch):
    args,curve,calls=source
    monkeypatch.setattr(current,'load_run',original.load_run)
    report=current.prepare(**args)
    assert calls==[(args['experiment_directory'],'grpo_v1')]
    assert report['policies']['selected']['policy_id']==curve['selected_checkpoint']['policy_id']
    assert current.verify(args['output'],current.sha(args['output']/'PACKAGE.json'))==report
    assert original.verify(args['output'],current.sha(args['output']/'PACKAGE.json'))==report


def test_source_uses_rounding_aware_curve_auditor():
    from tools.jev.compare_context_rl_curves_v2 import load_run
    assert current.load_run is load_run
