import copy

import pytest
import torch

from multimodal_jev.context_logits_wire import WIRE
from tools.reports.context_dual_measurement_candidate import (
    audit_dual_measurements, compare_dual, install_serial_wire)
from test_context_batch_measurement_candidate import measured
from test_context_batched_service_candidate import engine_case


def dual_measured(mode):
    result=measured(mode)
    # Fixture-only logits. Real producer must capture its actual projection;
    # inverse sigmoid here is not evidence of real-model logits or parity.
    for item in result['passes']:
        for record in item['records']:
            record.update(output_protocol=WIRE,
                logits=torch.tensor(record['prob']).logit().tolist())
    return result


def check(a,b):
    _,rows,_=engine_case()
    return audit_dual_measurements(a,b,rows,'a'*64,'a'*64,4,16384)


def test_actual_logits_required_in_all_passes_and_existing_input_audit_reused():
    a,b=dual_measured('serial'),dual_measured('batched')
    assert compare_dual(a,b)['exact_actual_logits']
    result=check(a,b)
    assert result['exact_actual_logits'] and result['actual_call_reduction']
    assert result['serial_over_batched_time_ratio']==2


@pytest.mark.parametrize('fault',[
    'missing_logits','wrong_protocol','nan_logit','incompatible_logit',
    'same_bad_sampling','same_bad_input','same_bad_order','same_bad_lengths',
    'same_bad_memory','missing_warmup'])
def test_agreement_alone_does_not_admit_corrupt_measurements(fault):
    a,b=dual_measured('serial'),dual_measured('batched')
    for report in (a,b):
        p=report['passes'][0];r=p['records'][0]
        if fault=='missing_logits':r.pop('logits')
        if fault=='wrong_protocol':r['output_protocol']='prob-only'
        if fault=='nan_logit':r['logits'][0]=float('nan')
        if fault=='incompatible_logit':r['logits'][0]+=1
        if fault=='same_bad_sampling':p['sampling'][0]['seed']+=1
        if fault=='same_bad_input':r['request_identity']['positions']=[0]
        if fault=='same_bad_order':report['passes'][3]['order']=[0,1]
        if fault=='same_bad_lengths':p['telemetry']['counts']['language']=[[8],[6]]
        if fault=='same_bad_memory':p['telemetry']['peak_allocated_bytes']=100
        if fault=='missing_warmup':report['passes'].pop(0)
    with pytest.raises(ValueError):check(a,b)


def test_exact_logit_equality_is_stricter_than_corruption_tolerance():
    a,b=dual_measured('serial'),dual_measured('batched')
    # Both logits pass the loose cross-device consistency bound, but a changed
    # actual logit is still rejected by exact serial/batch comparison.
    b=copy.deepcopy(b)
    for p in b['passes']:p['records'][0]['logits'][0]+=1e-7
    with pytest.raises(ValueError):compare_dual(a,b)
    with pytest.raises(ValueError):check(a,b)


def test_serial_wire_requires_loaded_model_and_preserves_readout():
    from types import SimpleNamespace as NS
    from test_context_pooling import case
    readout,*_=case()
    model=NS(pooler=NS(memory_profiling=True,readout=readout),
        _jev_base_weights_loaded=True,_jev_policy_id='a'*64)
    with pytest.raises(ValueError):install_serial_wire(model)
    model.pooler.memory_profiling=False;model._jev_base_weights_loaded=False
    with pytest.raises(ValueError):install_serial_wire(model)
    model._jev_base_weights_loaded=True
    receipt=install_serial_wire(model)
    assert model.pooler.readout is readout and receipt['extra_head_forwards']==0
    assert receipt['policy_id']=='a'*64 and receipt['wire']==WIRE


def test_named_rpc_dispatches_only_registered_operations(monkeypatch):
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    # Execute the actual method body without importing the unavailable local
    # vLLM base class. Actual worker registration remains an NPU runtime check.
    tree=ast.parse(Path('tools/reports/context_dual_worker_candidate.py').read_text())
    method=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)
                and node.name=='jev_dual_measurement')
    namespace={};exec(compile(ast.Module(body=[method],type_ignores=[]),'<measurement-method>','exec'),namespace)
    dispatch=namespace['jev_dual_measurement']
    from tools.reports import context_batch_measurement_candidate as counters
    import tools.reports.context_dual_measurement_candidate as measurement
    model=object();worker=SimpleNamespace(get_model=lambda:model)
    for operation, module, name in [('serial_wire',measurement,'install_serial_wire'),
            ('install',counters,'install_counters'),('reset',counters,'reset_counters'),
            ('read',counters,'read_counters')]:
        monkeypatch.setattr(module,name,lambda value: {'same_model':value is model})
        assert dispatch(worker,operation)=={'same_model':True}
    with pytest.raises(ValueError):dispatch(worker,'eval')


def test_serial_owner_only_changes_worker_registration(monkeypatch):
    from pathlib import Path
    import builtins
    from tools.reports.context_dual_measurement_candidate import serial_owner_class
    captured=[];real_compile=builtins.compile
    def capture(source,*args,**kw):
        captured.append(source)
        return real_compile("class SwitchingContextRollout: pass",*args,**kw)
    monkeypatch.setattr(builtins,'compile',capture)
    serial_owner_class(Path.cwd())
    original=Path('tools/jev/context_switch_rollout.py').read_text()
    assert captured==[original.replace('context_switch_worker.SwitchingContextWorker',
        'tools.reports.context_dual_worker_candidate.DualContextWorkerCandidate')]
