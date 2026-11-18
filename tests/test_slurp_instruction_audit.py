import copy
from pathlib import Path
import pytest
from multimodal_jev.slurp_instruction_audit import (
    completed_range, exported_at, input_policy_id, verify_rank_report,
)


def config():
    return dict(start_update=2, updates=632, total_updates=634, global_batch=128,
                checkpoint_interval=64, initial_adapter='/accepted/adapter-step0002', base_id='base')


def test_terminal_cursor_and_sparse_export_boundaries():
    c = config()
    assert completed_range(c, [317,317]) == (2,634,(2,0))
    assert [i for i in range(2,635) if exported_at(c,i,[317,317])] == [64,128,192,256,317,320,384,448,512,576,634]


@pytest.mark.parametrize('change', [dict(updates=631), dict(updates=633), dict(start_update=-1),
    dict(start_update=True), dict(updates=True), dict(total_updates=635), dict(initial_adapter=''),
    dict(checkpoint_interval=0), dict(global_batch=64)])
def test_reject_incomplete_or_changed_terminal_budget(change):
    with pytest.raises(ValueError):
        completed_range(dict(config(), **change), [317,317])


def test_policy_lineage_across_resume_exports_and_epoch_boundaries():
    c=config(); calls=[]
    def identity(p):
        calls.append(str(p));return Path(p).name+'-hash'
    expected={2:'adapter-step0002-hash',3:'adapter-step0002-hash:instruction-update3',
              64:'adapter-step0064-hash',65:'adapter-step0002-hash:instruction-update65',
              317:'adapter-step0317-hash',318:'adapter-step0002-hash:instruction-update318',
              633:'adapter-step0002-hash:instruction-update633'}
    for step,suffix in expected.items():
        assert input_policy_id(c,'/training',step,[317,317],identity)=='base:'+suffix
    assert '/training/run/adapter-step0003' not in calls
    with pytest.raises(ValueError):input_policy_id(c,'/training',634,[317,317],identity)


def test_prefix_remains_two_steps_with_original_adapter_sequence():
    c=dict(updates=2,global_batch=128,base_id='base')
    assert completed_range(c,[317])==(0,2,(0,2))
    assert input_policy_id(c,'/prefix',1,[317],lambda p:p.name)=='base:adapter-step0001'


def test_explicit_larger_batch_prefix_and_terminal_budget():
    assert completed_range(dict(updates=2,global_batch=256,microbatch_per_rank=2),[159])==(0,2,(0,2))
    c=dict(config(),global_batch=256,microbatch_per_rank=2,updates=316,total_updates=318)
    assert completed_range(c,[159,159])==(2,318,(2,0))


def report():
    return dict(passed=True,rank=0,updates=634,executed_updates=632,base_id='base',final_adapter_id='last',
        step_metrics=[dict(update=i,seconds=1.,metrics={'actor/grad_norm':[.5]}) for i in range(3,635)],
        execution_dtypes={str(i):['torch.bfloat16','torch.bfloat16'] for i in range(112)})


def test_full_rank_evidence_required():
    assert verify_rank_report(report(),{'exit_code':0},config(),rank=0,epoch_updates=[317,317],adapter_id='last')==632


@pytest.mark.parametrize('damage',['partial','duplicate','rank','endpoint','base','adapter','nan','zero','dtype','missing_hook','terminal'])
def test_reject_false_terminal_or_partial_rank_evidence(damage):
    r=copy.deepcopy(report());terminal={'exit_code':0}
    if damage=='partial':r['step_metrics'].pop()
    elif damage=='duplicate':r['step_metrics'][1]['update']=3
    elif damage=='rank':r['rank']=1
    elif damage=='endpoint':r['updates']=633
    elif damage=='base':r['base_id']='other'
    elif damage=='adapter':r['final_adapter_id']='other'
    elif damage in ('nan','zero'):r['step_metrics'][0]['metrics']['actor/grad_norm']=[float('nan') if damage=='nan' else 0.]
    elif damage=='dtype':r['execution_dtypes']['0']=['torch.float32']*2
    elif damage=='missing_hook':r['execution_dtypes'].pop('0')
    else:terminal['exit_code']=1
    with pytest.raises(ValueError):
        verify_rank_report(r,terminal,config(),rank=0,epoch_updates=[317,317],adapter_id='last')
