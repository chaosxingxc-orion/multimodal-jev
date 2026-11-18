import copy
import json
from pathlib import Path

import pytest

from tools.reports.context_infonce_tail import contract,select,generate
from multimodal_jev.context_replica_schedule import rebatch


def config():
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'configs/experiments/context-two-stage-v0.0.4.json').read_text())['stage1']
    return dict(c,experiment_kind='infonce_tail_admission',start_update=0,stop_update=2,resume_root=None,
                engineering_prefix=True,training_dispatch_allowed=False,checkpoint_interval=1)


def test_actual_tail_selected_without_changing_formal_schedule():
    c=config();population=[f'q{i}' for i in range(40513)]
    parent=[dict(epoch=e,index=0,slots=[dict(id=q,weight=1.,padding=False) for q in population]) for e in range(2)]
    batches=rebatch(parent,population,c['batch_config']);original=copy.deepcopy(batches)
    selected=select(batches,c,population)
    assert selected==[batches[79],batches[80]] and [b['real_queries'] for b in selected]==[65,512]
    assert batches==original and contract(c)['formal_history'] is False
    assert contract(c)['optimizer_cursor']=='fresh_0_1'
    source=generate(Path(__file__).resolve().parents[1])
    assert "contract['tail_admission']=tail_contract(c)" in source and 'batches=select_tail' in source
    compile(source,'<tail>','exec')


@pytest.mark.parametrize('field,value',[('start_update',79),('stop_update',81),('training_dispatch_allowed',True),('resume_root','/previous')])
def test_diagnostic_cannot_claim_formal_history(field,value):
    c=config();c[field]=value
    with pytest.raises(ValueError):contract(c)


def test_corrupt_tail_real_population_rejected():
    c=config();population=[f'q{i}' for i in range(40513)]
    parent=[dict(epoch=e,index=0,slots=[dict(id=q,weight=1.,padding=False) for q in population]) for e in range(2)]
    batches=rebatch(parent,population,c['batch_config']);batches[79]['slots'][0]['weight']=0.
    with pytest.raises(ValueError):select(batches,c,population)
