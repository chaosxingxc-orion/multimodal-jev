import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('input_throughput', Path(__file__).resolve().parents[1]/'tools/jev/context_input_throughput.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    rows=[dict(weight=int(i<65), input_tokens=100,
               audio_receipt=dict(output_samples=32000,sample_rate=16000)) for i in range(128)]
    batch=dict(records=rows,rank_indices=[list(range(r*32,(r+1)*32)) for r in range(4)])
    ranks=[dict(update=1,real_queries=65,checkpoint=True,total_seconds=10+r,
                preparation_seconds=2,actor_seconds=7,peak_allocated=100,
                input_service=dict(worker_seconds=20,wait_seconds=.1,bind_seconds=.2,cache_hits=128))
           for r in range(4)]
    return batch,ranks


def test_padding_parallel_times_and_overlapped_input():
    batch,ranks=fixture()
    step=module.summarize_step(1,batch,ranks)
    result=module.aggregate([step])
    assert result['real_queries']==65
    assert result['padding_slots']==63
    assert result['real_input_tokens']==6500
    assert result['slot_input_tokens']==12800
    assert result['real_audio_seconds']==130
    assert result['measured_step_seconds']==13
    assert result['real_queries_per_second']==pytest.approx(65/13)
    assert result['background_worker_seconds']==20
    assert module.aggregate([]) is None


@pytest.mark.parametrize('fault',['rank','partition','count','nan','serial','weight','save','service'])
def test_corrupt_ledgers_rejected(fault):
    batch,ranks=fixture()
    if fault=='rank':ranks.pop()
    elif fault=='partition':batch['rank_indices'][0][0]=1
    elif fault=='count':ranks[1]['real_queries']=128
    elif fault=='nan':ranks[1]['actor_seconds']=float('nan')
    elif fault=='serial':ranks[1]['total_seconds']=1
    elif fault=='weight':batch['records'][0]['weight']=.5
    elif fault=='save':ranks[1]['checkpoint']=False
    else:ranks[1]['input_service']['wait_seconds']=3
    with pytest.raises(ValueError):module.summarize_step(1,batch,ranks)


def test_residual_is_computed_within_rank():
    batch,ranks=fixture()
    ranks[0].update(total_seconds=20,preparation_seconds=15,actor_seconds=3)
    ranks[1].update(total_seconds=20,preparation_seconds=1,actor_seconds=15)
    step=module.summarize_step(1,batch,ranks)
    assert step['maximum_rank_other_seconds']==4
    assert step['total_seconds']==20
    assert step['preparation_seconds']+step['actor_seconds']>20


def test_collection_binding_and_complete_range(tmp_path):
    batch,ranks=fixture()
    files={}
    def write(name,value):
        p=tmp_path/name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value))
        files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    write('config.json',dict(stage='instruction',global_batch=128,microbatch=8,
          physical_devices=[4,5,6,7],start_update=0,stop_update=1))
    write('run/batch-0000.json',batch)
    for rank,row in enumerate(ranks):
        write(f'run/rank-{rank}/result.json',dict(passed=True,start_update=0,stop_update=1))
        write(f'run/rank-{rank}/update-0001.json',row)
    proof=tmp_path/'proof.json'
    proof.write_text(json.dumps(dict(container_id='synthetic',files=files,
          state=dict(Status='exited',ExitCode=0,OOMKilled=False))))
    result=module.run(tmp_path,proof)
    assert result['all_steps']['real_queries']==65
    assert result['normal_after_warmup'] is None
    assert result['first32_normal'] is None
    (tmp_path/'run/rank-3/update-0001.json').write_text('{}')
    with pytest.raises(ValueError,match='changed collected ledger'):
        module.run(tmp_path,proof)
