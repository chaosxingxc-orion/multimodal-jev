import copy
import pytest
from tools.reports.measure_context_transport_candidate import measure,compare
from test_context_bulk_rpc_candidate import row,record


class Engine:
    bundle={'policy_id':'a'*64}
    def __init__(self):self.single=0;self.bulk=0
    def score(self,**kw):
        self.single+=1
        return record(self.bundle['policy_id'],{k:kw[k] for k in ('inputs','positions')})
    def score_many(self,rows,**kw):
        self.bulk+=1
        return [record(self.bundle['policy_id'],r) for r in rows]


def pair():
    rows=[row(3),row(7),row(3)]
    clients=[Engine(),Engine()];reports=[]
    for client,protocol in zip(clients,('sequential_v1','bulk_sequential_v1')):
        reports.append(measure(client,rows,protocol=protocol,audio_token_id=1,clock=iter(range(6)).__next__))
    return clients,reports


def test_finite_passes_same_inputs_probabilities_and_sampling():
    clients,reports=pair()
    assert clients[0].single==9 and clients[1].bulk==3
    result=compare(*reports)
    assert result['exact_records_and_G16'] and result['observed_time_ratio']==1


@pytest.mark.parametrize('fault',['policy','probability','sampling','population','passes'])
def test_mismatched_comparisons_rejected(fault):
    _,reports=pair();a,b=copy.deepcopy(reports)
    if fault=='policy':b['policy_id']='c'*64
    if fault=='probability':b['passes'][1]['records'][0]['prob'][0]=.4
    if fault=='sampling':b['passes'][1]['sampling_receipts'][0]['seed']=999
    if fault=='population':b['rows']=2
    if fault=='passes':b['passes'].pop()
    with pytest.raises(ValueError):compare(a,b)
