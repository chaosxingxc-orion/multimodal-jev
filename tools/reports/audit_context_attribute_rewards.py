"""Full-population native reward contract audit with synthetic action controls."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tools.reports.context_attribute_reward_candidate import rewards


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(dataset, expected_manifest):
    dataset = Path(dataset)
    if sha(dataset/'MANIFEST.json') != expected_manifest:
        raise ValueError('Dataset identity changed')
    manifest = json.loads((dataset/'MANIFEST.json').read_text())
    for name in ('catalog.json','targets.jsonl','inputs.jsonl'):
        if sha(dataset/name) != manifest['files'][name]:
            raise ValueError('Dataset file changed: '+name)
    menu = json.loads((dataset/'catalog.json').read_text())
    final = [i for i,n in enumerate(menu['nodes']) if n['family']=='final']
    ids = [n['id'] for n in menu['nodes']]
    counts = {f:sum(n['family']==f for n in menu['nodes']) for f in ('scenario','action','final')}
    if counts != dict(scenario=18,action=46,final=60):
        raise ValueError('Expected native 124-node catalog')
    seen=set();batches=0;maximum=0.;records=0
    def check(rows):
        nonlocal batches,maximum,records
        if not rows:return
        for r in rows:
            if r['id'] in seen or r['candidate_ids']!=ids:
                raise ValueError('Duplicate recording or node order mismatch')
            seen.add(r['id'])
        y=torch.tensor([r['labels'] for r in rows],dtype=torch.bool)
        mask=torch.tensor([r['supervised'] for r in rows],dtype=torch.bool)
        a=torch.stack([y.long(),torch.zeros_like(y,dtype=torch.long),
                       torch.ones_like(y,dtype=torch.long),y.long()],dim=1)
        a[:,3,:]=0;a[:,3,final]=y[:,final].long()
        credit,parts=rewards(a,y,mask,menu,arm='q_credit')
        plain,_=rewards(a,y,mask,menu,arm='q_input')
        direct,_=rewards(a[:,:,final],y[:,final],mask[:,final],menu,arm='direct')
        if not torch.equal(plain,direct):raise ValueError('Final reward differs by arm')
        expected=torch.tensor([1.25,0.,2/61+.125*(2/19+2/47),1.]).expand_as(credit)
        torch.testing.assert_close(credit,expected,rtol=0,atol=2e-7)
        torch.testing.assert_close(credit-plain,.125*(parts['scenario']+parts['action']),rtol=0,atol=2e-7)
        maximum=max(maximum,float((credit-expected).abs().max()));records+=len(rows);batches+=1
    batch=[]
    with (dataset/'targets.jsonl').open() as stream:
        for line in stream:
            batch.append(json.loads(line))
            if len(batch)==256:check(batch);batch=[]
    check(batch)
    if records!=manifest['recordings']:raise ValueError('Population incomplete')
    return dict(dataset=str(dataset),manifest_sha256=expected_manifest,recordings=records,
                labels_checked=records*124,controls_per_recording=4,batches=batches,
                controls=['exact_truth','empty','all_selected','only_correct_Final'],
                max_reward_error=maximum)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    torch.set_num_threads(1)
    base=Path('.local/decision-objective-research')
    r=dict(status='full_population_synthetic_action_reward_contract_passed',torch_version=torch.__version__,
           scope='Local CPU reward contract only; no audio/model/sampled-policy/NPU/learning-gain evidence',
           populations=[audit(base/'slurp-native-attribute-prob-candidate-v1','d98db48e619a83e7eceb308da13f257a6cc3cfa4bf701f3f17022ff99a07a482'),
                        audit(base/'slurp-native-attribute-selection-candidate-v1','85d17ce7a12d2e39efb9bdc676cdb24b02f2531ccb4c93b7adeba303485d0710')],
           files={str(x):sha(x) for x in [Path(__file__),Path('tools/reports/context_attribute_reward_candidate.py'),
               Path('tests/test_context_attribute_reward_candidate.py'),Path('configs/experiments/context-native-attribute-comparison-design-v1.json')]})
    with a.output.open('x') as stream:stream.write(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
