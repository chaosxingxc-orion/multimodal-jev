"""Full release-record inference ledger, separate from label access and scoring."""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

from .audio2tool_input import validate
from .baseline import file_hash

INPUT_SHA='f7e4ebe4e7cf2cf411f3f26a8714bbdd0a493c0c7aeb98dd4199112d197e1a35'
BUDGET_SHA='87e750243a3dcc56ca006e9f661dcaa2ac130cc27fd7312d6e986892aa1ef223'
COUNTS={1:4292,2:6320,3:4292,4:4278,5:4292,6:4292,7:2146,8:821}
CAPACITY_PREPARATION_SHA='c626a456f3a82e394321eb933fa2fc3347330c7eb855e5b1c7d01cc9142da068'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def require_capacity(root,inspection_path,inspection_hash,container_id,processor_run):
    """Coordinator must capture and pin a live Docker inspection of this ID.

    An inspection file alone is not an independent trust root. No engine is
    loaded if the registered capacity attempt is incomplete or was preflight.
    """
    from .audio2tool_engine import ENGINE,VERSIONS,prior_records
    root=Path(root)
    if (file_hash(root/'preparation.json')!=CAPACITY_PREPARATION_SHA
            or file_hash(inspection_path)!=inspection_hash):
        raise ValueError('registered capacity plan and coordinator inspection required')
    plan=json.loads((root/'preparation.json').read_text())
    item=json.loads(Path(inspection_path).read_text());state=item['State'];host=item['HostConfig']
    argv=plan['argv']
    devices={d['PathOnHost'] for d in host.get('Devices') or []}
    expected_devices={argv[i+1].split(':')[0] for i,v in enumerate(argv[:-1]) if v=='--device'}
    env={argv[i+1] for i,v in enumerate(argv[:-1]) if v=='-e'}
    binds={argv[i+1] for i,v in enumerate(argv[:-1]) if v=='-v'}
    if (len(container_id)!=64 or item['Id']!=container_id or item['Image']!=plan['image']
            or item['Name']!='/audio2tool-engine-capacity-v1'
            or state['Status']!='exited' or state['ExitCode']!=0 or state['OOMKilled']
            or state['Running'] or state['Restarting'] or state['Dead'] or state['Error']
            or host['NanoCpus']!=16*10**9 or host['Memory']!=128*1024**3
            or host['ShmSize']!=8*1024**3 or host['NetworkMode']!='host' or host['Privileged']
            or devices!=expected_devices or set(host['Binds'])!=binds
            or not env.issubset(set(item['Config']['Env']))
            or item['Config']['Cmd']!=argv[argv.index(plan['image'])+1:]):
        raise ValueError('exact successful registered NPU capacity container required')
    for name,pin in plan['source_files'].items():
        if file_hash(Path(plan['source_root'])/name)!=pin:raise ValueError('capacity source changed')
    run=root/'run';result=json.loads((run/'result.json').read_text());identity=json.loads((run/'identity.json').read_text())
    if (json.loads((run/'terminal.json').read_text())['exit_code']!=0 or result['status']!='passed'
            or result['model_loaded'] is not True or result['weight_updates']!=0
            or result['annotations_read'] is not False or identity['preflight_only'] is not False
            or result['identity_sha256']!=file_hash(run/'identity.json')
            or identity['versions']!=VERSIONS or identity['engine']!=ENGINE
            or identity['sampling']!=dict(temperature=0.,max_tokens=1,n=1,seed=17)):
        raise ValueError('real capacity generation required, not CPU preflight')
    for name,pin in identity['source_files'].items():
        if plan['source_files'].get(name)!=pin:raise ValueError('capacity runtime source differs')
    prior,_=prior_records(processor_run)
    if (len(result['records'])!=4 or {r['selection'] for r in result['records']}!=set(prior)):
        raise ValueError('all four actual boundary generations required')
    for record in result['records']:
        p=prior[record['selection']]
        if (record!=json.loads((run/(record['selection']+'.json')).read_text())
                or record['id']!=p['id'] or record['input_tokens']!=p['input_tokens']
                or record['input_ids_sha256']!=p['input_ids_sha256']
                or record['audio_count']!=len(p['audio_receipts']) or not record['input_matches_processor']
                or len(record['token_ids'])!=1 or type(record['token_ids'][0]) is not int
                or record['token_ids'][0]<0 or record['finish_reason'] not in ('stop','length')
                or not math.isfinite(record['engine_seconds']) or record['engine_seconds']<=0):
            raise ValueError('capacity boundary result differs')
    return dict(container_id=container_id,result_sha256=file_hash(run/'result.json'),
                inspection_sha256=inspection_hash,preparation_sha256=CAPACITY_PREPARATION_SHA)


def read_population(inputs,budgets):
    if file_hash(inputs)!=INPUT_SHA or file_hash(budgets)!=BUDGET_SHA:
        raise ValueError('registered full release population required')
    rows=[json.loads(l) for l in Path(inputs).open()]
    budgets=[json.loads(l) for l in Path(budgets).open()]
    if (len(rows)!=30733 or len({r['id'] for r in rows})!=30733
            or Counter(r['tier'] for r in rows)!=COUNTS
            or [r['id'] for r in rows]!=[b['id'] for b in budgets]):
        raise ValueError('complete ordered release-record coverage required')
    unique=Counter()
    for row,budget in zip(rows,budgets,strict=True):
        validate(row['input']);unique[digest(row['input'])]+=1
        if (row['split']!='external_test' or row['tier']!=budget['tier']
                or row['input']['mode']!=budget['mode']
                or len(row['input']['audios'])!=len(budget['frames'])
                or budget['input_tokens']+1024>32768):
            raise ValueError('full audio/context budget required')
    if len(unique)!=30731 or sorted(v for v in unique.values() if v>1)!=[2,2]:
        raise ValueError('retain both known duplicate release pairs')
    return rows,budgets


def accept_generation(responses,expected_ids):
    if len(responses)!=1 or not responses[0].finished or len(responses[0].outputs)!=1:
        raise ValueError('one complete engine generation required')
    response=responses[0];answer=response.outputs[0]
    if list(response.prompt_token_ids)!=expected_ids:
        raise ValueError('engine expanded input IDs differ from actual HF input')
    if (answer.finish_reason not in ('stop','length') or type(answer.text) is not str
            or not 1<=len(answer.token_ids)<=1024
            or any(type(t) is not int or t<0 for t in answer.token_ids)
            or (answer.finish_reason=='length' and len(answer.token_ids)!=1024)):
        raise ValueError('registered 1024-token generation contract differs')
    # Preserve raw text, even malformed JSON or an empty decoded EOS response.
    return dict(raw_text=answer.text,error=None,token_ids=list(answer.token_ids),
        finish_reason=answer.finish_reason,length_limited=answer.finish_reason=='length',
        input_tokens=len(expected_ids),input_ids_sha256=digest(expected_ids))


def write_once(path,record):
    """Exclusive durable records; an interrupted write is not silently retried."""
    with Path(path).open('x') as handle:
        json.dump(record,handle,ensure_ascii=False,allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())


class Ledger:
    def __init__(self,root):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=False)
        (self.root/'attempts').mkdir();(self.root/'records').mkdir()

    def begin(self,row,request,receipts,expected_ids):
        record=dict(id=row['id'],tier=row['tier'],request_sha256=digest(row),
            input_sha256=digest(row['input']),prompt_sha256=hashlib.sha256(request['prompt'].encode()).hexdigest(),
            audio_receipts=receipts,input_tokens=len(expected_ids),input_ids_sha256=digest(expected_ids))
        write_once(self.root/'attempts'/(digest(row['id'])+'.json'),record)
        return record

    def finish(self,attempt,response,seconds):
        saved=json.loads((self.root/'attempts'/(digest(attempt['id'])+'.json')).read_text())
        if saved!=attempt or not math.isfinite(seconds) or seconds<0:
            raise ValueError('original attempt and finite elapsed time required')
        if (response.get('input_tokens')!=attempt['input_tokens']
                or response.get('input_ids_sha256')!=attempt['input_ids_sha256']
                or set(response)&set(attempt)-{'input_tokens','input_ids_sha256'}):
            raise ValueError('response identity differs from attempt')
        record=dict(**attempt,**{k:v for k,v in response.items() if k not in attempt},seconds=seconds)
        write_once(self.root/'records'/(digest(attempt['id'])+'.json'),record)
        return record

    def failure(self,attempt,exc,seconds):
        # Stop caller execution after persisting the actual failed attempt;
        # unattempted remaining rows must not become synthetic error predictions.
        return self.finish(attempt,dict(raw_text='',error=repr(exc),token_ids=[],finish_reason='runtime_error',
            length_limited=False,input_tokens=attempt['input_tokens'],input_ids_sha256=attempt['input_ids_sha256']),seconds)

    def complete(self,rows):
        names={digest(r['id'])+'.json' for r in rows}
        if (not rows or len(names)!=len(rows)
                or {p.name for p in (self.root/'attempts').iterdir()}!=names
                or {p.name for p in (self.root/'records').iterdir()}!=names):
            raise ValueError('complete unique attempts and terminal records required')
        records=[]
        for row in rows:
            name=digest(row['id'])+'.json'
            attempt=json.loads((self.root/'attempts'/name).read_text())
            record=json.loads((self.root/'records'/name).read_text())
            if (attempt['id']!=row['id'] or attempt['request_sha256']!=digest(row)
                    or attempt['input_sha256']!=digest(row['input'])
                    or any(record.get(k)!=v for k,v in attempt.items())
                    or record['error'] is not None):
                raise ValueError('complete successful unmodified generation ledger required')
            records.append(record)
        path=self.root/'predictions.jsonl'
        with path.open('x') as handle:
            for r in records:
                # This is the exact strict scorer interface; provenance stays
                # in terminal records, authenticated separately at acceptance.
                handle.write(json.dumps({k:r[k] for k in ('id','raw_text','error')},ensure_ascii=False)+'\n')
            handle.flush();os.fsync(handle.fileno())
        return dict(records=len(records),predictions_sha256=file_hash(path),
                    length_limited=sum(r['length_limited'] for r in records))
