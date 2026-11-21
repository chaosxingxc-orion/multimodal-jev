"""Independent complete-output comparison and campaign timing report.

Does not score target gold, resume a formal queue, or extrapolate finite panels.
Old container times are a partial legacy baseline, not full-controller timing.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.evaluation_resident_worker import compare_records


def read(path):return json.loads(Path(path).read_text())


def audit(c):
    begin=time.monotonic();root=Path(c['run']);config=read(root/'experiment.json')
    controller=read(root/'terminal.json');inspect=read(root/'inspect.json');state=inspect['State']
    if (controller['exit_code'] or state['ExitCode'] or state['OOMKilled'] or state['Status']!='exited'):
        raise ValueError('completed successful resident candidate required')
    collection=root/'collected'
    if read(collection/'terminal.json')['exit_code']:raise ValueError('runner failed')
    results=[];old_total=0.;stage_work={};stage_worker_max={};inputs_bytes=0
    expected=config['limit'] or config['expected_population']
    for rank in range(len(config['devices'])):
        report=read(collection/f'worker-{rank}/result.json')
        if read(collection/f'worker-{rank}/terminal.json')['exit_code']:raise ValueError('worker failed')
        if report['model_initializations']!=1:raise ValueError('one engine initialization per worker required')
        inputs_bytes+=report.get('resident_bytes',0)
        for stage,seconds in report['timing']['stage_work_seconds'].items():
            stage_work[stage]=stage_work.get(stage,0)+seconds
            stage_worker_max[stage]=max(stage_worker_max.get(stage,0),seconds)
    for index,point in enumerate(config['points']):
        records=[]
        for rank in range(len(config['devices'])):
            job=collection/f'worker-{rank}'/point['job_id']
            records.extend(read(job/'predictions.json')['records'])
            if index and config['points'][index-1]['policy_id']!=point['policy_id']:
                switch=read(job/'switch.json')
                if (switch['policy_id']!=point['policy_id'] or switch['native_loaded_tensors']!=112
                        or switch['head_marker_exact'] is not True):raise ValueError('actual adapter/head activation missing')
        if len(records)!=expected or len({r['id'] for r in records})!=expected:
            raise ValueError('complete unique campaign population required')
        prior=c['prior'][point['job_id']];raw=Path(prior['predictions']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=point['predictions']['sha256']:
            raise ValueError('collected old predictions differ from candidate reference')
        old=json.loads(raw)['records']
        ids={r['id'] for r in records};selected=[r for r in old if r['id'] in ids]
        proof=compare_records(records,selected,point['policy_id'])
        if not proof['exact']:raise ValueError('same-policy full outputs differ')
        if config['limit'] is None and len(old)!=len(records):raise ValueError('full old population required')
        prior_state=read(prior['inspect'])['State']
        if prior_state['ExitCode'] or prior_state['OOMKilled']:raise ValueError('old run not successful')
        seconds=(datetime.fromisoformat(prior_state['FinishedAt'].replace('Z','+00:00'))-
                 datetime.fromisoformat(prior_state['StartedAt'].replace('Z','+00:00'))).total_seconds()
        per_worker=[]
        for rank in range(len(config['devices'])):
            events=read(collection/f'worker-{rank}/result.json')['timing'].get('events',[])
            per_worker.append(sum(e['seconds'] for e in events if e['stage']=='inference' and e['job_id']==point['job_id']))
        old_total+=seconds;results.append(dict(job_id=point['job_id'],parity=proof,old_container_seconds=seconds,
            candidate_max_worker_inference_seconds=max(per_worker),
            old_client_seconds=read(prior['inference_result'])['seconds'] if prior.get('inference_result') else None))
    total_inputs=sum(read(collection/f'worker-{r}/result.json')['input_loads'] for r in range(len(config['devices'])))
    if total_inputs!=expected:raise ValueError('one input load per sample required')
    seconds=(datetime.fromisoformat(state['FinishedAt'].replace('Z','+00:00'))-
             datetime.fromisoformat(state['StartedAt'].replace('Z','+00:00'))).total_seconds()
    result=dict(status='finite_resident_campaign_independently_audited' if config['limit'] else 'full_resident_campaign_independently_audited',
        points=results,unique_inputs=expected,predictions=expected*len(results),input_loads=total_inputs,
        model_initializations=len(config['devices']),old_container_sum_seconds=old_total,
        candidate_container_seconds=seconds,candidate_controller_seconds=controller['wall_seconds'],
        collection_seconds=controller['stages']['collection'],stage_work_seconds=stage_work,
        stage_max_worker_seconds=stage_worker_max,resident_input_bytes=inputs_bytes,
        container_speedup=None if config['limit'] else old_total/seconds,
        end_to_end_speedup=None,end_to_end_missing='old controller preparation/collection/audit wall intervals unavailable',
        independent_audit_seconds=time.monotonic()-begin,formal_admitted=False,performance_10x_passed=False)
    commit_json(c['output'],result)
    if c.get('markdown'):
        lines=['# 四卡完整评测复用实测','',
            f"状态：{result['status']}。正式评测准入：未通过；不声明端到端10倍。",'',
            '| 项目 | 实测 |','| --- | ---: |',
            f"| 完整预测数 | {result['predictions']} |",
            f"| 输入实际加载次数 | {total_inputs} |",
            f"| 模型初始化次数 | {len(config['devices'])} |",
            f"| 旧两轮容器合计 | {old_total:.2f} 秒 |",
            f"| 新任务组容器 | {seconds:.2f} 秒 |",
            f"| 新控制器含准备/收集 | {controller['wall_seconds']:.2f} 秒 |",
            f"| 收集 | {controller['stages']['collection']:.2f} 秒 |",
            f"| 独立审核 | {result['independent_audit_seconds']:.2f} 秒 |",'']
        if result['container_speedup'] is not None:
            lines += [f"同容器边界速度比：{result['container_speedup']:.3f}×，节省{(1-seconds/old_total)*100:.2f}%。",'']
        else:lines += ['有限面板与旧全量规模不同，不计算速度比。','']
        lines+=['## 阶段分解','', '| 阶段 | 四进程工作秒数合计 | 单进程最大工作秒数 |',
                '| --- | ---: | ---: |']
        lines += [f'| {stage} | {value:.2f} | {stage_worker_max[stage]:.2f} |' for stage,value in stage_work.items()]
        lines+=['','并行工作秒数不可求和冒充wall time。旧完整控制器边界缺失，端到端速度比为unknown。','',
                '所有检查点的全量prob/logit均与同策略旧输出精确一致，阈值判决与Top1保持。',
                '这是效率/一致性报告，不是新的TEST或迁移成绩。']
        Path(c['markdown']).write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,_=load_config(a.config,a.local);print(json.dumps(audit(c),indent=2))
