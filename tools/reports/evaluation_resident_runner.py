"""Run configured TP1 evaluation workers once; collect all CPU audit tails."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from multimodal_jev.local_checkpoint import commit_json


def run(c, config_path):
    from context_process_supervisor import stop_owned
    root=Path(c['output']);root.mkdir(exist_ok=False);start=time.monotonic()
    workers=[];logs=[]
    try:
        for rank,device in enumerate(c['devices']):
            cache=Path(c['runtime_cache'])/f'rank-{rank}';cache.mkdir(exist_ok=False)
            env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS=str(c['torch_threads']))
            for key in ('TMPDIR','HF_HOME','XDG_CACHE_HOME','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR',
                        'ASCEND_WORK_PATH','ASCEND_CACHE_PATH','VLLM_RPC_BASE_PATH'):env[key]=str(cache)
            log=(root/f'worker-{rank}.log').open('x');logs.append(log)
            workers.append(subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('evaluation_resident_worker.py')),
                '--config',str(config_path),'--rank',str(rank)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        codes=[p.wait() for p in workers]
        if any(codes):raise ValueError('resident worker failed: '+str(codes))
        reports=[json.loads((root/f'worker-{rank}/result.json').read_text()) for rank in range(len(workers))]
        counts=[sum(r['points'][i]['recordings'] for r in reports) for i in range(len(c['points']))]
        if counts!=[c['limit'] or c['expected_population']]*len(c['points']):raise ValueError('full campaign coverage differs')
        result=dict(wall_seconds=time.monotonic()-start,counts=counts,workers=reports,
                    formal_admitted=False,scope='container start through all worker audit/shutdown; collection measured by controller')
        commit_json(root/'result.json',result);commit_json(root/'terminal.json',dict(exit_code=0))
        return result
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),seconds=time.monotonic()-start));raise
    finally:
        for p in workers:stop_owned(p,30.)
        for log in logs:log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    print(json.dumps(run(json.loads(Path(a.config).read_text()),a.config),indent=2))
