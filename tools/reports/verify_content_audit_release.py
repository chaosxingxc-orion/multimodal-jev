"""Actual full-proof differential and reuse timing, no NPU dispatch."""
import argparse,json,time
from pathlib import Path
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.context_content_audit_engine import ContentAuditEngine
from tools.reports.prepare_objective_training_candidate import read_bound
from tools.reports.prepare_context_reference_job import collected_audit_config
from multimodal_jev.source_snapshot import verify


def run(c,p):
    snapshot=c['source_snapshot']
    if Path(__file__).resolve().parents[2]!=Path(snapshot['root']).resolve():
        raise ValueError('execute full proof from its bound isolated source snapshot')
    verify(snapshot['root'],manifest_sha256=snapshot['manifest_sha256'],io_workers=c['io_workers'])
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c)
    try:
        source=json.loads(read_bound(c['source']));parent=json.loads(read_bound(c['reference']))
        expected_source=json.loads(Path(source['source_plan']).read_text())
        expected_reference=json.loads(read_bound(c['expected_reference']))
        spec=dict(parent['reference_job']);spec.update({k:parent[k] for k in ('collection','results_audit_output')})
        leaf=collected_audit_config(spec)
        with ContentAuditEngine(cpu_workers=c['cpu_workers'],io_workers=c['io_workers']) as engine:
            timings={}
            for name in ('source','reference'):
                timings[name]=[]
                for i in range(2):
                    start=time.perf_counter()
                    actual=engine.reconstruct(source) if name=='source' else engine.replay(leaf,expected_reference['provenance'])
                    elapsed=time.perf_counter()-start
                    if actual!=(expected_source if name=='source' else expected_reference):raise ValueError('full '+name+' proof differs')
                    timings[name].append(elapsed);commit_json(root/f'{name}-{i}.json',dict(seconds=elapsed,exact=True))
            events=engine.statistics()
            engine.verify()
        verify(snapshot['root'],manifest_sha256=snapshot['manifest_sha256'],io_workers=c['io_workers'])
        if [v['cache_hit'] for v in events]!=[False,True,False,True]:raise ValueError('exact full proof reuse required')
        report=dict(status='full_source_reference_proof_reuse_verified',timings=timings,events=events,
            full_proofs_equal=True,model_calls=0,npu_executed=False,source_snapshot=snapshot,
            training_dispatch_allowed=False)
        commit_json(root/'report.json',report);commit_json(root/'terminal.json',dict(exit_code=0,status=report['status']));return report
    except BaseException as e:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(e)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,prov=load_config(a.config,a.local);print(json.dumps(run(c,prov),indent=2))
