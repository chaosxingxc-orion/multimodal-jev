"""Independent post-collection audit of one exact view package, no model or dispatch."""
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.stage2_view_runtime import pinned,indices,layout
from tools.reports.audit_context_cached_training import audit_collected
from tools.reports.audit_context_replica_training import audit_outputs
from tools.reports.audit_objective_training import packaged_plugin
from tools.reports.audit_objective_traces import audit as replay_loss
from tools.reports.context_stage2_release_components import command


def audit(c,p,*,trace_auditor=replay_loss):
    def check(spec,_):
        root=Path(spec['output']);m=pinned(root/'PREPARED.json',spec['manifest_sha256'])
        files={str(f.relative_to(root)):file_hash(f) for f in root.rglob('*') if f.is_file() and f!=root/'PREPARED.json'}
        if root.is_symlink() or any(f.is_symlink() for f in root.rglob('*')) or files!=m['files'] or m['status']!='stage2_view_package_prepared_not_admitted':
            raise ValueError('exact immutable view package required')
        return dict(manifest_sha256=spec['manifest_sha256'],dataset_sha256=m['dataset_sha256'])
    def outputs(package,collection,*,reference_root):
        cfg=json.loads((package/'config.json').read_bytes())
        indices(cfg);kind=cfg['view_window'];run=collection/'run'
        world=layout(cfg)['dp_replicas']
        result=audit_outputs(package,collection,initial_policy=package/'initial/policy',reference_root=reference_root)
        if kind in ('prefix','recovery'):
            traces=[];paths=set()
            for i in range(cfg['start_update'],cfg['stop_update']):
                groups=run/f'groups-{i:04d}.pt';ranks=[]
                for rank in range(world):
                    path=run/f'rank-{rank}/objective-traces/objective-{i:04d}.json';paths.add(path)
                    ranks.append(dict(rank=rank,path=str(path),sha256=file_hash(path)))
                traces.append(dict(update=i,groups=dict(path=str(groups),sha256=file_hash(groups)),traces=ranks))
            if set(run.glob('rank-*/objective-traces/**/*'))!=paths:raise ValueError('unexpected trace membership')
            result['trace_audit']=trace_auditor(dict(objective_plugin=packaged_plugin(package,cfg),world_size=world,
                local_batch=layout(cfg)['local_effective_batch'],microbatch=cfg['microbatch'],updates=traces,loss_atol=c['objective_loss_atol'],output='unused'),p)
        if kind=='prefix':result['step1_commit_sha256']=file_hash(run/'checkpoint-0001-committed.json')
        return dict(result,status='stage2_view_'+kind+'_outputs_audited',dataset_sha256=cfg['schedule_sha256'],lineage=cfg['view_lineage'],
            real_exposures=sum(u['real_queries'] for u in result['updates']),efficacy_evaluated=False)
    result=audit_collected(c,p,package_check=check,build_command=command,check_outputs=outputs)
    if result['status']=='stage2_view_recovery_outputs_audited':
        from tools.reports.audit_objective_recovery import compare
        prefix=c['prefix'];proof=pinned(prefix['audit']['path'],prefix['audit']['sha256'])
        run=Path(prefix['run']);pkg=Path(c['training_package']['output'])
        if (proof['status']!='stage2_view_prefix_outputs_audited' or proof['container_id']==result['container_id']
                or proof['dataset_sha256']!=result['dataset_sha256'] or proof['step1_commit_sha256']!=file_hash(pkg/'resume/checkpoint-0001-committed.json')
                or proof['step1_commit_sha256']!=file_hash(run/'checkpoint-0001-committed.json')):
            raise ValueError('same independently audited continuous prefix and fresh replay required')
        result.update(prefix_audit_sha256=prefix['audit']['sha256'],replay=compare(run,Path(c['collection'])/'run',world_size=result['world_size']),fresh_process_replay_verified=True)
    return result


if __name__=='__main__':
    import argparse
    from multimodal_jev.local_config import load_config
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--local',required=True);x=a.parse_args()
    c,p=load_config(x.config,x.local);commit_json(Path(c['output']),audit(c,p))
