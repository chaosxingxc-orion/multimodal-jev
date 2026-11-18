"""Exact comparison of continuous and fresh-process update artifacts."""
import torch


def require_exact(left,right,path='root'):
    """Tensor dtype/shape/value and container structure must all match."""
    if type(left) is not type(right):raise ValueError('type differs: '+path)
    if isinstance(left,torch.Tensor):
        if (left.dtype!=right.dtype or left.shape!=right.shape or
                not torch.equal(left.cpu(),right.cpu())):raise ValueError('tensor differs: '+path)
    elif isinstance(left,dict):
        if left.keys()!=right.keys():raise ValueError('keys differ: '+path)
        for key in left:require_exact(left[key],right[key],path+'/'+str(key))
    elif isinstance(left,(list,tuple)):
        if len(left)!=len(right):raise ValueError('length differs: '+path)
        for index,(a,b) in enumerate(zip(left,right,strict=True)):
            require_exact(a,b,path+'/'+str(index))
    elif left!=right:raise ValueError('value differs: '+path)


def verify_update(continuous,resumed,*,step=2):
    """Both directories must already pass independent source/run collection.

    This checks the executed batch, sampled groups, all master/Adam shards,
    assembled execution weights and joint policy weights. Timing, cache hits
    and path-dependent activation receipts intentionally are not compared.
    """
    import json
    from pathlib import Path
    left,right=Path(continuous),Path(resumed)
    def read_json(root,name):return json.loads((root/name).read_text())
    def read_torch(root,name):return torch.load(root/name,map_location='cpu',weights_only=True)
    require_exact(read_json(left,f'batch-{step-1:04d}.json'),read_json(right,f'batch-{step-1:04d}.json'),'batch')
    require_exact(read_torch(left,f'groups-{step-1:04d}.pt'),read_torch(right,f'groups-{step-1:04d}.pt'),'groups')
    a=read_json(left,f'checkpoint-{step:04d}-committed.json')
    b=read_json(right,f'checkpoint-{step:04d}-committed.json')
    require_exact(a['contract'],b['contract'],'contract')
    if a['updates']!=step or b['updates']!=step:raise ValueError('update cursor differs')
    for rank in range(4):
        name=f'rank-{rank}/checkpoint-{step:04d}.pt'
        require_exact(read_torch(left,name),read_torch(right,name),'rank-'+str(rank))
        restored=read_json(right,f'rank-{rank}/restored.json')
        if restored!={'exact_masters':True,'exact_adam':True,'completed_updates':step-1}:
            raise ValueError('fresh-process restore receipt differs')
    require_exact(read_torch(left,a['execution_file']),read_torch(right,b['execution_file']),'execution')
    require_exact(read_torch(left,a['joint_policy_directory']+'/execution.pt'),
                  read_torch(right,b['joint_policy_directory']+'/execution.pt'),'joint_policy')
    first=read_json(left,f'checkpoint-{step-1:04d}-committed.json')
    batch=read_json(right,f'batch-{step-1:04d}.json')
    if batch['policy_id']!=first['joint_policy']['policy_id']:
        raise ValueError('resumed behavior not from previous committed actor')
    return dict(step=step,rank_shards=4,exact_masters=True,exact_adam=True,
                exact_samples=True,exact_execution=True,exact_joint_policy=True)
