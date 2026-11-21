"""Actor-only extension entry: preserve the sealed engine tree and its identity."""
import argparse
import importlib.util
import json
from pathlib import Path


def load(name,path,digest):
    import hashlib
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError('actor component changed: '+name)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--preflight',action='store_true');g.add_argument('--output',type=Path)
    a=p.parse_args();root=a.config.resolve().parent;c=json.loads(a.config.read_bytes())
    if Path(__file__).resolve()!=root/'view-entry.py':raise ValueError('packaged actor entry required')
    def component(name,path):return load(name,path,c['pins'][str(path)])
    boot=component('sealed_bootstrap',root/'objective-extension/bootstrap.py')
    plugin=boot.install(c)
    import importlib,sys
    import multimodal_jev
    name='multimodal_jev.context_task_reward'
    helper=Path(c['source_root'])/'src/multimodal_jev/context_task_reward.py'
    reward=root/'task-reward.py'
    if helper.exists():
        if helper.read_bytes()!=reward.read_bytes():raise ValueError('conflicting inherited task reward')
        importlib.import_module(name)
    else:
        module=component(name,reward);sys.modules[name]=module
        setattr(multimodal_jev,'context_task_reward',module)
    from multimodal_jev.objective_plugin import ObjectivePlugin
    from multimodal_jev.context_stage_lora import admit_stage2
    from tools.reports.context_dual_runtime_candidate import verify_runtime
    import inspect
    from multimodal_jev.cached_context_preparation import CachedContextPreparer
    if 'prompt_layout' not in inspect.signature(CachedContextPreparer).parameters:
        raise ValueError('forward runtime lacks versioned prompt layout')
    verify_runtime(c);admit_stage2(c['stage_lora_transition'],c['source_contract'])
    ObjectivePlugin(c['objective_plugin'])
    view=component('view_contract',root/'view-runtime.py')
    trainer=component('view_trainer',root/'view-trainer.py')
    view.indices(c)
    if 'objective_config' in c:
        view.editable_objective(c)
    else:
        view.validate_inherited_objective(c)
    if c['view_window']=='formal' and (c['training_dispatch_allowed'] is not True or set(c.get('view_admission',{}))!={'prefix','recovery','tail'}):
        raise ValueError('formal dispatch requires independent finite receipts')
    if a.preflight:return
    a.output.mkdir(parents=True,exist_ok=True)
    trainer.run(c,a.output,view=view,actor_loader=plugin.load_actor)


if __name__=='__main__':main()
