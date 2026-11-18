"""New-package trainer entry: editable objective file, original runtime loop."""
import argparse
import ast
from contextvars import ContextVar
import json
from pathlib import Path
from types import FunctionType

from multimodal_jev.objective_plugin import ObjectivePlugin
from tools.reports.context_cached_reference_trainer import generate as original_generate


def load_actor(source_root, plugin_config):
    from tools.reports import context_dual_actor_candidate as original
    plugin=ObjectivePlugin(plugin_config)
    trace={'writer':None}
    detail_slot=ContextVar('jev_objective_metric_detail',default=None)
    def objective(*args,**kwargs):
        loss,detail=plugin.hybrid_loss(*args,**kwargs,trace_observer=trace['writer'])
        detail_slot.set(detail)
        return loss,detail
    scoped=FunctionType(original.load.__code__,dict(original.load.__globals__,
        _loaded=None,hybrid_loss=objective))
    actor=scoped(source_root)
    native_loss=actor.registered_loss
    def registered_loss(*args,**kwargs):
        token=detail_slot.set(None)
        try:
            loss,metrics=native_loss(*args,**kwargs)
            detail=detail_slot.get()
            if detail is None:raise ValueError('objective did not report metrics for this forward')
            for key,value in detail.items():
                if key in ('rl','auxiliary','kl'):continue
                if type(value) in (int,float):
                    if not isinstance(key,str) or not key:
                        raise ValueError('custom numeric metrics require nonempty string names')
                    metrics['jev/objective/'+key]=value
            return loss,metrics
        finally:
            detail_slot.reset(token)
    actor.registered_loss=registered_loss
    def begin_trace(root,update):
        from multimodal_jev.objective_trace import FiniteTraceWriter
        if trace['writer'] is None:trace['writer']=FiniteTraceWriter(root)
        trace['writer'].begin(update)
    def finish_trace():
        return trace['writer'].finish()
    actor.begin_objective_trace=begin_trace
    actor.finish_objective_trace=finish_trace
    actor.objective_plugin_identity=plugin.identity
    return actor


def generate(source_root, *, formal=False):
    source=original_generate(source_root,formal=formal)
    edits=[
        ('def run(c,root):\n',
         "def run(c,root):\n"
         "    if type(c.get('objective_trace',False)) is not bool:\n"
         "        raise ValueError('explicit Boolean objective_trace required')\n"
         "    if c.get('objective_trace') and (c['experiment_kind']!='engineering_prefix' or c['stop_update']-c['start_update']>2):\n"
         "        raise ValueError('full objective traces limited to two-update prefix admission')\n"),
        ('    from tools.reports.context_dual_actor_candidate import load as load_dual_actor',
         '    from tools.reports.context_plugin_trainer import load_actor as load_dual_actor'),
        ("    dual_actor=load_dual_actor(c['source_root'])",
         "    dual_actor=load_dual_actor(c['source_root'],c['objective_plugin'])"),
        ("        contract['stage2_objective']=objective_contract",
         "        contract['stage2_objective']=objective_contract\n"
         "        contract['objective_plugin']=dual_actor.objective_plugin_identity"),
        ("                metrics=actor.update_policy(batch);torch.npu.synchronize();seconds=time.monotonic()-started",
         "                if c.get('objective_trace'):dual_actor.begin_objective_trace(out/'objective-traces',index)\n"
         "                metrics=actor.update_policy(batch);torch.npu.synchronize();seconds=time.monotonic()-started\n"
         "                if c.get('objective_trace'):dual_actor.finish_objective_trace()"),
    ]
    for before,after in edits:
        if source.count(before)!=1:raise ValueError('plugin trainer integration anchor changed')
        source=source.replace(before,after)
    ast.parse(source)
    return source


def run(config,output):
    # Validate before constructing a model or entering distributed execution.
    ObjectivePlugin(config['objective_plugin'])
    source=generate(config['source_root'],formal=config['experiment_kind']=='formal_source')
    scope={'__name__':__name__+'.derived'}
    exec(compile(source,'<objective-plugin-trainer>','exec'),scope)
    return scope['run'](config,Path(output))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(json.loads(args.config.read_text()),args.output)
