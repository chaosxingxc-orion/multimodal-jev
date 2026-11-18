"""Scope the original objective lifecycle to the admitted parallel source audit.

No validator is removed, no stored PASS is trusted, and no module globals are
patched. Imported function/module dependencies are cloned into a private graph.
The model/runtime package and reference numerical checks are unchanged.
"""
import importlib
import inspect
from types import FunctionType,SimpleNamespace

MODULES=(
 'context_infonce_selection','context_infonce_stage2_source','prepare_context_stage2_runtime',
 'prepare_context_train_parity_job','audit_context_train_parity',
 'prepare_context_reference_job','context_reference_job_operations',
 'prepare_context_cached_training','context_cached_training_operations',
 'prepare_objective_training_candidate','audit_objective_training',
 'context_objective_training_operations','prepare_context_cached_recovery',
 'prepare_objective_recovery','audit_context_cached_recovery','audit_objective_recovery',
 'context_objective_recovery_operations','run_objective_acceptance',
)


def clone_graph(modules,overrides):
    """Clone module-owned functions preserving signatures and shared globals."""
    views={m.__name__:SimpleNamespace(**vars(m)) for m in modules}
    namespaces={m.__name__:dict(vars(m)) for m in modules};functions={}
    for m in modules:
        for name,value in vars(m).items():
            if inspect.isfunction(value) and value.__module__==m.__name__:
                fn=FunctionType(value.__code__,namespaces[m.__name__],value.__name__,value.__defaults__,value.__closure__)
                fn.__kwdefaults__=value.__kwdefaults__;fn.__module__=value.__module__
                functions[value]=overrides.get((m.__name__,name),fn)
    for m in modules:
        ns=namespaces[m.__name__]
        for name,value in list(ns.items()):
            if inspect.ismodule(value) and value.__name__ in views:value=views[value.__name__]
            elif inspect.isfunction(value) and value in functions:value=functions[value]
            ns[name]=value;setattr(views[m.__name__],name,value)
    return views


def implementation(engine):
    modules=[importlib.import_module('tools.reports.'+n) for n in MODULES]
    views=clone_graph(modules,{('tools.reports.context_infonce_selection','reconstruct_plan'):engine.reconstruct})
    return views['tools.reports.run_objective_acceptance']
