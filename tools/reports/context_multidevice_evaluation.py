"""Inference-only device topology; one TP1 service per configured physical card."""
import inspect
from types import FunctionType


def topology(c):
    if set(c)!={'devices','service_batch_queries','waves_per_macro'}:
        raise ValueError('explicit evaluation devices, service batch and waves required')
    devices=c['devices'];batch=c['service_batch_queries'];waves=c['waves_per_macro']
    if (not isinstance(devices,list) or not 1<=len(devices)<=8
            or any(type(d)is not int or d<0 for d in devices) or len(set(devices))!=len(devices)
            or type(batch)is not int or batch not in (32,48)
            or type(waves)is not int or not 1<=waves<=16 or len(devices)*batch*waves>4096):
        raise ValueError('bounded unique devices and batch configuration required')
    mounted=sorted(devices)
    return dict(training_devices=[],inference_services=[dict(name=f'eval{i}',device=d) for i,d in enumerate(devices)],
        mounted_physical=mounted,hf_device=devices[0],hf_visible_ordinal=str(mounted.index(devices[0])),
        service_batch_queries=batch,macro_queries=len(devices)*batch*waves,waves_per_macro=waves)


def visibility(m):
    t=topology(m['evaluation_topology'])
    if m['training_devices']!=[] or m['inference_services']!=t['inference_services']:
        raise ValueError('inference-only topology differs')
    return '',[(s['name'],str(t['mounted_physical'].index(s['device']))) for s in t['inference_services']]


def transformed(fn, replacements, **scope):
    """Fail closed on changes in a frozen shared function; no module mutation."""
    source=inspect.getsource(fn)
    for before,after in replacements:
        if source.count(before)!=1:raise ValueError('shared evaluation anchor changed: '+before)
        source=source.replace(before,after)
    namespace=dict(fn.__globals__,**scope)
    exec(compile(source,'<configured-evaluation>','exec'),namespace)
    return namespace[fn.__name__]


def runner(m):
    from tools.reports import prepare_context_train_parity_job as parity
    from tools.reports import prepare_context_replica_selection as selection
    adapted=FunctionType(parity.runner.__code__,dict(parity.runner.__globals__,visibility=visibility))
    return FunctionType(selection.runner.__code__,dict(selection.runner.__globals__,parity_runner=adapted))(m)


def client_source(original):
    replacements=[
        ("client['max_queries']!=128", "client['max_queries']!=c['macro_queries']"),
        ("client['rollout_queries']!=32 or len(client['replicas'])!=2", "client['rollout_queries']!=c['evaluation_topology']['service_batch_queries'] or len(client['replicas'])!=len(c['evaluation_topology']['devices'])"),
        ("c['macro_queries']!=128", "c['macro_queries']!=len(c['evaluation_topology']['devices'])*c['evaluation_topology']['service_batch_queries']*c['evaluation_topology']['waves_per_macro']"),
        ("macro_queries=128,output=root/'inference'", "macro_queries=c['macro_queries'],output=root/'inference'"),
    ]
    for before,after in replacements:
        if original.count(before)!=1:raise ValueError('shared client anchor changed: '+before)
        original=original.replace(before,after)
    compile(original,'<configured-selection-client>','exec')
    return original


def command(c,m,config,*,producer=True):
    from tools.reports import context_cached_storage_operations as storage
    if not producer:raise ValueError('evaluation cannot dispatch training')
    visibility(m)
    begin="    train=m['training_devices'];services=m['inference_services']\n"
    source=inspect.getsource(storage.command);end='    # Transform only storage decisions.'
    start=source.index(begin);stop=source.index(end,start)
    ns=dict(storage.command.__globals__)
    exec(compile(source[:start]+source[stop:],'<evaluation-readonly-storage>','exec'),ns)
    return ns['command'](c,m,config,producer=True)


def audit_macros(*args, replicas, **kwargs):
    from tools.reports.audit_context_macro_evaluation import audit
    if type(replicas)is not int or not 1<=replicas<=8:raise ValueError('bounded replica count required')
    fn=transformed(audit,[('1<=macro_queries<=128','1<=macro_queries<=4096'),
        ("job['replica_id'] not in (0,1)","job['replica_id'] not in range(replica_count)")],replica_count=replicas)
    return fn(*args,**kwargs)
