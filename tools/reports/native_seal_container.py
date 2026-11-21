"""Run a fixed stage on an isolated native evaluation container."""
import argparse
import json
import os
from pathlib import Path
import sys

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.source_snapshot import verify
from tools.reports.prepare_native_seal_cache import run as cache_run
from tools.reports.run_native_seal_endpoint import run as endpoint_run


def endpoint_config(c):
    endpoint = dict(c['endpoint'])
    for key, path in c.get('evidence', {}).items():
        endpoint[key] = dict(path=path, sha256=file_hash(path))
    return endpoint


def run_services(c, args):
    from context_process_supervisor import supervise_services
    specs = []
    def env(name, ordinal):
        path = Path(os.environ['TMPDIR'])/name
        path.mkdir()
        result = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=ordinal)
        for key in ('TMPDIR','HF_HOME','XDG_CACHE_HOME','TORCHINDUCTOR_CACHE_DIR',
                    'TRITON_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','VLLM_RPC_BASE_PATH'):
            result[key] = str(path)
        return result

    for service in c['services']:
        config = json.loads(Path(service['config']).read_text())
        output = Path(service['output'])
        exchange = config['exchange']
        Path(exchange).mkdir(parents=True)
        specs.append(dict(name=service['name'], ready_path=output/'ready.json',
                          socket_path=Path(config['socket_path']),
                          env=env(service['name'],config['policy']['process_device_scope']['visible_ordinal']),
                          command=[sys.executable, '-B','-m','tools.reports.context_replica_runtime',
                                   '--config',service['config'],'--output',str(output)]))
    result = supervise_services(services=specs,
        actor_command=[sys.executable,'-B','-m','tools.reports.native_seal_container',
                       '--config',args.config,'--local',args.local,'--client'],
        actor_env=env('client',''),output=c['supervision'])
    if not result['passed']:
        raise ValueError('exact inference service/client lifecycle failed')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    p.add_argument('--client', action='store_true')
    a = p.parse_args()
    c, provenance = load_config(a.config, a.local)
    source = c['source']
    verify(source['root'], manifest_sha256=source['sha256'], io_workers=c['io_workers'])
    if c['phase'] == 'model_provider_probe':
        from tools.reports.probe_audio_model_provider import run
        result = run(c['probe'])
    elif c['phase'] == 'tpo_score':
        from tools.reports.evaluate_tpo_endpoints import collect
        result = collect(c)
    elif c['phase'] == 'tpo_eval':
        from tools.reports.evaluate_tpo_endpoints import run
        result = run(c)
    elif c['phase'] == 'joint_tpo':
        from tools.reports.joint_tpo_window import run
        result = run(c)
    elif c['phase'] == 'tpo_train':
        from tools.reports.tpo_training_window import run
        result = run(c)
    elif c['phase'] in ('publish_eval_cache', 'preflight_eval_cache'):
        from multimodal_jev.cached_context_preparation import processor_cache_identity
        from tools.reports.native_evaluation_cache import materialize, preflight
        cache = c['cache']
        if processor_cache_identity(cache['processor']['policy']) != cache['processor']['sha256']:
            raise ValueError('evaluation processor differs from the sealed cache identity')
        if c['phase'] == 'publish_eval_cache':
            result = materialize(cache, provenance, run=cache_run, io_workers=c['io_workers'])
        else:
            result = preflight(cache, io_workers=c['io_workers'])
    elif c['phase'] in ('encode', 'verify'):
        cache = c['cache']
        result = cache_run(cache, provenance, verify=c['phase']=='verify')
        if c['phase']=='verify':
            commit_json(Path(cache['output'])/'VERIFIED.json', result)
    elif c.get('services') and not a.client:
        result = run_services(c, a)
    else:
        result = endpoint_run(endpoint_config(c), provenance, c['phase'])
    if c['phase'] in ('publish_eval_cache', 'preflight_eval_cache'):
        commit_json(Path(c['result_path']), dict(result, cache_root=c['cache']['output']))
    print(json.dumps({k: v for k,v in result.items() if k!='records'}))


if __name__=='__main__':
    main()
