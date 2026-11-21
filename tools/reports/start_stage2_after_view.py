"""Bind an audited Stage2 dataset to new formal windows, then launch and observe."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.context_stage2_view_pipeline import pin
from tools.reports.watch_stage2_first_losses import Events


def read(path):
    return json.loads(Path(path).read_text())


def await_release(root, timeout_inputs):
    watch = Events()
    try:
        while True:
            for parent in {p.parent for p in timeout_inputs} | {root, root.parent}:
                watch.add(parent)
            finished = root / 'terminal.json'
            if finished.is_file() and read(finished)['exit_code'] != 0:
                raise ValueError(f'dataset producer failed: {finished}')
            if finished.is_file() and all(p.is_file() for p in timeout_inputs):
                return read(finished)
            watch.wait()
    finally:
        watch.close()


def bind(config, terminal):
    source = Path(config['dataset_producer'])
    receipt = source / 'dataset-audit.json'
    release = read(receipt)
    if (terminal['status'] != 'stage2_dataset_release_audited'
            or terminal['receipt_sha256'] != file_hash(receipt)
            or terminal['training_dispatched'] is not False):
        raise ValueError('audited dataset release required')
    ds, _ = load_config(config['dataset_template'], source / 'dataset-local.json')
    ds.update(manifest_sha256=release['manifest_sha256'], stage2_config=None,
              stage2_output=config['rebinding_output'],
              receipts=dict(ds['receipts'], bind=config['rebinding_receipt']))
    if (ds['stage1_expected']['policy_id'] != config['source_policy_id']
            or ds['batch_config']['training_devices'] != 6 or ds['batch_config']['inference_instances'] != 2
            or ds['output'] != config['view']):
        raise ValueError('require this Stage1 endpoint and the released DP6 view')
    original = read(source/'dataset-local.json')['macros']
    parent = Path(original['STAGE2_INPUT_CONFIG']['path']).parent
    runtime = json.loads((parent/'RUNTIME-SOURCES.json').read_text())
    extension_path = Path(config['extension_report'])
    extension = read(extension_path)
    window_path = Path(config['window_report'])
    window = read(window_path)
    for result, count in ((extension, 47), (window, 9)):
        if (result['counts'] != dict(tests=count, errors=0, failures=0, skipped=0)
                or result['state']['ExitCode'] or result['state']['OOMKilled']
                or result['image'] != config['image']):
            raise ValueError('successful fixed image CPU reports required')
    ext_sources = Path(extension['collection'])/'package/FILES.json'
    actual = read(ext_sources)
    if (file_hash(ext_sources) != extension['source_manifest_sha256']
            or any(actual.get(k) != v for k, v in runtime.items())):
        raise ValueError('extension report must cover every active forward source')
    prepared = dict(config['objective_parent'], parent=dict(
        output=str(parent), manifest_sha256=file_hash(parent/'PREPARED.json'),
        parent=ds['reference_job']['parent'], reference_job=ds['reference_job']),
        extension_source=config['extension_source'], extension_cpu_report=pin(extension_path))
    experiment = dict(config['pipeline'], dataset=ds, objective_parent=prepared)
    experiment['window_operations'] = dict(experiment['window_operations'], image_cpu=pin(window_path))
    return experiment


def run(config, out):
    out.mkdir(exist_ok=False)
    commit_json(out/'config.json',config)
    release = await_release(Path(config['dataset_producer']), [Path(config[k]) for k in
                            ('extension_report','window_report')])
    pipeline = bind(config,release)
    cfg = out/'pipeline-local.json'
    commit_json(cfg,dict(schema='jev-local-macros-v1',macros={'STAGE2_VIEW_PIPELINE':pipeline}))
    formal = pipeline['windows']['formal']
    observer = out/'observer-local.json'
    commit_json(observer,dict(schema='jev-local-macros-v1',macros={'STAGE2_FIRST_LOSSES':dict(
        output=config['observer_output'], controller=pipeline['output'], host=pipeline['deployment']['host'],
        formal_operations=formal['operations_root'],formal_tag=formal['tag'], require_view_topology=True)}))
    subprocess.run([sys.executable,'-B','-m','tools.reports.context_stage2_view_pipeline',
                    '--config',config['pipeline_template'],'--local',str(cfg),'--detach'],check=True)
    with (out/'observer.log').open('x') as log:
        child = subprocess.Popen([sys.executable,'-B',config['observer_source'],
            '--config',config['observer_template'],'--local',str(observer)],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    commit_json(out/'started.json',dict(observer_pid=child.pid))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(read(args.config),args.output)
