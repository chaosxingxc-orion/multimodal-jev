"""Configuration-only entry for a release. No retries or implicit next stage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from multimodal_jev.context_execution import width
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.source_snapshot import verify

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = {
    'stage1': 'context-stage1-start-pipeline.json',
    'stage2_dataset': 'stage2-dataset.json',
    'stage2_dataset_dp6': 'stage2-dataset-dp6.json',
    'stage2_window':'stage2-view-window.json',
    'stage2_prepare': 'stage2-view-package.json',
    'stage2_prepare_dp6': 'stage2-view-package-dp6.json',
    'stage2_full': 'composed-stage2.json',
}


def stage1_entry(cfg, design):
    from multimodal_jev.context_topology_batch_plan import derive
    from multimodal_jev.stage_experiment_contract import source_curve
    from multimodal_jev.stage2_view_runtime import pinned

    package = cfg['package']
    batch = design['stage1']['batch_config']
    plan = derive(batch)
    cards = package['physical_devices']
    profiles = json.loads((Path(__file__).resolve().parents[2] /
        'configs/experiments/stage1-simple-entry-profiles-v1.json').read_bytes())
    mode = 'layout_bound' if 'stage1_layout' in package else 'legacy'
    profile = profiles[mode]
    if (not isinstance(cards, list) or any(type(d) is not int or d < 0 for d in cards)
            or len(cards) != len(set(cards)) or len(cards) != plan['dp_replicas']
            or plan['dp_replicas'] not in profile['dp_replicas']
            or batch['per_dp_replica_batch'] not in profile['per_dp_replica_batch']
            or batch['microbatch'] not in profile['microbatch']
            or plan['global_batch'] != design['stage1']['global_batch']
            or batch['inference_instances'] != 0
            or package['package'] != cfg['finite']['package']
            or package['remote_root'] != cfg['finite']['remote_root']):
        raise ValueError('physical topology, microbatch, global batch or finite package differs from entry profile')
    if mode == 'layout_bound':
        source_curve(design)
        layout_design = package['stage1_layout']['design']
        manifest = pinned(layout_design['path'], layout_design['sha256'])
        if manifest['stage1']['sha256'] != cfg['design_sha256']:
            raise ValueError('layout plan refers to a different Stage1 experiment')


def request(template, local):
    spec, identity = load_config(template, local)
    if (set(spec) != {'schema','stage','source_snapshot','io_workers','backend_sha256','driver_root'}
            or spec['schema'] != 'jev-simple-experiment-v1' or spec['stage'] not in BACKENDS):
        raise ValueError('registered simple experiment stage required')
    width(spec['io_workers'])
    source = spec['source_snapshot']
    if ROOT != Path(source['root']).resolve() or Path.cwd().resolve() != ROOT:
        raise ValueError('run this controller from its frozen source root')
    if not Path(spec['driver_root']).is_absolute():
        raise ValueError('absolute local driver root required')
    cfg, provenance = load_config(ROOT/'configs/portable/operations'/BACKENDS[spec['stage']],local)
    if provenance['resolved_semantic_sha256'] != spec['backend_sha256']:
        raise ValueError('backend configuration changed')
    stage = spec['stage']
    if stage == 'stage1':
        if 'io_workers' in cfg and cfg['io_workers']!=spec['io_workers']:raise ValueError('conflicting Stage1 I/O width')
        cfg=dict(cfg,io_workers=spec['io_workers'])
        raw=Path(cfg['package']['design']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=cfg['design_sha256']:raise ValueError('Stage1 experiment design changed')
        stage1_entry(cfg,json.loads(raw))
    elif stage == 'stage2_window':
        from tools.reports.run_stage2_view_window import request as check_window
        check_window(cfg)
    elif stage in ('stage2_prepare', 'stage2_prepare_dp6'):
        from tools.reports.prepare_stage2_view_package import validate
        validate(cfg)
    elif stage == 'stage2_full':
        from tools.reports.run_composed_stage2 import validate_request
        if cfg['source_snapshot']!=source:raise ValueError('Stage2 controller source differs')
        validate_request(cfg)
    else:
        from multimodal_jev.stage2_dataset_release import validate_execution
        validate_execution(cfg['execution'])
        if 'release' not in cfg['receipts']:raise ValueError('dataset release receipt required')
    driver=Path(spec['driver_root']).resolve();out=Path(cfg['output']).resolve()
    if driver.is_relative_to(out) or out.is_relative_to(driver) or driver.is_relative_to(ROOT):
        raise ValueError('driver must be disjoint from stage output and outside the snapshot')
    return spec, cfg, provenance, identity


def execute(spec,c,p):
    # Stage2 performs the exact same source check inside its own pre-lease gate.
    if spec['stage']=='stage2_full':
        from tools.reports.run_composed_stage2 import run
        return run(c,p)
    snap=spec['source_snapshot']
    verify(ROOT,manifest_sha256=snap['manifest_sha256'],io_workers=spec['io_workers'])
    if spec['stage']=='stage1':
        from tools.reports.context_stage1_start_pipeline import run
        return run(c,p)
    if spec['stage']=='stage2_window':
        from tools.reports.run_stage2_view_window import run
        return run(c,p)
    if spec['stage'] in ('stage2_prepare', 'stage2_prepare_dp6'):
        from tools.reports.prepare_stage2_view_package import run
        return run(c,p)
    from tools.reports.prepare_stage2_dataset import run
    return run(c,p,'release')


def run_owned(args,driver,key):
    # The directory was exclusively created by this attempt. Never adopt it on retry.
    try:
        spec,c,p,identity=request(args.config,args.local)
        if identity['resolved_semantic_sha256']!=key or Path(spec['driver_root'])!=driver:
            raise ValueError('detached request changed')
        execute(spec,c,p)
    except BaseException as exc:
        commit_json(driver/'terminal.json',dict(exit_code=1,request_sha256=key,error=repr(exc),restart=False))
        raise
    commit_json(driver/'terminal.json',dict(exit_code=0,request_sha256=key,stage=spec['stage'],
        evidence_root=c['output']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    parser.add_argument('--action',choices=('plan','run','detach'),default='plan')
    parser.add_argument('--request-sha256',help=argparse.SUPPRESS)
    parser.add_argument('--owned-driver',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.owned_driver:
        if not args.request_sha256 or args.action!='run':parser.error('invalid detached child')
        run_owned(args,Path(args.owned_driver),args.request_sha256)
        return
    spec,c,p,identity=request(args.config,args.local)
    key=identity['resolved_semantic_sha256']
    if args.request_sha256 and key!=args.request_sha256:raise ValueError('detached request changed')
    if args.action=='plan':
        print(json.dumps(dict(stage=spec['stage'],backend_config_sha256=p['resolved_semantic_sha256'],
            source_manifest_sha256=spec['source_snapshot']['manifest_sha256'],request_sha256=key,
            executed=False,device_admission=False)));return
    if args.action=='run':
        driver=Path(spec['driver_root']);check_paths({'driver':driver});driver.mkdir(parents=True,exist_ok=False)
        run_owned(args,driver,key)
        return
    root=Path(spec['driver_root']);check_paths({'driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.run_simple_experiment','--config',str(Path(args.config).resolve()),
          '--local',str(Path(args.local).resolve()),'--action','run','--request-sha256',key,'--owned-driver',str(root)]
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
    commit_json(root/'started.json',dict(pid=child.pid,argv=argv,request_sha256=key,
        process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19]))
    print(child.pid)


if __name__=='__main__':
    main()
