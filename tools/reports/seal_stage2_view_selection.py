"""Freeze source selection inputs and the audited Stage2 view checkpoint roster."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_decision_contract import PREFIX_LAYOUT, render_prefix_context_request
from multimodal_jev.local_checkpoint import commit_json, sync_directory
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.slurp_portable_release import load
from multimodal_jev.source_snapshot import inventory
from multimodal_jev.stage2_view_runtime import pinned

DATASET = 'SLURP_clean_SELECTION60'


def read(path):
    return json.loads(Path(path).read_bytes())


def bound(entry):
    return pinned(entry['path'], entry['sha256'])


def source_plan(c, provenance):
    window, audit, terminal = (bound(c[k]) for k in ('window', 'audit', 'terminal'))
    deployment = window['deployment']
    package = Path(window['training_package']['output'])
    manifest_sha = window['training_package']['manifest_sha256']
    m = pinned(package/'PREPARED.json', manifest_sha)
    cfg = pinned(package/'config.json', m['files']['config.json'])
    stem = Path(deployment['operations_root'])/deployment['tag']
    transport_path = Path(str(stem)+'-collected.transport.json')
    transport = pinned(transport_path, audit['transport_sha256'])
    launch, inspect, wait = [read(str(stem)+'-'+k+'.json') for k in ('launch','inspect','wait')]
    cid = audit['container_id']
    if (window['window'] != 'formal' or cfg['view_window'] != 'formal'
            or terminal != dict(exit_code=0,status=audit['status'],audit_sha256=c['audit']['sha256'])
            or audit['status'] != 'stage2_view_formal_outputs_audited'
            or audit['manifest_sha256'] != launch['manifest_sha256'] or launch['manifest_sha256'] != manifest_sha
            or audit['image'] != inspect['Image'] or audit['image'] != m['image']
            or cid != launch['container_id'] or cid != inspect['Id'] or cid != transport['container_id']
            or wait != dict(container_id=cid,exit_code=0)
            or transport['state'] != inspect['State'] or inspect['State']['Status'] != 'exited'
            or inspect['State']['OOMKilled'] or inspect['State']['ExitCode'] != 0
            or transport['status'] != 'transport_complete_independent_audit_required'
            or any(transport[k] for k in ('bad_pins','bad_package','launch_binding_mismatches'))
            or audit['dataset_sha256'] != cfg['schedule_sha256']
            or cfg['prompt_layout'] != PREFIX_LAYOUT
            or cfg['release'] != c['release']['root'] or cfg['release_sha256'] != c['release']['sha256']):
        raise ValueError('independently audited formal Stage2 view run required')
    contract = audit['formal_contract']; steps = contract['evaluation_steps']
    if (not contract['source_selection_only'] or steps != cfg['evaluation_steps']
            or sorted(set(steps)) != steps or steps[0] != 0 or steps[-1] != cfg['stop_update']
            or steps[1:] != contract['recovery_checkpoint_steps']
            or len(audit['updates']) != steps[-1]):
        raise ValueError('complete registered source selection curve required')
    collection = Path(str(stem)+'-collected')

    def files(root, rel, pinned_files):
        folder = root/rel
        paths = sorted(str(p.relative_to(root)) for p in folder.rglob('*') if p.is_file())
        expected = {k:v for k,v in pinned_files.items() if k.startswith(rel+'/')}
        if not expected or set(paths) != expected.keys() or inventory(root, paths, io_workers=c['io_workers']) != expected:
            raise ValueError('audited policy payload changed: '+rel)
        return folder

    points = []
    for step in steps:
        if step:
            rel = f'run/checkpoint-{step:04d}-committed.json'
            commit = pinned(collection/rel, transport['files'][rel])
            name = f'policy-{step:04d}'
            if (commit['updates'] != step or len(commit['rank_shards']) != audit['world_size']
                    or commit['joint_policy_directory'] != name
                    or commit['joint_policy_manifest_sha256'] != transport['files'][f'run/{name}/MANIFEST.json']):
                raise ValueError('registered checkpoint does not match its audited publication')
            policy = files(collection, 'run/'+name, transport['files'])
            adapter = files(collection, f'run/adapter-{step:04d}', transport['files'])
            policy_id = commit['joint_policy']['policy_id']
        else:
            policy = files(package, 'initial/policy', m['files'])
            adapter = files(package, 'initial/adapter', m['files'])
            policy_id = cfg['source_policy_id']
        joint = read(policy/'MANIFEST.json')
        native = read(adapter/'CONTEXT_POLICY.json')
        if (joint['policy_id'] != policy_id or joint['contract'] != cfg['source_contract']
                or native['policy_id'] != policy_id):
            raise ValueError('audited checkpoint, contract and native adapter must agree')
        points.append(dict(step=step, policy_id=policy_id, snapshot=str(policy), adapter=str(adapter),
                           manifest_sha256=file_hash(policy/'MANIFEST.json')))
    return dict(schema='jev-stage2-view-selection-source-v1', evaluation_dataset=DATASET,
                base_id=cfg['base_id'], source_contract=cfg['source_contract'], checkpoints=points,
                target_tuning_allowed=False, model_selection_performed=False,
                release=c['release'], training_audit=c['audit'], provenance=provenance)


def seal(c, provenance):
    out = Path(c['output'])
    check_paths({'selection seal':out})
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    source = source_plan(c, provenance)
    release = c['release']
    rows, task, manifest = load(release['root'], release['sha256'], 'selection', purpose='evaluate')
    candidates = [dict(id=x['id'], definition=x['description']) for x in task['schema']['candidates']]
    ids = [x['id'] for x in candidates]
    if len(rows) != manifest['counts']['selection']['recordings'] or len(ids) != 60 or len(set(ids)) != 60:
        raise ValueError('full native60 source selection required')
    req = compile_candidates(rows[0]['audio'], candidates, task['thinking']['requirements'], semantics='exclusive_event')
    system, history, readouts = render_prefix_context_request(req)
    if history:
        raise ValueError('single audio evaluation must have empty history')
    inputs, targets = [], []
    shared = Path(c['shared_root']).resolve()
    for r in rows:
        if (r['role'] != 'selection' or r['official_split'] != 'train'
                or r['intent'] not in ids or not Path(r['audio']).resolve().is_relative_to(shared)):
            raise ValueError('source selection label or shared audio identity differs')
        keys = {k:r[k] for k in ('id','audio_sha256')}
        inputs.append(dict(keys,audio=r['audio']))
        targets.append(dict(keys,gold=r['intent'],group=r['source_group']))
    with TemporaryDirectory(dir=out.parent) as work:
        staging = Path(work)/'selection'
        payload = {
            f'inference/{DATASET}/panel.json': dict(system=system, readouts=readouts,
                questions=[asdict(q) for q in req.questions], requirements=list(req.requirements),
                version=req.version, candidate_ids=ids, prompt_layout=PREFIX_LAYOUT, history=[]),
            f'inference/{DATASET}/rows.json': inputs,
            f'scoring/{DATASET}/targets.json': targets,
            'source-plan.json': source,
        }
        for rel, data in payload.items():
            (staging/rel).parent.mkdir(parents=True, exist_ok=True)
            commit_json(staging/rel, data)
        commit_json(staging/'MANIFEST.json', dict(schema='jev-native-evaluation-seal-v1',
            prompt_layout=PREFIX_LAYOUT, threshold=c['diagnostic_threshold'],
            datasets={DATASET: dict(recordings=len(inputs), candidates=len(ids), split='source_selection')},
            target_tuning_allowed=False, files=inventory(staging, payload, io_workers=c['io_workers']),
            source_release_sha256=release['sha256']))
        staging.rename(out)
        sync_directory(out.parent)
    return dict(output=str(out), manifest_sha256=file_hash(out/'MANIFEST.json'),
                recordings=len(rows), steps=[p['step'] for p in source['checkpoints']], npu_executed=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    a = p.parse_args()
    spec, meta = load_config(a.config, a.local)
    print(json.dumps(seal(spec, meta)))
