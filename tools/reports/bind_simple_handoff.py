"""Render the existing endpoint successor from a recipe and this attempt's local evidence."""
import argparse
import json
import os
from pathlib import Path
import re
import tempfile

from multimodal_jev.local_checkpoint import sync_directory
from multimodal_jev.local_config import load_config, resolve
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.source_snapshot import regular
from multimodal_jev.stage_experiment_contract import reference_batch
from tools.reports.compile_simple_experiment import compile_recipe, encoded, pin, ROOT


def read(path):
    return json.loads(Path(path).read_bytes())


def binding(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'ordinary artifact required: {path}')
    return pin(path, path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def device_plan(cards, batch, services):
    n = batch['training_devices']
    require(len(cards) == n + batch['inference_instances'] and
            len(services) == batch['inference_instances'] and len(set(cards)) == len(cards)
            and all(type(d) is int and d >= 0 for d in cards), 'device inventory and design differ')
    return dict(training_devices=cards[:n], inference_services=[
        dict(name=name, device=d) for name, d in zip(sorted(services), cards[n:], strict=True)])


def check_recipe(pkg, pipeline, recipe, compiled, final):
    manifest_pin = pkg['stage1_layout']['design']
    require(manifest_pin == binding(manifest_pin['path']), 'Stage1 layout manifest changed')
    manifest = read(manifest_pin['path'])
    actual = read(pkg['design'])
    require(binding(pkg['design'])['sha256'] == pipeline['design_sha256'] == manifest['stage1']['sha256']
            and binding(final['operations_design'])['sha256'] == pipeline['design_sha256'],
            'Stage1 package, formal run and layout must share the design')
    new = json.loads(compiled['two-stage.json'])
    # The first production run used the pinned base before the recipe compiler existed.
    # Permit only its descriptive text to differ; never drop a loss, schedule or LoRA field.
    if actual != new:
        require(pipeline['design_sha256'] == recipe['base']['sha256'],
                'Stage1 did not consume the compiled design or the pinned base')
        for key in ('status', 'model', 'batch_rule', 'implementation_defaults'):
            new[key] = actual[key]
        for stage, key in [('stage1', 'initialization'), ('stage2', 'population_plan')]:
            new[stage][key] = actual[stage][key]
        require(actual == new, 'Stage1 execution differs from compiled recipe')
    expected = json.loads(compiled['experiment.json'])
    require(manifest['stage2'] == expected['stage2'] == actual['stage2']['objective_config']
            and manifest['dataset']['sha256'] == expected['dataset']['sha256']
            and manifest['training'] == expected['training'], 'training layout, dataset or objective differs')
    layout = dict(design=manifest_pin['sha256'], plan=pkg['stage1_layout']['plan']['sha256'],
                  cache=pkg['layout_cache']['sha256'])
    require(layout == final['pipeline']['expected_stage1_layout']
            and final['pipeline']['root'] == pipeline['output'], 'wrong predecessor layout or pipeline')
    if 'runtime' in manifest:
        require(manifest['runtime']['sha256'] == expected['runtime']['sha256'],
                'compiled serving policy differs from the Stage1 manifest')
    return actual, layout


def build(template, local):
    spec = read(template)
    require(spec['schema'] == 'jev-simple-stage2-handoff-v1', 'unknown handoff schema')
    local = read(local)
    require(set(local) == {'schema','macros'} and local['schema'] == 'jev-local-macros-v1',
            'local macro file required')
    i, _ = resolve(spec['inputs'], local['macros'])
    paths = {k:regular(ROOT, v) for k,v in spec['templates'].items()}
    require(Path(i['run_root']).is_absolute() and re.fullmatch(r'[a-zA-Z0-9_-]+', i['attempt']),
            'absolute output root and safe attempt name required')
    recipe = read(i['recipe'])
    pipe, _ = load_config(paths['pipeline'], i['pipeline_local'])
    pkg = pipe['package']
    final = read(i['finalization_config'])
    ops, _ = load_config(final['operations_config'], final['operations_local'])
    # No publication: destination affects only paths in the synthesized manifest.
    compiled = compile_recipe(recipe, Path(pkg['design']).parent)
    actual, layout = check_recipe(pkg, pipe, recipe, compiled, dict(final,operations_design=ops['design']))
    d = reference_batch(actual)
    limits = json.loads(compiled['stage2-runtime.json'])
    prepared = Path(i['runtime_package'])/'PREPARED.json'
    rt = read(prepared)
    topo = device_plan(pkg['physical_devices'], d, rt['layout']['services'])
    inf = [v['device'] for v in topo['inference_services']]
    scoring = i['scoring_devices']
    require(isinstance(scoring, list) and scoring
            and all(type(device) is int and device in pkg['physical_devices'] for device in scoring)
            and len(set(scoring)) == len(scoring), 'offline scoring devices must be distinct cards in this run')
    scoring_macro = len(scoring) * d['inference_batch_queries']
    schedule = Path(i['reference_schedule'])/'MANIFEST.json'
    sched = read(schedule)
    require(sched['batch_plan']['configuration'] == d and sched['release_sha256'] == pkg['release_sha256']
            and sched['parent_manifest_sha256'] == pkg['parent_schedule_sha256']
            and sched['parent_release_sha256'] == pkg['parent_release_sha256'], 'wrong reference schedule')

    panel, _ = load_config(paths['panel'], i['parity_local'])
    audit = read(panel['audit_output'])
    require(panel['panel_config'] == limits['panel_config'] and panel['release_sha256'] == pkg['release_sha256']
            and panel['cache_root'] == pkg['encoded_audio_cache'] and panel['prompt_layout'] == pkg['prompt_layout']
            and audit['status'] == 'train_parity_inputs_independently_audited'
            and audit['manifest_sha256'] == panel['manifest_sha256'] and audit['labels_used'] is False
            and audit['population'] == d['source_queries']
            and max(limits['panel_config']['minimum_queries'], scoring_macro) <= audit['recordings'] <= audit['population'],
            'TRAIN parity panel must cover a complete offline scoring macro')
    suite, _ = load_config(paths['suite'], i['image_local'])
    report = read(suite['report'])
    src = Path(suite['output'])/(suite['tag']+'-package')
    require(rt['image'] == suite['image'] == report['image']
            and report['counts'] == dict(tests=suite['expected_tests'], errors=0, failures=0, skipped=0)
            and report['state']['ExitCode'] == 0 and report['state']['Status'] == 'exited'
            and report['terminal']['exit_code'] == 0 and read(src/'TESTS.json') == suite['tests']
            and binding(src/'FILES.json')['sha256'] == report['source_manifest_sha256'],
            'CPU suite, image or source evidence differs')
    shared = {k:pkg[k] for k in ('encoded_audio_cache','audio_mirror_root','prompt_layout')}
    shared['manifests'] = dict(encoded_audio_cache=pipe['shared_cache_manifest_sha256'],
        audio_mirror_root=pkg['external_pins'][str(Path(pkg['audio_mirror_root'])/'MANIFEST.json')])
    deployment = {k:ops[k] for k in ('host','lease','driver_root','shared_root','release_evidence')}
    dm, _ = resolve(spec['dataset_macros'], dict(RUN_ROOT=i['run_root']))
    ds, _ = resolve(json.loads(compiled['dataset.json']), dm)
    require(ds['batch_config'] == {k:v for k,v in d.items() if k!='source_queries'}, 'view geometry differs')
    variables = dict(RUN_ROOT=i['run_root'], REMOTE_PARENT=str(Path(pkg['remote_root']).parent),
        ATTEMPT=i['attempt'], PREDECESSOR=dict(config=binding(i['finalization_config']),driver=binding(i['finalizer_driver'])),
        DESIGN_PATH=pkg['design'], DESIGN_SHA=pipe['design_sha256'], LAYOUT=layout, DATASET=ds,
        RUNTIME=dict(package=i['runtime_package'],manifest_sha256=binding(prepared)['sha256']),
        TRAINING_DEVICES=topo['training_devices'], INFERENCE_DEVICES=inf, TOPOLOGY=topo,
        SCORING_DEVICES=scoring, SCORING_PREFETCH_AHEAD=scoring_macro,
        SCHEDULE=i['reference_schedule'], SCHEDULE_SHA256=binding(schedule)['sha256'],
        SHARED_INPUTS=shared, SERVING=limits['serving'], PANEL=panel, INPUT_EVIDENCE={
            k:panel[k] for k in ('input_proof','input_proof_sha256','input_receipts','input_receipts_sha256')},
        DEPLOYMENT=deployment, EXTENSION_SOURCE=str(src), EXTENSION_CPU_REPORT=binding(suite['report']),
        EXTENSION_EXPECTED_TESTS=suite['expected_tests'], BOOTSTRAP=binding(paths['bootstrap']),
        OBJECTIVE_DESIGN=binding(regular(ROOT,recipe['objective']['path'])), MONITOR_SECONDS=actual['monitor_seconds'],
        HOST=ops['host'], LEASE=ops['lease'], DRIVER_ROOT=ops['driver_root'], REPLAY_DEVICE=inf[-1],
        IMAGE=rt['image'], DEADLINE=ops.get('lease_deadline_unix'),
        **{k.upper():pkg[k] for k in ('parent_schedule','parent_schedule_sha256','parent_release_sha256')})
    request, _ = resolve(spec['request'], variables)
    from tools.reports.context_stage2_endpoint_successor import validate
    validate(request)
    return dict(schema='jev-local-macros-v1',macros=dict(STAGE2_ENDPOINT_SUCCESSOR=request))


def publish(path, value):
    # Publish a complete local macro file without an overwrite race (also on concurrent calls).
    path = Path(path)
    check_paths({'handoff':path})
    with tempfile.NamedTemporaryFile(dir=path.parent) as tmp:
        tmp.write(encoded(value))
        tmp.flush()
        os.fsync(tmp.fileno())
        try:
            os.link(tmp.name, path)
            sync_directory(path.parent)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded(value):
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','local','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    value = build(args.config,args.local)
    publish(args.output,value)
    print(json.dumps(dict(output=str(args.output), file_sha256=binding(args.output)['sha256'], dispatched=False)))


if __name__ == '__main__':
    main()
