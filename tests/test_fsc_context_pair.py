import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'tools/jev'))
from tools.jev import run_fsc_context_pair as module
from multimodal_jev.slurp_training import content_hash


@pytest.fixture
def inputs():
    rows = [dict(id=str(i), speaker_id=str(i % 10)) for i in range(3793)]
    receipts = [dict(tokens=2400+i % 70, positions=[list(range(31))],
                    tensors=dict(input_ids={'sha256':str(i)}, positions={'sha256':'positions'}))
                for i in range(3793)]
    return rows, receipts


def config(role):
    return dict(policy_id=role, source_role=role,
                input_manifest=dict(sha256='inputs'))


def prediction(row, receipt, c):
    identity = dict(policy_id=c['policy_id'], positions=receipt['positions'][0],
                    inputs={k:v for k,v in receipt['tensors'].items() if k != 'positions'})
    return dict(id=row['id'], request_identity=identity, prob=[.125]*31,
                policy_id=c['policy_id'], request_id=content_hash(identity))


def panel(root, c, rows, receipts):
    root.mkdir(exist_ok=True)
    indices = module.panel_indices(rows, receipts)
    value = dict(policy_id=c['policy_id'], indices=indices, input_manifest_sha256='inputs',
                 records=[prediction(rows[i], receipts[i], c) for i in indices])
    for phase in ('hf', 'vllm'):
        module.write(root/(phase+'.json'), value)
    module.write(root/'comparison.json', module.compare(value, value))


def full(root, c, rows, receipts):
    root.mkdir()
    module.write(root/'predictions.json', dict(records=[prediction(row, receipt, c)
                 for row, receipt in zip(rows, receipts, strict=True)]))
    module.write(root/'result.json', dict(recordings=3793, decisions=31, policy_id=c['policy_id'],
                 source_role=c['source_role'], predictions_sha256=module.sha(root/'predictions.json')))


@pytest.mark.parametrize('fault', ['policy', 'panel', 'input', 'both_inputs', 'probability', 'summary'])
def test_panel_recomputed_against_frozen_inputs(tmp_path, inputs, fault):
    rows, receipts = inputs;c = config('initial');panel(tmp_path, c, rows, receipts)
    assert module.check_panel(tmp_path, c, rows, receipts)['exact']
    hf = module.read(tmp_path/'hf.json');vl = module.read(tmp_path/'vllm.json')
    if fault == 'policy':hf['policy_id'] = vl['policy_id'] = 'other'
    elif fault == 'panel':hf['indices'][0] = vl['indices'][0] = 99999
    elif fault == 'input':vl['records'][0]['id'] = 'other'
    elif fault == 'both_inputs':
        hf['records'][0]['request_identity']['inputs'] = {}
        vl['records'][0]['request_identity']['inputs'] = {}
    elif fault == 'probability':vl['records'][0]['prob'][0] += .001
    else:(tmp_path/'comparison.json').write_text('{}')
    (tmp_path/'hf.json').write_text(json.dumps(hf));(tmp_path/'vllm.json').write_text(json.dumps(vl))
    with pytest.raises(ValueError):module.check_panel(tmp_path, c, rows, receipts)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'stale', 'nan', 'digest'])
def test_full_predictions_rechecked_not_only_summary(tmp_path, inputs, fault):
    rows, receipts = inputs;c = config('selected');root = tmp_path/'inference'
    full(root, c, rows, receipts)
    assert module.check_full(root, c, rows, receipts)['recordings'] == 3793
    p = module.read(root/'predictions.json')
    if fault == 'missing':p['records'].pop()
    elif fault == 'duplicate':p['records'][-1] = p['records'][0]
    elif fault == 'stale':p['records'][1]['policy_id'] = 'initial'
    elif fault == 'nan':p['records'][1]['prob'][0] = float('nan')
    else:p['records'][1]['request_id'] = 'bad'
    (root/'predictions.json').write_text(json.dumps(p))
    result = module.read(root/'result.json');result['predictions_sha256'] = module.sha(root/'predictions.json')
    (root/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError):module.check_full(root, c, rows, receipts)


@pytest.fixture
def flow(tmp_path, monkeypatch, inputs):
    import multimodal_jev.local_runtime_storage as storage
    rows, receipts = inputs
    monkeypatch.setattr(storage, 'require_training_storage', lambda p: [])
    pair = {role:config(role) for role in ('initial', 'selected')}
    monkeypatch.setattr(module, 'configs', lambda spec: (pair, rows, receipts))
    spec = dict(source_package=str(tmp_path/'source'), model=str(tmp_path/'model'),
                source_manifest_sha256='source', inputs=dict(cache_root=str(tmp_path/'cache'),
                requests=str(tmp_path/'inputs/requests.jsonl')))
    calls = []
    def invoke(script, args, log):
        c = module.read(args[args.index('--config')+1]);role = c['source_role']
        phase = args[args.index('--phase')+1] if '--phase' in args else 'full'
        calls.append((role, phase));log.write_text('mocked model subprocess')
        root = Path(args[args.index('--output')+1])
        if phase == 'vllm':panel(root, c, rows, receipts)
        elif phase == 'full':full(root, c, rows, receipts)
    monkeypatch.setattr(module, 'invoke', invoke)
    return spec, tmp_path/'output', calls, invoke


def test_both_parities_before_full_inference_no_scoring(flow):
    spec, out, calls, _ = flow
    result = module.run(spec, out)
    assert calls == [('initial', 'hf'), ('initial', 'vllm'), ('selected', 'hf'),
                     ('selected', 'vllm'), ('initial', 'full'), ('selected', 'full')]
    assert result['target_tuning'] is False and len(result['policies']) == 2
    assert module.read(out/'terminal.json')['exit_code'] == 0
    with pytest.raises(FileExistsError):module.run(spec, out)
    assert len(calls) == 6


def test_failure_preserves_files_no_full_inference_or_retry(flow, monkeypatch):
    spec, out, calls, original = flow
    def invoke(script, args, log):
        original(script, args, log)
        if calls[-1] == ('selected', 'vllm'):
            root = Path(args[args.index('--output')+1])
            v = module.read(root/'vllm.json');v['records'][0]['prob'][0] = .25
            (root/'vllm.json').write_text(json.dumps(v))
    monkeypatch.setattr(module, 'invoke', invoke)
    with pytest.raises(ValueError):module.run(spec, out)
    assert len(calls) == 4 and (out/'initial/parity/hf.json').exists()
    terminal = module.read(out/'terminal.json')
    assert terminal['exit_code'] == 1 and terminal['retry'] is False
    assert terminal['failed_phase'] == 'selected-parity-vllm'
    assert not (out/'initial/inference').exists()


def test_output_cannot_overwrite_sources(flow):
    spec, _, calls, _ = flow
    with pytest.raises(ValueError):module.run(spec, Path(spec['source_package'])/'output')
    assert not calls


def test_source_pair_inputs_and_runtime_bound_before_dispatch(tmp_path, monkeypatch, inputs):
    rows, receipts = inputs
    script = tmp_path/'repo/tools/jev/run_fsc_context_pair.py';script.parent.mkdir(parents=True)
    script.write_text('fixture');(tmp_path/'repo/src').mkdir()
    (tmp_path/'repo/src/model.py').write_text('fixture')
    monkeypatch.setattr(module, '__file__', str(script))
    package = tmp_path/'package';package.mkdir();(package/'source-selection.json').write_text('{}')
    manifest = dict(input_audit_sha256='audit', processor_sha256='processor', policies={
        role:dict(policy_id=role, contract={'base_id':'base'}, policy=role+'/policy', adapter=role+'/adapter')
        for role in ('initial', 'selected')})
    monkeypatch.setattr(module, 'verify', lambda *a:manifest)
    monkeypatch.setattr(module, 'load_inputs', lambda c:(rows, receipts, dict(manifest={'processor_sha256':'processor'})))
    selected = []
    monkeypatch.setattr(module, 'validate_source', lambda c:selected.append(c['source_role']))
    spec = dict(source_package=str(package), source_manifest_sha256='source', physical_devices=[7],
                model='model', pins={str(p.resolve()):module.sha(p) for p in (tmp_path/'repo').rglob('*.py')},
                inputs=dict(input_manifest={}, input_audit={'sha256':'audit'}, requests='requests',
                            input_receipts='receipts', cache_root='cache'))
    pair, _, _ = module.configs(spec)
    assert selected == ['initial', 'selected'] and pair['selected']['policy_id'] == 'selected'
    for fault in ('target_labels', 'audit', 'source', 'device'):
        bad = copy.deepcopy(spec)
        if fault == 'target_labels':bad['inputs']['target_csv'] = 'test.csv'
        elif fault == 'audit':bad['inputs']['input_audit']['sha256'] = 'wrong'
        elif fault == 'source':bad['pins'].pop(str(script.resolve()))
        else:bad['physical_devices'] = [4, 7]
        with pytest.raises(ValueError):module.configs(bad)
