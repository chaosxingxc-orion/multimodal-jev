"""Read independently verified native audio tensors and run endpoint inference."""
import argparse
import json
import math
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_preparation import processor_cache_identity
from multimodal_jev.context_decision_contract import PREFIX_LAYOUT
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.context_macro_rollout import digest
from tools.reports.prepare_native_seal_cache import plan
from tools.reports.native_evaluation_cache import read_verified
from tools.reports.context_selection_audited_cache import encoder


def read(p):
    return json.loads(Path(p).read_text())


def bound(entry):
    if file_hash(entry['path']) != entry['sha256']:
        raise ValueError('pinned endpoint evidence changed')
    return read(entry['path'])


def selected_panel(c, assets, candidates, encode):
    return dict(dataset=c['cache']['dataset'], assets=assets, candidate_ids=candidates,
                input_receipts=[encode(asset)[1] for asset in assets], labels_used=False)


def load(c):
    from multimodal_jev.model_provider import cache_namespace, require_same_provider
    spec = c['cache']
    require_same_provider(c,spec)
    namespace = cache_namespace(c)
    root = Path(spec['output'])
    receipts = root/'inputs.jsonl'
    m, verification = read_verified(spec, c['cache_verification'])
    if m['processor_sha256'] != processor_cache_identity(c['snapshot']):
        raise ValueError('frozen endpoint processor differs from cache')
    jobs = plan(spec)
    candidates = [q.id for q in jobs[0][2].questions]
    requests, assets, priors = {}, [], []
    with receipts.open() as stream:
        for (name, row, req), line in zip(jobs, stream, strict=True):
            prior = json.loads(line)
            asset = dict(id=row['id'], audio_sha256=row['audio_sha256'],
                         request_sha256=content_hash(req.model_input()))
            if any(prior[k] != value for k, value in
                   dict(dataset=name, id=row['id'], audio_sha256=row['audio_sha256']).items()):
                raise ValueError('cache roster or audio differs')
            requests[row['id']] = req
            assets.append(asset)
            priors.append(dict(prior, request_sha256=asset['request_sha256']))
    if len(assets) != m['recordings'] or len(assets) != verification['recordings']:
        raise ValueError('complete frozen population required')
    enc = spec['encoding']
    if enc['decision_token_id'] != c['contract']['token']['id'] or enc['audio_token_id'] != c['audio_token_id']:
        raise ValueError('processor and endpoint markers differ')
    encode = encoder(root/'cache', requests, candidates, priors,
                     processor_sha256=m['processor_sha256'], decision_token_id=enc['decision_token_id'],
                     audio_token_id=enc['audio_token_id'], prompt_layout=PREFIX_LAYOUT,
                     provider_namespace=namespace,max_tokens=enc['max_tokens'])

    def prepare(asset):
        row, receipt = encode(asset)
        return row, {k: v for k, v in receipt.items() if k != 'audio_receipt'}

    return assets, candidates, prepare


def hf_partition(assets, index, width):
    if type(width) is not int or type(index) is not int or not 0 <= index < width <= len(assets):
        raise ValueError('explicit nonempty HF shard required')
    return assets[index::width]


def validate_hf_records(result, selected, policy_id):
    import torch
    if (result.get('schema') != 'jev-target-width-hf-reference-v1'
            or result['panel_sha256'] != digest(selected) or result['policy_id'] != policy_id):
        raise ValueError('HF input roster or policy differs')
    width = len(selected['candidate_ids'])
    for asset, receipt, row in zip(selected['assets'], selected['input_receipts'], result['records'], strict=True):
        if row['id'] != asset['id'] or row['request_identity'] != dict(
                policy_id=policy_id, positions=receipt['positions'], inputs=receipt['inputs']):
            raise ValueError('HF request evidence differs')
        logits = torch.tensor(row['logits'], dtype=torch.float32)
        probs = torch.tensor(row['prob'], dtype=torch.float32)
        if (logits.shape != (width,) or probs.shape != (width,) or not logits.isfinite().all()
                or not probs.isfinite().all() or ((probs <= 0) | (probs >= 1)).any()
                or (logits.sigmoid()-probs).abs().max().item() > 2e-7):
            raise ValueError('invalid HF logit or probability evidence')


def hf_check(result, selected, reference, reference_panel, policy_id):
    validate_hf_records(result, selected, policy_id)
    validate_hf_records(reference, reference_panel, policy_id)
    expected = {r['id']: r for r in reference['records']}
    if any(expected.get(row['id']) != row for row in result['records']):
        raise ValueError('HF shard differs from the original HF reference')


def validate_hf_endpoint(source, point, cache):
    if cache['dataset'] == 'SLURP_clean_SELECTION60':
        raise ValueError('direct HF successor requires a held-out dataset')
    if source['schema'] == 'jev-stage2-view-heldout-endpoint-v1':
        if (source['checkpoints'] != [point] or source['fixed_terminal_step'] != point['step']
                or source['evaluation_dataset'] != cache['dataset']
                or source['seal_sha256'] != cache['seal']['sha256']):
            raise ValueError('direct HF successor requires the frozen Stage2 terminal checkpoint and held-out seal')
        return
    if source['schema'] != 'jev-infonce-source-evaluation-plan-v1':
        raise ValueError('direct HF successor requires a frozen held-out endpoint')
    points = source['checkpoints']
    audit = source['source_audit']
    step = point['step']
    if (source['status'] != 'audited_source_policies_bound_evaluation_pending'
            or source['algorithm'] != 'candidate_infonce'
            or type(step) is not int or step <= 0
            or point != points[-1]
            or source['reference_endpoint'] != {'step': step, 'policy_id': point['policy_id']}
            or step != source['experiment']['task_stage']['updates']
            or audit['status'] != 'formal_infonce_artifacts_audited'
            or audit['updates'][-1]['step'] != step or audit['updates'][-1]['checkpoint'] is not True):
        raise ValueError('direct HF successor requires the independently audited Stage1 terminal checkpoint')


def hf_sharded(c, source, phase, out, bundle, assets, candidates, encode):
    from tools.reports.context_transfer_macro_parity import hf_reference

    validate_hf_endpoint(source, c['point'], c['cache'])
    spec = c['hf_shards']
    devices = spec['devices']
    width = len(devices)
    if not devices or devices != sorted(set(devices)):
        raise ValueError('distinct ordered devices required')
    panel = bound(spec['reference_panel'])
    prior = bound(spec['reference'])
    policy_id = c['point']['policy_id']
    if panel['dataset'] != c['cache']['dataset'] or panel['candidate_ids'] != candidates:
        raise ValueError('HF reference dataset or decision roster differs')

    def binding(index):
        return dict(dataset=panel['dataset'], policy_id=policy_id, device=devices[index],
                    index=index, width=width, reference_sha256=spec['reference']['sha256'],
                    reference_panel_sha256=spec['reference_panel']['sha256'])

    def prior_shard(index):
        # The panel is independently checked against the original frozen input receipts.
        by_id = {r['id']: r for r in panel['input_receipts']}
        subset = hf_partition(panel['assets'], index, width)
        return dict(panel, assets=subset, input_receipts=[by_id[a['id']] for a in subset])

    def read_shard(phase, index):
        root = out.parent/f'{phase}-{index}'
        if read(root/'terminal.json') != {'exit_code': 0}:
            raise ValueError('HF shard did not finish')
        receipt = read(root/'receipt.json')
        path = root/'reference.json'
        if receipt != dict(binding(index), sha256=file_hash(path)):
            raise ValueError('HF shard output or device binding differs')
        return read(path)

    if phase != 'hf-shard-check':
        for index in range(width):
            hf_check(read_shard('hf-shard-check', index), prior_shard(index), prior, panel, policy_id)
    if phase == 'audit-hf-shards':
        if c['physical_devices'] or c['cache'].get('panel_size') is not None:
            raise ValueError('full HF audit must be CPU only')
        records = {}
        for index in range(width):
            selected = selected_panel(c, hf_partition(assets, index, width), candidates, encode)
            result = read_shard('hf-shard-full', index)
            validate_hf_records(result, selected, policy_id)
            for row in result['records']:
                if row['id'] in records:
                    raise ValueError('duplicate HF shard output')
                records[row['id']] = row
        ordered = [records[a['id']] for a in assets]
        predictions = dict(policy_id=policy_id, records=ordered)
        commit_json(out/'predictions.json', predictions)
        result = dict(score(c, assets, candidates, ordered), runtime='HF_standalone',
                      devices=devices, panel_recordings=len(panel['assets']),
                      predictions_sha256=file_hash(out/'predictions.json'),
                      status='complete_hf_shards_audited')
        commit_json(out/'metrics.json', result)
        return result

    index = spec['index']
    subset = hf_partition(assets, index, width)
    is_check = phase == 'hf-shard-check'
    if (c['physical_devices'] != [devices[index]] or is_check != (c['cache'].get('panel_size') is not None)):
        raise ValueError('HF shard device or input scope differs')
    selected = selected_panel(c, subset, candidates, encode)
    if is_check and selected != prior_shard(index):
        raise ValueError('HF shard panel inputs differ from frozen reference')
    result = hf_reference(dict(c, policy_id=policy_id), {}, selected, bundle, encode, out)
    validate_hf_records(result, selected, policy_id)
    if is_check:
        hf_check(result, selected, prior, panel, policy_id)
    commit_json(out/'reference.json', result)
    receipt = dict(binding(index), sha256=file_hash(out/'reference.json'))
    commit_json(out/'receipt.json', receipt)
    return receipt


def run(c, provenance, phase):
    import torch
    torch.set_num_threads(c['torch_threads'])
    snapshot = Path(c['snapshot'])
    point = c['point']
    source = bound(c['source_plan'])
    points = source['checkpoints']
    if source.get('schema') == 'jev-stage2-view-selection-source-v1':
        if (source['evaluation_dataset'] != 'SLURP_clean_SELECTION60'
                or c['cache']['dataset'] != source['evaluation_dataset']):
            raise ValueError('registered Stage2 points may only use source selection')
    elif source.get('schema') == 'jev-stage2-view-heldout-endpoint-v1':
        if (source['evaluation_dataset'] == 'SLURP_clean_SELECTION60'
                or c['cache']['dataset'] != source['evaluation_dataset']
                or c['cache']['seal']['sha256'] != source['seal_sha256']
                or len(points) != 1 or type(source['fixed_terminal_step']) is not int
                or source['fixed_terminal_step'] <= 0 or points[0]['step'] != source['fixed_terminal_step']):
            raise ValueError('only the frozen Stage2 terminal policy may use this held-out dataset')
    else:
        points = (points[0], points[-1])
    if (source['target_tuning_allowed'] is not False or source['model_selection_performed'] is not False
            or point not in points
            or source['base_id'] != c['base_id'] or source['source_contract'] != c['contract']):
        raise ValueError('checkpoint role must be frozen in the audited source plan')
    if (file_hash(snapshot/'MANIFEST.json') != point['manifest_sha256']
            or checkpoint_identity(c['model'])[0] != c['base_id']):
        raise ValueError('actual source checkpoint or base differs')
    bundle = read_execution_bundle(snapshot, expected_policy_id=point['policy_id'],
                                   expected_contract=c['contract'], expected_base_id=c['base_id'])
    assets, candidates, encode = load(c)
    out = Path(c['output'])
    check_paths({'native endpoint evaluation': out})
    out.mkdir(parents=True, exist_ok=False)
    try:
        commit_json(out/'provenance.json', provenance)
        if phase in ('hf-shard-check', 'hf-shard-full', 'audit-hf-shards'):
            result = hf_sharded(c, source, phase, out, bundle, assets, candidates, encode)
        elif phase == 'hf':
            from tools.reports.context_transfer_macro_parity import hf_reference
            if c['cache'].get('panel_size') is None:
                raise ValueError('HF is bound to the finite admission panel')
            selected = selected_panel(c, assets, candidates, encode)
            commit_json(out/'panel.json', selected)
            result = hf_reference(dict(c, policy_id=point['policy_id']), {}, selected, bundle, encode, out)
            commit_json(out/'reference.json', result)
        elif phase in ('parity', 'full', 'audit-full', 'audit-parity'):
            from tools.reports.context_replica_macro_client import connect_pool
            from tools.reports.context_macro_evaluation import evaluate
            from tools.reports.audit_context_macro_evaluation import audit
            from tools.reports.context_transfer_macro_parity import compare
            from tools.reports.context_dual_runtime_candidate import verify_runtime
            client = c['macro_client']
            identity = verify_runtime(c)
            if identity != client['runtime_identity']:
                raise ValueError('actual service runtime differs')
            parity = phase.endswith('parity')
            full = not parity
            if full != (c['cache'].get('panel_size') is None):
                raise ValueError('finite and full input populations cannot substitute for each other')
            if parity:
                selected = selected_panel(c, assets, candidates, encode)
                reference = bound(c['hf_reference'])
                if reference['policy_id'] != point['policy_id'] or reference['panel_sha256'] != digest(selected):
                    raise ValueError('HF policy or panel differs')
            else:
                admission = bound(c['parity_receipt'])
                if (admission.get('exact') is not True or admission['policy_id'] != point['policy_id']
                        or admission['dataset'] != c['cache']['dataset'] or admission['runtime_identity'] != identity
                        or admission['devices'] != c['physical_devices']
                        or admission['decision_width'] != len(candidates)
                        or admission['seal_sha256'] != c['cache']['seal']['sha256']):
                    raise ValueError('same dataset width, endpoint, runtime and topology must pass parity')
            if phase.startswith('audit-'):
                infer = Path(c['inference'])
                expected = selected['input_receipts'] if parity else [encode(a)[1] for a in assets]
                result = audit(infer, Path(client['journal_root'])/client['attempt_id'],
                               population=assets, candidate_ids=candidates, input_receipts=expected,
                               policy_id=point['policy_id'], runtime_identity=identity,
                               macro_queries=c['macro_queries'], replicas=len(client['replicas']))
                records = read(infer/'predictions.json')['records']
                if parity:
                    result.update(compare(reference, records, selected, policy_id=point['policy_id']),
                                  runtime_identity=identity, devices=c['physical_devices'],
                                  decision_width=len(candidates), seal_sha256=c['cache']['seal']['sha256'])
                    commit_json(out/'comparison.json', result)
                    if not result['exact']:
                        raise ValueError('target-width HF/replica parity failed')
                else:
                    result.update(score(c, assets, candidates, records))
                    commit_json(out/'metrics.json', result)
            else:
                pool = connect_pool(client, bundle=bundle)
                try:
                    result = evaluate(pool, assets, encode, candidate_ids=candidates, policy_id=point['policy_id'],
                                      runtime_identity=identity, audio_token_id=c['audio_token_id'],
                                      macro_queries=c['macro_queries'], output=out/'inference')
                finally:
                    pool.close()
        else:
            raise ValueError('unsupported phase')
        commit_json(out/'terminal.json', dict(exit_code=0))
        return result
    except BaseException as exc:
        commit_json(out/'terminal.json', dict(exit_code=1, error=repr(exc)))
        raise


def score(c, assets, candidates, records):
    from multimodal_jev.context_event_metrics import ContextEventMetrics
    spec = c['cache']['seal']
    root = Path(spec['root'])
    seal = bound(dict(path=root/'MANIFEST.json', sha256=spec['sha256']))
    dataset = c['cache']['dataset']
    rel = 'scoring/'+dataset+'/targets.json'
    targets = bound(dict(path=root/rel, sha256=seal['files'][rel]))
    if seal['threshold'] != .5 or seal['target_tuning_allowed'] is not False:
        raise ValueError('frozen scoring policy required')
    metrics = ContextEventMetrics(candidates, threshold=seal['threshold'])
    groups = {}
    empty = multiple = 0
    for asset, truth, row in zip(assets, targets, records, strict=True):
        if any(asset[k] != truth[k] for k in ('id','audio_sha256')) or row['id'] != asset['id']:
            raise ValueError('scoring identity differs from audited inference')
        group = truth['group']
        if group not in groups:
            groups[group] = ContextEventMetrics(candidates, threshold=seal['threshold'])
        for meter in (metrics, groups[group]):
            meter.add(row['id'], truth['gold'], row['prob'])
        n = sum(p >= seal['threshold'] for p in row['prob'])
        empty += n == 0
        multiple += n > 1
    overall = metrics.result()
    overall.update(mean_top1_probability=math.fsum(max(r['prob']) for r in records)/len(assets),
                   empty_selection_rate=empty/len(assets), multiple_selection_rate=multiple/len(assets))
    return dict(dataset=dataset, step=c['point']['step'], seal_sha256=spec['sha256'], metrics=overall,
                by_group={group: m.result() for group,m in groups.items()}, target_tuning_allowed=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    p.add_argument('--phase', choices=['hf', 'parity', 'full', 'audit-parity', 'audit-full'], required=True)
    a = p.parse_args()
    c, meta = load_config(a.config, a.local)
    run(c, meta, a.phase)
