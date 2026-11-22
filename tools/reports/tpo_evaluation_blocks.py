"""Plan, run and independently collect a shared TPO evaluation queue."""
from functools import partial
import json
from pathlib import Path
import time

from multimodal_jev import evaluation_blocks as blocks
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import content_key, map_io
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.ordered_prefetch import OrderedPrefetch
from tools.reports.native_evaluation_cache import read_verified
from tools.reports.prepare_native_seal_cache import plan as sealed_jobs


def shape(row, name):
    dims = row['tensors'][name]['shape']
    if not dims or any(type(v) is not int or v < 1 for v in dims):
        raise ValueError('invalid audited evaluation tensor geometry')
    return dims[-1]


def roster(cache):
    manifest, proof = read_verified(cache)
    input_path = Path(cache['output']) / 'inputs.jsonl'
    result = []
    with input_path.open() as stream:
        for (dataset, asset, request), line in zip(sealed_jobs(cache), stream, strict=True):
            row = json.loads(line)
            if (row['dataset'], row['id'], row['audio_sha256']) != (dataset, asset['id'], asset['audio_sha256']):
                raise ValueError('block roster differs from the sealed population')
            features = row['tensors']
            audio = 'input_features_mask' if 'input_features_mask' in features else 'input_features'
            result.append(dict(id=asset['id'], tokens=shape(row, 'input_ids'),
                               frames=shape(row, audio), width=len(request.questions)))
    if len(result) != manifest['recordings']:
        raise ValueError('incomplete block roster')
    return result, dict(inputs_sha256=manifest['inputs_sha256'], seal_sha256=manifest['seal_sha256'],
                        manifest_sha256=proof['manifest_sha256'])


def build(spec):
    from tools.reports.evaluate_tpo_endpoints import endpoint_config, policy_sources, read
    from tools.reports.evaluate_choice_tasks import with_baseline

    start = time.monotonic()
    root = Path(spec['queue']['root'])
    check_paths({'evaluation block plan': root})
    root.mkdir()
    parent = read(spec['parent'])
    if file_hash(spec['parent']) != spec['parent_sha']:
        raise ValueError('evaluation parent changed')
    all_blocks, policies, inputs, arm_order = [], {}, {}, []
    choice = 'choice' in spec
    for dataset, endpoint in spec['endpoints'].items():
        c = endpoint_config(spec, parent, endpoint)
        sources = policy_sources(spec, c, spec['inferred_arms'][dataset])
        if choice:
            sources = with_baseline(spec, sources)
        for arm, (_, pid) in sources.items():
            if policies.setdefault(arm, pid) != pid:
                raise ValueError('inconsistent evaluation policy')
            if arm not in arm_order:
                arm_order.append(arm)
        if 'train_view' in endpoint:
            from tools.reports.calibrate_tpo_probabilities import source_panel
            from tools.reports.run_native_seal_endpoint import bound
            view = endpoint['train_view']
            _, assets, ids, _ = source_panel(view, parent)
            sizes = bound(view['lengths'])
            rows = [dict(id=a['id'], width=len(ids), **sizes[a['id']]) for a in assets]
            inputs[dataset] = dict(roster=view['roster']['sha256'], lengths=view['lengths']['sha256'],
                                   source=parent['release_sha256'], parent=spec['parent_sha'])
        else:
            rows, inputs[dataset] = roster(c['cache'])
        all_blocks.extend(blocks.make_blocks(dataset, rows, sources, batch=spec['batch'],
            batches_per_block=spec['queue']['batches_per_block'],
            partitions=spec['queue']['partitions'][dataset], choice=choice))
    if 'parity' in spec['queue']:
        limit = spec['queue']['parity']['blocks_per_dataset']
        # Include both long batches and short tails in the finite replay.
        selected = []
        for dataset in spec['endpoints']:
            for arm in spec['inferred_arms'][dataset]:
                candidates = [b for b in all_blocks if (b['dataset'], b['arm']) == (dataset, arm)]
                mixed = []
                while candidates and len(mixed) < limit:
                    longest = max(candidates, key=lambda b: (b['work'], len(b['ids'])))
                    candidates.remove(longest)
                    mixed.append(longest)
                    if candidates and len(mixed) < limit:
                        tail = min(candidates, key=lambda b: (len(b['ids']), b['work']))
                        candidates.remove(tail)
                        mixed.append(tail)
                selected.extend(mixed)
        all_blocks = selected
    plan = dict(schema=blocks.SCHEMA, namespace=spec['queue']['namespace'], inputs=inputs,
                policies=policies, choice=choice, blocks=blocks.ordered_blocks(all_blocks, arm_order))
    manifest = dict(plan=plan, sha256=content_key(plan))
    commit_json(root / 'plan.json', manifest)
    reused = 0
    if 'resume_from' in spec['queue']:
        prior = Path(spec['queue']['resume_from'])
        if blocks.checked_plan(prior, plan['namespace']) != manifest:
            raise ValueError('resume requires the identical evaluation plan')
        def restore(block):
            if not blocks.artifact(prior, block).is_file():
                return 0
            products = blocks.read_block(prior, manifest, block)
            blocks.publish(root, manifest, block, products)
            return 1
        restored, _ = map_io(restore, plan['blocks'], workers=spec['io_workers'])
        reused = sum(restored)
    result = dict(status='evaluation_blocks_prepared', plan_sha256=manifest['sha256'],
                  blocks=len(plan['blocks']), reused_blocks=reused, npu_executed=False,
                  seconds=time.monotonic()-start)
    commit_json(Path(spec['result_path']), result)
    return result


def worker(spec):
    import torch
    from multimodal_jev.cached_context_preparation import processor_cache_identity
    from multimodal_jev.context_policy_snapshot import load_policy
    from tools.reports.evaluate_tpo_endpoints import (
        endpoint_config, load, policy_sources, probabilities, read_parent, start_model,
    )
    from tools.reports.evaluate_choice_tasks import full_ontology, with_baseline

    root = Path(spec['queue']['root'])
    manifest = blocks.checked_plan(root, spec['queue']['namespace'])
    plan = manifest['plan']
    outputs = Path(spec['output'])
    completed = []
    model_loading = None
    start = time.monotonic()
    for block in plan['blocks']:
        with blocks.claim(root, block) as owned:
            if not owned:
                continue
            tick = time.monotonic()
            claimed_unix = time.time()
            dataset, arm = block['dataset'], block['arm']
            if model_loading is None:
                parent = read_parent(spec)
                outputs.mkdir()
                c = endpoint_config(spec, parent, spec['endpoints'][dataset])
                provider, model, device, processor, collate, handle = start_model(c, outputs)
                initial_processor = processor_cache_identity(c['snapshot'])
                current = c['policy_id']
                compiler = (provider.choice_compiler(processor, collate,
                            parent['view_execution']['max_input_tokens']) if plan['choice'] else None)
                loaded = {}
                model_loading = time.monotonic() - tick
            if dataset not in loaded:
                c = endpoint_config(spec, parent, spec['endpoints'][dataset])
                verify = dict(expected_cache=plan['inputs'][dataset])
                if 'train_view' in c:
                    from tools.reports.calibrate_tpo_probabilities import cached_source
                    view = c['train_view']
                    if (view['roster']['sha256'] != plan['inputs'][dataset]['roster']
                            or spec['parent_sha'] != plan['inputs'][dataset]['parent']):
                        raise ValueError('TRAIN population or input contract differs')
                    assets, _, encode = cached_source(view, parent)
                    requests = None
                elif compiler is None:
                    assets, candidates, encode = load(c, **verify)
                    if 'card_view' in c:
                        from multimodal_jev.card_evaluation_view import wrap
                        encode = wrap(c['card_view'], processor, candidates, encode)
                    requests = None
                else:
                    assets, _, encode, requests = load(c, include_requests=True, **verify)
                sources = policy_sources(spec, c, spec['inferred_arms'][dataset])
                if compiler is not None:
                    sources = with_baseline(spec, sources)
                if any(plan['policies'].get(a) != pid for a, (_, pid) in sources.items()):
                    raise ValueError('worker policies differ from block plan')
                loaded[dataset] = (c, {asset['id']: asset for asset in assets}, encode, sources, requests)
            c, assets, encode, sources, requests = loaded[dataset]
            path, pid = sources[arm]
            if pid != current:
                if processor_cache_identity(path) != initial_processor:
                    raise ValueError('changed processor invalidates paired evaluation')
                load_policy(path, model, expected_policy_id=pid,
                            expected_contract=c['contract'], expected_base_id=c['base_id'])
                current = pid
            rows = []
            first_batch_unix = None
            def encode_one(uid):
                return assets[uid], encode(assets[uid])[0]
            with OrderedPrefetch(encode_one, block['ids'], workers=spec['io_workers'],
                                 ahead=spec['queue']['prefetch']) as stream:
                def remember():
                    nonlocal first_batch_unix
                    for item in stream:
                        if compiler is not None:
                            rows.append(item)
                        if first_batch_unix is None:
                            first_batch_unix = time.time()
                        yield item
                products = {'prob': probabilities(model, device, provider, collate, pid, remember(),
                                                  spec['batch'], check_singleton=block['offset'] == 0)}
            if compiler is not None:
                def forward(batch, policy):
                    return provider.forward(model, batch, device=device, dtype=torch.bfloat16,
                                            policy_id=policy, **collate).float()
                products['choices'] = full_ontology(rows, requests, compiler, spec['batch'],
                                                     spec['choice']['temperature'], forward, pid)
            blocks.publish(root, manifest, block, products)
            completed.append(dict(block=block['key'], dataset=dataset, arm=arm,
                                  recordings=len(block['ids']), claimed_unix=claimed_unix,
                                  first_input_unix=first_batch_unix, committed_unix=time.time(),
                                  seconds=time.monotonic()-tick))
    result = dict(plan_sha256=manifest['sha256'], blocks=completed, model_load_seconds=model_loading,
                  seconds=time.monotonic()-start, devices=spec['devices'])
    commit_json(Path(spec['result_path']), result)
    return result


def publications(spec):
    """Audit every committed block before scoring; a missing block fails closed."""
    root = Path(spec['queue']['root'])
    manifest = blocks.checked_plan(root, spec['queue']['namespace'])
    plan = manifest['plan']
    if set(plan['inputs']) != set(spec['endpoints']):
        raise ValueError('missing dataset in evaluation block plan')
    for dataset, endpoint in spec['endpoints'].items():
        _, proof = read_verified(endpoint['cache'])
        if proof['manifest_sha256'] != plan['inputs'][dataset]['manifest_sha256']:
            raise ValueError('scoring inputs differ from block plan')
    expected = {block['key'] + '.json' for block in plan['blocks']}
    if len(expected) != len(plan['blocks']) or {p.name for p in root.glob('block-*.json')} != expected:
        raise ValueError('incomplete or unexpected evaluation block files')
    result = {}
    read_one = partial(blocks.read_block, root, manifest)
    payloads, _ = map_io(read_one, plan['blocks'], workers=spec['io_workers'])
    for block, products in zip(plan['blocks'], payloads, strict=True):
        for kind, payload in products.items():
            result.setdefault((block['dataset'], block['arm'], kind), []).append(payload)
    return result


def parity(spec):
    """Compare exactly the frozen sample blocks against pinned legacy predictions."""
    from tools.reports.run_native_seal_endpoint import bound

    predictions = publications(spec)
    comparisons = {}
    for dataset, arms in spec['queue']['parity']['legacy'].items():
        comparisons[dataset] = {}
        for arm, pins in arms.items():
            observed = predictions[dataset, arm, 'prob']
            policy = observed[0]['policy_id']
            expected = {}
            for pin in pins:
                ref = bound(pin)
                if ref['policy_id'] != policy:
                    raise ValueError('legacy parity policy differs')
                for row in ref['records']:
                    if row['id'] in expected:
                        raise ValueError('duplicate legacy parity record')
                    expected[row['id']] = row
            count = 0
            for payload in observed:
                for row in payload['records']:
                    if row != expected.get(row['id']):
                        raise ValueError(f'changed frozen prediction: {dataset}/{arm}/{row["id"]}')
                    count += 1
            comparisons[dataset][arm] = dict(recordings=count, policy_id=policy)
    result = dict(status='evaluation_block_parity_passed', comparisons=comparisons)
    commit_json(Path(spec['output']), result)
    return result
