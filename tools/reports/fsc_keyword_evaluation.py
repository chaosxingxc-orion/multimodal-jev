"""Seal and score paired FSC VALID prompt variants with the native Jev runtime."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import math
from pathlib import Path
import random
from statistics import quantiles
import tempfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_decision_contract import PREFIX_LAYOUT, render_prefix_context_request
from multimodal_jev.fsc_span_transfer import FIELDS, validate
from multimodal_jev.fsc_transfer_population import official_sources
from multimodal_jev.local_checkpoint import commit_json, sync_directory
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.jev.prepare_fsc_context_transfer import compile_native
from tools.reports.fsc_annotation_cues import mine
from tools.reports.prepare_native_seal_cache import plan
from tools.reports.summarize_audited_top1 import rescore, verified_rows


def read(path, expected=None):
    if expected is not None and file_hash(path) != expected:
        raise ValueError(f'Content changed: {path}')
    return json.loads(Path(path).read_text())


def command_values(commands):
    return {c['id']: tuple(c['values'][key] for key in FIELDS) for c in commands}


def make_panels(protocol, task, cues):
    validate(task)
    request = compile_native('<audio>', task)
    commands = command_values(task['commands'])
    labels = list(commands)
    arms = protocol['arms']
    combos = {k: (v['keywords'], v['examples']) for k, v in arms.items()}
    if (combos != {'baseline': (False, False), 'keywords': (True, False),
                   'examples': (False, True), 'keywords_examples': (True, True)}
            or any(type(flag) is not bool for pair in combos.values() for flag in pair)):
        raise ValueError('Complete crossed keyword/example design required')
    if set(cues['candidates']) != set(labels):
        raise ValueError('Mined cues must cover the fixed native command IDs')
    panels = {}
    for arm, (use_keywords, use_examples) in combos.items():
        questions = []
        for original in request.questions:
            candidate = cues['candidates'][original.id]
            definition = original.definition
            if use_keywords:
                keys = {k: ', '.join(term['word'] for term in candidate['fields'][k]['selected']) for k in FIELDS}
                hints = '; '.join(f'{k}: {terms}' for k, terms in keys.items() if terms)
                if hints:
                    definition += ' Possible spoken cue words (illustrative): ' + hints + '.'
            if use_examples:
                definition += ' Example (illustrative): ' + json.dumps(candidate['example']['text'], ensure_ascii=False)
            questions.append(replace(original, definition=definition))
        req = replace(request, questions=tuple(questions))
        system, history, readouts = render_prefix_context_request(req)
        if history:
            raise ValueError('FSC native evaluation requires no history')
        panels[arm] = dict(system=system, readouts=readouts, questions=req.model_input()['questions'],
            requirements=req.requirements, version=req.version, candidate_ids=labels,
            prompt_layout=PREFIX_LAYOUT, history=[])
    return panels


def verify(root, manifest):
    if manifest['schema'] != 'jev-native-evaluation-seal-v1' or manifest['purpose'] != 'fsc_keyword_prompt_development':
        raise ValueError('Expected FSC keyword development seal')
    for relative, digest in manifest['files'].items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != digest:
            raise ValueError(f'Sealed file changed: {relative}')
    return manifest


def prepare(c, provenance):
    protocol = read(c['protocol'])
    task = read(c['native_task'], protocol['native_task_sha256'])
    endpoint = protocol['endpoint']
    source = read(c['source_plan'], endpoint['source_plan_sha256'])
    point = source['checkpoints'][-1]
    if (protocol['schema'] != 'jev-fsc-keyword-evaluation-v1' or protocol['split'] != 'valid'
            or source['status'] != 'audited_source_policies_bound_evaluation_pending'
            or source['source_audit']['status'] != 'formal_infonce_artifacts_audited'
            or source['reference_endpoint'] != {k: endpoint[k] for k in ('step', 'policy_id')}
            or any(point[k] != endpoint[k] for k in ('step', 'policy_id'))):
        raise ValueError('Official FSC VALID and fixed audited Stage1 endpoint required')
    output = Path(c['output']).resolve()
    if output.exists():
        manifest = verify(output, read(output/'MANIFEST.json'))
        if manifest['provenance'] != provenance or manifest['protocol_sha256'] != file_hash(c['protocol']):
            raise ValueError('Existing seal belongs to a different preparation')
        pin = dict(root=str(output), sha256=file_hash(output/'MANIFEST.json'))
        for dataset in manifest['arm_datasets'].values():
            plan(dict(seal=pin, dataset=dataset))
        return manifest
    check_paths({'FSC evaluation seal': output})
    dataset_root = Path(c['dataset_root']).resolve()
    tables = official_sources(dataset_root, protocol['source_csv_sha256'])
    cues = mine(tables['train'], protocol['annotation_mining'], command_values(task['commands']))
    cues['source_csv_sha256'] = protocol['source_csv_sha256']['train']
    panels = make_panels(protocol, task, cues)
    rows = tables['valid']
    if (len(rows) != protocol['population']['recordings']
            or len({r['speakerId'] for r in rows}) != protocol['population']['speakers']):
        raise ValueError('Complete speaker-disjoint official VALID population required')

    def asset(row):
        path = dataset_root / row['path']
        return dict(id='fsc:'+row['path'], audio=str(path), audio_sha256=file_hash(path))

    with ThreadPoolExecutor(max_workers=protocol['execution']['io_workers']) as pool:
        roster = list(pool.map(asset, rows))
    mapping = {v: k for k, v in command_values(task['commands']).items()}
    targets = [dict(id=r['id'], audio_sha256=r['audio_sha256'], group=row['speakerId'],
                    gold=mapping[tuple(row[k] for k in FIELDS)])
               for r, row in zip(roster, rows, strict=True)]
    if len({r['audio_sha256'] for r in roster}) != len(roster) or {t['gold'] for t in targets} != set(mapping.values()):
        raise ValueError('Unique audio and all official VALID classes required')
    datasets = {arm: protocol['dataset']+'_'+arm for arm in panels}
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.fsc-keywords-') as parent:
        stage = Path(parent)/'seal'
        stage.mkdir()
        commit_json(stage/'protocol.json', protocol)
        commit_json(stage/'commands.json', task['commands'])
        commit_json(stage/'annotation-cues.json', cues)
        files = ['protocol.json', 'commands.json', 'annotation-cues.json']
        reports = {}
        for arm, dataset in datasets.items():
            inp = stage/'inference'/dataset
            gold = stage/'scoring'/dataset
            inp.mkdir(parents=True)
            gold.mkdir(parents=True)
            commit_json(inp/'panel.json', panels[arm])
            if arm == 'baseline':
                commit_json(inp/'rows.json', roster)
                commit_json(gold/'targets.json', targets)
            else:
                (inp/'rows.json').hardlink_to(stage/'inference'/datasets['baseline']/'rows.json')
                (gold/'targets.json').hardlink_to(stage/'scoring'/datasets['baseline']/'targets.json')
            reports[dataset] = dict(split='official_valid', recordings=len(roster),
                                    speakers=protocol['population']['speakers'], candidates=len(mapping))
            files.extend(str(p.relative_to(stage)) for p in (inp/'panel.json', inp/'rows.json', gold/'targets.json'))
        manifest = dict(schema='jev-native-evaluation-seal-v1', purpose='fsc_keyword_prompt_development',
            datasets=reports, arm_datasets=datasets, endpoint=endpoint, prompt_layout=PREFIX_LAYOUT, threshold=.5,
            protocol_sha256=file_hash(c['protocol']), source_only_policy_selection=True, target_tuning_allowed=False,
            test_feedback_used=False, files={p: file_hash(stage/p) for p in files}, provenance=provenance,
            stable_prefix_encoded_cache_ready=False, npu_admitted=False)
        commit_json(stage/'MANIFEST.json', manifest)
        pin = dict(root=str(stage), sha256=file_hash(stage/'MANIFEST.json'))
        for dataset in datasets.values():
            if len(plan(dict(seal=pin, dataset=dataset))) != len(roster):
                raise ValueError('Native runner did not reconstruct the full arm')
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
        sync_directory(output.parent)
    return manifest


def matched_pairs(values):
    """Compare every legal candidate differing only in one annotated field."""
    pairs = {gold: [] for gold in values}
    for gold, expected in values.items():
        for other, alternative in values.items():
            differences = [field for field, a, b in zip(FIELDS, expected, alternative, strict=True) if a != b]
            if len(differences) == 1:
                pairs[gold].append((differences[0], other))
    return pairs


def speaker_stats(rows, targets, labels, pairs):
    stats = {}
    lookup = {k: i for i, k in enumerate(labels)}
    for row, target in zip(rows, targets, strict=True):
        if row['id'] != target['id']:
            raise ValueError('Prediction roster differs')
        p = row['prob']
        gold = target['gold']
        best = labels[max(range(len(p)), key=p.__getitem__)]
        s = stats.setdefault(target['group'], dict(n=0, maximum=0., gold=[0]*len(labels),
            pred=[0]*len(labels), hits=[0]*len(labels), matched={k: [0, 0] for k in FIELDS}))
        s['n'] += 1
        s['maximum'] += max(p)
        s['gold'][lookup[gold]] += 1
        s['pred'][lookup[best]] += 1
        if best == gold:
            s['hits'][lookup[gold]] += 1
        for field, other in pairs[gold]:
            a, b = p[lookup[gold]], p[lookup[other]]
            s['matched'][field][0] += 2 if a > b else 1 if a == b else 0
            s['matched'][field][1] += 1
    return stats


def summary(stats, speakers):
    groups = [stats[s] for s in speakers]
    n = sum(x['n'] for x in groups)
    hits = sum(sum(g['hits']) for g in groups)
    f1 = []
    for i in range(len(groups[0]['gold'])):
        denom = sum(g['gold'][i]+g['pred'][i] for g in groups)
        f1.append(2*sum(g['hits'][i] for g in groups)/denom if denom else 0.)
    result = dict(recordings=n, top1_accuracy=hits/n, top1_macro_f1=math.fsum(f1)/len(f1),
                  top1_mean_probability=math.fsum(g['maximum'] for g in groups)/n)
    for k in FIELDS:
        twice_wins = sum(g['matched'][k][0] for g in groups)
        count = sum(g['matched'][k][1] for g in groups)
        result['matched_'+k] = twice_wins/(2*count) if count else None
        result['matched_'+k+'_comparisons'] = count
    return result


def score(root, seal_sha, arms):
    manifest = verify(root, read(root/'MANIFEST.json', seal_sha))
    protocol = read(root/'protocol.json')
    datasets = manifest['arm_datasets']
    endpoint = manifest['endpoint']
    if set(arms) != set(datasets):
        raise ValueError('Exactly four matched audits required')
    values = command_values(read(root/'commands.json'))
    labels = list(values)
    pairs = matched_pairs(values)
    stats, results, evidence = {}, {}, {}
    target_path = root/'scoring'/datasets['baseline']/'targets.json'
    targets = read(target_path)
    for arm, dataset in datasets.items():
        bind = arms[arm]
        audit = read(bind['audit'], bind['audit_sha256'])
        if (audit['seal_sha256'] != seal_sha or audit['dataset'] != dataset
                or audit['policy_id'] != endpoint['policy_id'] or audit['step'] != endpoint['step']
                or audit['metrics']['candidate_ids'] != labels
                or manifest['files'][f'scoring/{dataset}/targets.json'] != file_hash(target_path)):
            raise ValueError('Arm must bind the same seal, endpoint, candidate order and target roster')
        audited = rescore(bind['predictions'], audit, seal_root=root)
        rows = verified_rows(bind['predictions'], sha256=audited['predictions_sha256'],
                             recordings=len(targets), decisions=len(labels), policy_id=endpoint['policy_id'])
        stats[arm] = speaker_stats(rows, targets, labels, pairs)
        results[arm] = summary(stats[arm], sorted(stats[arm]))
        evidence[arm] = dict(audit_sha256=bind['audit_sha256'], predictions_sha256=audited['predictions_sha256'])
    config = protocol['analysis']
    rng = random.Random(config['seed'])
    speakers = sorted(stats['baseline'])
    metrics = config['primary_metrics']
    contrasts = {arm: {key: [] for key in metrics} for arm in datasets if arm != 'baseline'}
    for _ in range(config['paired_bootstrap_samples']):
        sample = rng.choices(speakers, k=len(speakers))
        control = summary(stats['baseline'], sample)
        for arm, fields in contrasts.items():
            treatment = summary(stats[arm], sample)
            for key, deltas in fields.items():
                if control[key] is not None:
                    deltas.append(treatment[key]-control[key])
    bins = 1000
    low_percentile = round((1-config['confidence_level'])/2 * bins)
    if not 0 < low_percentile < bins/2:
        raise ValueError('Confidence level outside supported percentile grid')
    deltas = {}
    for arm, fields in contrasts.items():
        deltas[arm] = {}
        for key, values in fields.items():
            qs = quantiles(values, n=bins, method='inclusive')
            deltas[arm][key] = dict(delta=results[arm][key]-results['baseline'][key],
                paired_speaker_interval=[qs[low_percentile-1], qs[bins-low_percentile-1]],
                n_bootstrap=len(values))
    primary = deltas[config['primary_contrast']]
    supported = all(v['paired_speaker_interval'][0] > config['min_primary_delta'] for v in primary.values())
    return dict(seal_sha256=seal_sha, endpoint=endpoint, split='official_valid', speaker_clusters=len(speakers),
                keyword_primary_criterion_met=supported,
                results=results, contrasts_vs_baseline=deltas, analysis=config, evidence=evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    parser.add_argument('--results-local', help='Resolve the four audited arm paths and score the frozen seal')
    args = parser.parse_args()
    c, provenance = load_config(args.config, args.local)
    if args.results_local:
        binding, _ = load_config('configs/portable/operations/fsc-keyword-results.json', args.results_local)
        if Path(c['output']).resolve() != Path(binding['seal']['root']).resolve():
            raise ValueError('Result binding points to a different seal')
        result = score(Path(binding['seal']['root']), binding['seal']['sha256'], binding['arms'])
        commit_json(binding['report'], result)
        print(json.dumps(result, indent=2))
    else:
        manifest = prepare(c, provenance)
        print(json.dumps(dict(output=c['output'], seal_sha256=file_hash(Path(c['output'])/'MANIFEST.json'),
                              datasets=manifest['datasets'], npu_admitted=False), indent=2))


if __name__ == '__main__':
    main()
