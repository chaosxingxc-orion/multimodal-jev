"""Build or re-audit a Stage2 dataset from isolated, real clean TRAIN contrasts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from multimodal_jev import stage2_dataset_release as release
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_reference_dataset import ReferenceDataset, digest
from multimodal_jev.context_release_handle import ReleaseHandle
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.build_jev_boundary_panel import source_rows
from tools.reports.inventory_stage1_local_cues import local_edits
from tools.reports.jev_train_confusions import pinned
from tools.reports.prepare_stage2_dataset import save_receipt


def read(spec):
    return json.loads(pinned(Path(spec['path']), spec['sha256']))


def evidence_rules(spec):
    rules = read(spec)
    if rules['schema'] == 'stage1-cue-pairs-v1':
        return rules, None
    if rules['schema'] not in {'jev-stage2-paired-evidence-policy-v2','jev-stage2-paired-evidence-policy-v3'}:
        raise ValueError('frozen paired evidence policy required')
    ref = rules['lexical_rules']
    if Path(ref['name']).name != ref['name']:
        raise ValueError('lexical rules must be a sibling versioned config')
    lexical = read(dict(path=Path(spec['path']).parent/ref['name'], sha256=ref['sha256']))
    if lexical['schema'] != 'stage1-cue-pairs-v1':
        raise ValueError('frozen local cue rules required')
    return lexical, dict(policy_sha256=spec['sha256'], **rules)


def make_channel_pairs(metadata, policy):
    """Accept the corpus naming convention only when utterance metadata agrees."""
    close, distant = policy['close_suffix'], policy['distant_suffix']
    if not close or not distant or close == distant or not close.endswith(distant):
        raise ValueError('distinct close/distant recording suffixes required')
    roster = {r['id']:r for r in metadata}
    pairs, conflicts = [], 0
    for uid, row in sorted(roster.items()):
        if not uid.endswith(close):
            continue
        other = roster.get(uid.removesuffix(close)+distant)
        if other is None:
            continue
        if (row['audio_sha256'] == other['audio_sha256']
                or any(row[k] != other[k] for k in ('speaker','source','group','text','gold'))):
            conflicts += 1
            continue
        pairs.append([uid, other['id']])
    return pairs, conflicts


def compile_evidence(rows, c):
    """Construct contrast identities without consulting any model outputs."""
    _, metadata = source_rows(c)
    matched = {x['id']: x for x in metadata}
    if (len(matched) != len(metadata) or set(matched) != {x['id'] for x in rows}
            or any((r['audio_sha256'], r['intent'], r['source_group']) !=
                   tuple(matched[r['id']][k] for k in ('audio_sha256', 'gold', 'group')) for r in rows)):
        raise ValueError('TRAIN metadata does not match the admitted population')
    policy = c.get('evidence_policy')
    full_train = policy is not None and policy['schema']=='jev-stage2-paired-evidence-policy-v3'
    if full_train:
        if c['probe'] is not None or c['probe_manifest_sha256'] is not None:
            raise ValueError('full TRAIN policy does not consume historical probes')
        protected, fields = [], ()
    else:
        seal = Path(c['probe'])
        m = json.loads(pinned(seal/'MANIFEST.json', c['probe_manifest_sha256']))
        files = m['files']
        targets = json.loads(pinned(seal/'scoring/targets.json', files['scoring/targets.json']))
        support = json.loads(pinned(seal/'support.json', files['support.json']))
        members = targets + list(support.values())
        protected = [matched[r['id']] for r in members]
        if any(t['audio_sha256'] != p['audio_sha256'] for t,p in zip(members,protected,strict=True)):
            raise ValueError('protected probe and TRAIN audio differ')
        fields = tuple(policy['holdout_keys']) if policy is not None else ('audio_sha256','speaker','source','group','text')
        if len(fields) != len(set(fields)) or set(fields) not in (
                {'audio_sha256','source','group','text'}, {'audio_sha256','source','group','text','speaker'}):
            raise ValueError('probe audio, source group and wording must stay isolated')
    blocked = {k: {r[k] for r in protected} for k in fields}
    eligible = [r for r in sorted(metadata, key=lambda r: r['id']) if r['speaker'] and
                r['text'] and all(r[k] not in blocked[k] for k in fields)]
    text_labels = defaultdict(set)
    for row in metadata:
        text_labels[row['text']].add(row['gold'])
    # Conflicting labels cannot supply a positive determinacy control.
    eligible = [r for r in eligible if len(text_labels[r['text']]) == 1]

    lookup = defaultdict(list)
    for row in eligible:
        lookup[row['gold'], row['text']].append(row)
    cohorts = []
    for _, group in sorted(lookup.items()):
        speakers = defaultdict(list)
        for r in group:
            speakers[r['speaker']].append(r['id'])
        if len(speakers) > 1:
            cohorts.append([speakers[s] for s in sorted(speakers)])

    families = c['families']
    by_label = {label: family for family in families for label in family}
    if any(len(f) != 2 for f in families) or len(by_label) != 2 * len(families):
        raise ValueError('distinct binary cue families required')
    contrast = local_edits(eligible, {frozenset(x) for x in families},
        c['max_changed_words'], c['min_shared_words'])
    pairs = set()
    for p in contrast:
        a, b = p['labels']
        edits = p['edits']
        if p['kind'] == 'flip':
            if not all(set(e.split()) & set(by_label[g][g]) and not set(e.split()) & set(by_label[g][w])
                       for e, g, w in zip(edits, (a, b), (b, a), strict=True)):
                continue
        elif not ('' in edits and set(edits) & set(c['preserve_edge_insertions'])
                  and (p['frame'].startswith('<edit> ') or p['frame'].endswith(' <edit>'))):
            continue
        left, right = [lookup[label, matched[uid]['text']] for label, uid in
                       zip((a, b), p['references'], strict=True)]
        for x in left:
            for y in right:
                # Local lexical edits also need a same-speaker control; an
                # uncontrolled speaker change cannot isolate the edit.
                if x['speaker'] == y['speaker']:
                    pairs.add(('cue_'+p['kind'], *sorted((x['id'], y['id']))))
    evidence = dict(schema='jev-paired-train-evidence-v1', release_sha256=c['clean']['sha256'],
        rows_sha256=digest(sorted(rows, key=lambda r: r['id'])),
        native_metadata_sha256=c['native']['sha256'], protected_seal_sha256=c['probe_manifest_sha256'],
        eligible_ids=[r['id'] for r in eligible], same_words=cohorts, local_pairs=[list(p) for p in sorted(pairs)])
    if policy is not None:
        channel, conflicts = make_channel_pairs(eligible, policy['channel'])
        evidence.update(schema='jev-paired-train-evidence-v3' if full_train else 'jev-paired-train-evidence-v2', channel_pairs=channel,
            channel_conflicts_excluded=conflicts, holdout_keys=list(fields),
            evidence_policy_sha256=policy['policy_sha256'], lexical_rules_sha256=policy['lexical_rules']['sha256'])
    return evidence


def run(c, provenance, *, manifest_sha256=None):
    if c['schema'] != 'jev-stage2-paired-dataset-preparation-v1':
        raise ValueError('explicit paired dataset preparation schema required')
    release.validate_execution(c['execution'])
    out, receipt, bound = (Path(c[k]).resolve() for k in ('output','receipt','stage2_output'))
    frozen = Path(c['stage2_config']['path']).resolve()
    paths = (out, receipt, bound, frozen)
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(paths) for b in paths[:i]):
        raise ValueError('new disjoint view, receipt and config outputs required')
    check_paths({'paired_dataset': out, 'receipt': receipt, 'bound_config': bound})
    base = read(c['stage2_config'])
    audited = read(c['reference_audit'])
    proof = audited['output_audit']
    ref = base['reference_dataset']
    if (audited['status'] != 'reference_job_and_same_endpoint_parity_audited'
            or proof['status'] != 'reference_dataset_journal_audited'
            or proof['manifest_sha256'] != ref['manifest_sha256']
            or proof['policy_id'] != ref['identity']['policy_id']
            or proof['stage1_commit_sha256'] != c['checkpoint']['sha256']
            or c['clean']['sha256'] != ref['identity']['release_sha256']
            or proof['recordings'] != len(ref['population'])):
        raise ValueError('pinned independent reference audit does not bind the Stage1 endpoint')
    reference = ReferenceDataset(c['reference_collection'], manifest_sha256=ref['manifest_sha256'],
        expected_identity=ref['identity'], expected_population=ref['population'], execution=c['execution'])
    rows, task, _, _ = ReleaseHandle().load(c['clean']['root'], c['clean']['sha256'], 'learning', 'optimize')
    cue, policy = evidence_rules(c['cue_rules'])
    evidence = compile_evidence(rows, {**cue, **{k:c.get(k) for k in ('clean','native','probe','probe_manifest_sha256')},
        'evidence_policy':policy})
    options = dict(c['replay'], mode='paired_train_v1', evidence=evidence)
    endpoint = release.endpoint_binding(read(c['checkpoint']), commit_sha256=c['checkpoint']['sha256'],
        reference=reference, source_expected=c['stage1_expected'], sampling=options)
    args = dict(rows=rows, task=task, reference=reference, endpoint=endpoint, options=options,
        batch_config=c['batch_config'], evaluation=c['evaluation'], execution=c['execution'])
    if manifest_sha256 is not None:
        result = release.audit(out, manifest_sha256, **args)
    else:
        result = release.publish(out, **args)
    config = release.bind_config(base, root=out, manifest_sha256=result['manifest_sha256'],
        consumer=result['consumer'], reference=reference)
    if manifest_sha256 is None:
        save_receipt(bound, config)
    elif json.loads(bound.read_text()) != config:
        raise ValueError('training config differs from independent dataset binding')
    result = dict(result, bound_config_sha256=file_hash(bound), reference_audit_sha256=c['reference_audit']['sha256'],
        cue_rules_sha256=c['cue_rules']['sha256'], evidence_builder_sha256=file_hash(Path(__file__)),
        config_provenance=provenance)
    if manifest_sha256 is None:
        save_receipt(receipt, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    parser.add_argument('--audit-manifest-sha256')
    args = parser.parse_args()
    result = run(*load_config(args.config, args.local), manifest_sha256=args.audit_manifest_sha256)
    s = result['summary']
    print(json.dumps(dict(status=result['status'], manifest_sha256=result['manifest_sha256'],
        consumer=result['consumer'], summary={k:v for k,v in s.items() if k != 'by_class'}), indent=2))
