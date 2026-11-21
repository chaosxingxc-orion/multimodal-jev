"""Independently score complete paired predictions against a semantic audio seal."""
import math
from pathlib import Path

from multimodal_jev.paired_event_decisions import compare
from tools.reports.run_native_seal_endpoint import bound


def score(spec, jobs, noul, choice, policies, protocol):
    dataset = spec['dataset']
    seal = spec['seal']
    root = Path(seal['root'])
    manifest = bound(dict(path=root/'MANIFEST.json', sha256=seal['sha256']))
    data = manifest['datasets'][dataset]
    rel = f'scoring/{dataset}/targets.json'
    targets = bound(dict(path=root/rel, sha256=manifest['files'][rel]))
    ids = [q.id for q in jobs[0][2].questions]
    if (manifest['schema'] != 'jev-semantic-evaluation-seal-v1'
            or manifest['target_tuning_allowed'] is not False
            or len(jobs) != data['recordings'] or len(ids) != data['candidates']
            or len({t['group'] for t in targets}) != data['utterance_groups']
            or set(noul) != set(choice) or set(noul) != set(policies)
            or set(policies) != set(protocol['pair']) or len(set(policies.values())) != len(policies)):
        raise ValueError('complete frozen semantic panels and paired endpoints required')
    shared = protocol['shared_label_space']
    if type(shared) is not bool or (not shared and 'interval' in protocol):
        raise ValueError('semantic macro F1 requires shared labels across recordings')
    for (name, row, request), target in zip(jobs, targets, strict=True):
        if (name != dataset or any(row[k] != target[k] for k in ('id', 'audio_sha256'))
                or [q.id for q in request.questions] != ids or target['gold'] not in ids):
            raise ValueError('semantic scoring identity or candidate order differs')
    truth = {t['id']: ids.index(t['gold']) for t in targets}
    n = len(truth)
    metrics, choice_metrics, maps, choice_maps = {}, {}, {}, {}
    for arm, policy_id in policies.items():
        if set(noul[arm]) != set(truth) or set(choice[arm]) != set(truth):
            raise ValueError('missing semantic predictions')
        probs, ranked = {}, {}
        correct = choice_correct = rejected = 0
        top, choice_top, nll, brier = [], [], [], []
        for uid, y in truth.items():
            values, logp = noul[arm][uid]['prob'], choice[arm][uid]['logp']
            if (len(values) != len(ids) or len(logp) != len(ids) + 1
                    or not all(math.isfinite(v) and 0 <= v <= 1 for v in values)
                    or not all(math.isfinite(v) and v <= 0 for v in logp)
                    or not math.isclose(math.fsum(map(math.exp, logp)), 1, abs_tol=1e-5)):
                raise ValueError('invalid semantic prediction vector')
            probs[uid] = dict(zip(ids, values, strict=True))
            valid = logp[:-1]
            offset = max(valid)
            normalizer = offset + math.log(math.fsum(math.exp(v-offset) for v in valid))
            conditional = [math.exp(v-normalizer) for v in valid]
            ranked[uid] = dict(zip(ids, conditional, strict=True))
            correct += values.index(max(values)) == y
            choice_correct += valid.index(max(valid)) == y
            rejected += logp.index(max(logp)) == len(ids)
            top.append(max(values))
            choice_top.append(max(conditional))
            nll.append(normalizer-valid[y])
            brier.append(math.fsum((p-(i == y))**2 for i, p in enumerate(conditional)))
        metrics[arm] = dict(policy_id=policy_id, queries=n, top1_accuracy=correct/n,
                            mean_top1_probability=math.fsum(top)/n)
        choice_metrics[arm] = dict(policy_id=policy_id, queries=n, top1_accuracy=choice_correct/n,
                                   mean_top1_probability=math.fsum(choice_top)/n,
                                   false_none_rate=rejected/n, categorical_nll=math.fsum(nll)/n,
                                   categorical_brier=math.fsum(brier)/n)
        maps[arm], choice_maps[arm] = probs, ranked
    paired = {}
    if shared:
        left, right = protocol['pair']
        for name, views, result in (('noul', maps, metrics), ('choice', choice_maps, choice_metrics)):
            proof = compare(ids, targets, views[left], views[right], interval=protocol['interval'], cluster='group')
            for arm, side in ((left, 'left'), (right, 'right')):
                result[arm]['top1_macro_f1'] = proof[side]['top1_macro_f1']
            paired[name] = dict(cluster_count=proof['clusters'],
                                macro_f1_delta_ci95=proof['macro_f1_delta_ci95'])
    return dict(metrics, choices=choice_metrics, paired=paired)
