"""Paired, semantic-ID rescore of complete fixed-HF diagnostic runs."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shlex
import subprocess

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_decision_contract import ApplicabilityQuestion, ContextDecisionRequest
from multimodal_jev.context_reference_dataset import digest
from multimodal_jev.decision_evidence import DecisionEvidence, metric
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.paired_event_decisions import compare
from multimodal_jev.slurp_training import content_hash
from tools.reports.jev_train_confusions import pinned


def calendar_contrasts(targets, predictions, threshold):
    expected = ('forward', 'reverse', 'missing', 'irrelevant')
    if set(targets) != set(expected) or set(predictions) != set(expected):
        raise ValueError('all four calendar arms are required')
    groups = {name: [] for name in ('baseline_wrong', 'reverse_wrong', 'missing_wrong', 'irrelevant_wrong', 'all_correct')}
    per_arm = {}
    correctness = {}
    ids = [r['id'] for r in targets['forward']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('unique nonempty calendar roster required')
    keys = {'intent', 'intent_sufficient', 'availability', 'availability_sufficient'}
    for arm in expected:
        if set(predictions[arm]) != set(ids):
            raise ValueError('complete calendar predictions required')
        decisions = []
        correctness[arm] = {}
        for uid, row in zip(ids, targets[arm], strict=True):
            p = predictions[arm][uid]
            gold = row['gold']
            if uid != row['id'] or set(gold) != keys or set(p) != keys:
                raise ValueError('calendar question or recording identity differs')
            answer = p['availability'] >= threshold
            sufficient = p['availability_sufficient'] >= threshold
            decisions.append((DecisionEvidence(gold['availability_sufficient'], gold['availability']),
                              p['availability_sufficient'], answer))
            correctness[arm][uid] = (
                (p['intent'] >= threshold) == gold['intent']
                and (p['intent_sufficient'] >= threshold) == gold['intent_sufficient']
                and sufficient == gold['availability_sufficient']
                and (not sufficient or answer == gold['availability']))
        per_arm[arm] = dict(**metric(decisions, threshold=threshold), all_questions_correct=sum(correctness[arm].values()))
    for uid in ids:
        if not correctness['forward'][uid]:
            groups['baseline_wrong'].append(uid)
            continue
        failures = [arm for arm in expected[1:] if not correctness[arm][uid]]
        for arm in failures:
            groups[arm + '_wrong'].append(uid)
        if not failures:
            groups['all_correct'].append(uid)
    return dict(recordings=len(ids), per_arm=per_arm, recording_ids=groups,
                counts={k:len(v) for k,v in groups.items()}, threshold=threshold)


def score(c, *, operations=None):
    root=Path(c['output']);ops=root/'hf/run' if operations is None else Path(operations);seal=Path(c['seal'])
    read=lambda p:json.loads(Path(p).read_bytes())
    m=json.loads(pinned(seal/'MANIFEST.json',c['seal_sha256']))
    get=lambda name:json.loads(pinned(seal/name,m['files'][name]))
    if read(ops/'terminal.json')!=dict(exit_code=0,status='declared_stages_finished'):
        raise ValueError('complete HF lifecycle required')
    diagnostic=m.get('diagnostic')
    calendar=diagnostic=='calendar_fact_availability'
    layout=diagnostic=='full_train_unseen_layout'
    automatic=diagnostic=='automatic_class_boundary'
    if not calendar:
        targets=get('scoring/targets.json');pairs=get('scoring/pairs.json')
    canonical_order = get('inference/original/panel.json')['candidate_ids'] if automatic else []
    receipt=read(root/'receipt.json')
    policies=receipt['policy'];out={};guesses={}
    if (calendar or layout or automatic) and (policies!=c['expected_policy_id'] or receipt['seal_sha256']!=c['seal_sha256']):
        raise ValueError('collector must bind this round and the declared seal')
    calendar_targets, calendar_predictions = {}, {}
    layout_probabilities = {}
    for arm in m['datasets']:
        role='hf-'+arm
        handle=read(ops/(role+'-launch.json'))
        obj=read(ops/(role+'-inspect.json'));wait=read(ops/(role+'-wait.json'))
        if (obj['Id']!=handle['container_id'] or obj['State']['Status']!='exited' or obj['State']['OOMKilled']
                or obj['State']['ExitCode']!=0 or wait!={'exit_code':0}):raise ValueError('unverified endpoint exit')
        pred=read(ops/(role+'-collected.json'))
        # Bind row and candidate identities from this actual HF run, not an inferred order.
        panelpath=Path(handle['remote_root'])/role/'panel.json'
        cmd=['ssh','-o','BatchMode=yes',handle['host'],shlex.join(['cat',str(panelpath)])]
        panel=json.loads(subprocess.check_output(cmd,timeout=60))
        roster=get(f'inference/{arm}/rows.json');sealed=get(f'inference/{arm}/panel.json');labels=sealed['candidate_ids']
        if (pred['policy_id']!=policies or pred['panel_sha256']!=digest(panel) or panel['dataset']!=arm
                or panel['candidate_ids']!=labels or panel['labels_used'] is not False):
            raise ValueError('actual HF panel or policy mismatch')
        bucket={};win={};conf=Counter();ranks={}
        if calendar:
            targets=get(f'scoring/{arm}.json')
            calendar_targets[arm]=targets
            calendar_predictions[arm]={}
        if layout:
            layout_probabilities[arm] = {}
        if calendar or layout or automatic:
            questions=tuple(ApplicabilityQuestion(**q) for q in sealed['questions'])
        for tr,asset,prior,r,row in zip(targets,panel['assets'],panel['input_receipts'],pred['records'],roster,strict=True):
            uid=tr['id'];g=tr['gold'];p=r['prob'];ident=r['request_identity']
            if (any(v['id']!=uid for v in (r,row,asset,prior)) or row['audio_sha256']!=asset['audio_sha256']
                    or row['audio_sha256']!=tr['audio_sha256'] or len(p)!=len(labels)
                    or len(r['logits'])!=len(labels) or not all(type(v) in (float,int) and 0<=v<=1 for v in p)
                    or ident!=dict(policy_id=policies,positions=prior['positions'],inputs=prior['inputs'])
                    or any(not math.isfinite(v) for v in r['logits'])):
                raise ValueError('HF record does not match source audio or policy')
            if calendar or layout or automatic:
                request=ContextDecisionRequest(row['audio'], questions, tuple(sealed['requirements']),
                                               version=sealed['version'])
                if asset['request_sha256']!=content_hash(request.model_input()):
                    raise ValueError('HF request does not match the sealed diagnostic intervention')
            if calendar:
                calendar_predictions[arm][uid]=dict(zip(labels,p,strict=True))
                continue
            if layout:
                layout_probabilities[arm][uid] = dict(zip(labels, p, strict=True))
            order=sorted(range(len(p)), key=(lambda i:(-p[i], canonical_order.index(labels[i]))) if automatic
                         else (lambda i:-p[i]))
            win[uid]=labels[order[0]];ranks[uid]=[labels[i] for i in order[:3]]
            b=bucket.setdefault(tr['stratum'],Counter());b['n']+=1;b['correct']+=win[uid]==g;b['top3']+=g in ranks[uid]
            if win[uid]!=g:conf[g+' → '+win[uid]]+=1
        guesses[arm]=win
        details=dict(fingerprint=pred['panel_sha256'],predictions_file_sha256=file_hash(ops/(role+'-collected.json')))
        out[arm]=details if calendar else dict(strata=bucket, errors=conf, pair_count=len(pairs),
                      direction_both_correct=sum(win[x['up']]=='audio_volume_up' and win[x['down']]=='audio_volume_down' for x in pairs),
                      **details)
    if calendar:
        for arm,rows in calendar_targets.items():
            if [(r['id'],r['audio_sha256']) for r in rows]!=[
                    (r['id'],r['audio_sha256']) for r in calendar_targets['forward']]:
                raise ValueError('same recordings and audio required across calendar arms')
        return dict(policy_id=policies,seal_sha256=c['seal_sha256'],arms=out,
                    scope='TRAIN synthetic calendar evidence diagnostic only',
                    contrasts=calendar_contrasts(calendar_targets,calendar_predictions,m['threshold']))
    if layout and 'layout_design' in c:
        design = c['layout_design']
        if design['sha256'] != m['design_sha256'] or set(layout_probabilities) != {'native', 'unseen'}:
            raise ValueError('the declared TRAIN layout design and both paired arms are required')
        protocol = json.loads(pinned(Path(design['path']), design['sha256']))
        native = get('inference/native/panel.json')['candidate_ids']
        unseen = get('inference/unseen/panel.json')['candidate_ids']
        if native == unseen:
            raise ValueError('the unseen candidate order must differ')
        paired = compare(native, targets, layout_probabilities['native'], layout_probabilities['unseen'],
                         interval=protocol['interval'], cluster=c['layout_cluster'], collect_ids=True)
        return dict(policy_id=policies, seal_sha256=c['seal_sha256'], arms=out,
                    scope='source TRAIN layout diagnostic only', contrasts=paired,
                    sampling=m.get('sampling'))
    changes={}
    evidence={}
    comparisons=([] if layout else c.get('comparisons',[
        ('original','boundary'),('boundary','fewshot'),('boundary','reversed_boundary')]))
    for old,new in comparisons:
        cells={}
        groups = {k: [] for k in ('fixed','regressed','both_correct','both_wrong')}
        for tr in targets:
            uid=tr['id'];g=tr['gold'];a,b=guesses[old][uid],guesses[new][uid]
            cell=cells.setdefault(tr['stratum'],Counter())
            cell['fixed']+=a!=g and b==g;cell['regressed']+=a==g and b!=g;cell['changed_label']+=a!=b
            groups['fixed' if a!=g and b==g else 'regressed' if a==g and b!=g
                   else 'both_correct' if a==g else 'both_wrong'].append(uid)
        changes[f'{old}_to_{new}']=cells
        evidence[f'{old}_to_{new}']=groups
    return dict(policy_id=policies,seal_sha256=c['seal_sha256'],arms=out,paired=changes,
                **(dict(recording_ids=evidence) if automatic else {}),
                scope='in-sample source TRAIN mechanism probe; HF only; no target or UNKNOWN labels')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,_=load_config(a.config,a.local);v=score(c);commit_json(Path(c['output'])/'paired-score.json',v)
