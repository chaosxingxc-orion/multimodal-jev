"""Read-only native FSC comparison for the source-selected context policy pair.

Historical SpeechBrain evidence is rechecked from retained artifacts, without
requiring a deleted Docker object. This is explicitly archival provenance.
Kept outside tools/jev so it does not enter the queued inference package.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from multimodal_jev.baseline import file_hash
from multimodal_jev.fsc_published import FIELDS, parse_prediction
from multimodal_jev.fsc_reference_alignment import align
from multimodal_jev.fsc_transfer_population import scoring_targets
from multimodal_jev.fsc_transfer_report import command, interval
from tools.jev.run_context_baseline_sequence import check_pins
from tools.jev.score_fsc_context_pair import run as reaudited_score


def read(path):
    return json.loads(Path(path).read_text())


def authenticate_archive(spec):
    """No labels, inference, SSH or mutable container-state assumptions."""
    check_pins(spec['pins'])
    names = ('receipt', 'alignment_receipt', 'population', 'task')
    for name in names:
        if spec['pins'].get(spec[name]) != file_hash(spec[name]):
            raise ValueError('pinned archive inputs required')
    receipt = read(spec['receipt']); prior = read(spec['alignment_receipt'])
    root = Path(receipt['run']); historical = Path(prior['root'])/'result.json'
    result = read(root/'result.json'); identity = read(root/'identity.json')
    predictions_path = root/'predictions.jsonl'
    if (file_hash(historical) != prior['full_result_sha256']
            or file_hash(spec['receipt']) != prior['reference_receipt_sha256']
            or file_hash(spec['population']) != prior['population_sha256']
            or file_hash(predictions_path) != prior['predictions_sha256']
            or file_hash(predictions_path) != receipt['result']['predictions_sha256']
            or file_hash(root/'identity.json') != receipt['identity_sha256']
            or result != receipt['result'] or prior['passed'] is not True
            or receipt['independent_rescore_equal'] is not True
            or receipt['model_files_reverified'] is not True):
        raise ValueError('archived identity, predictions or prior proof changed')
    full = read(historical)
    if any(full.get(k) != v for k,v in prior.items() if k not in ('full_result_sha256','root')):
        raise ValueError('archived alignment summary differs')
    cid = receipt['container_id']; state = prior['actual_state']
    terminal = read(root.parent/(cid+'-terminal.json'))
    if (prior['reference_container_id'] != cid or terminal != dict(id=cid,state=state)
            or state['Status'] != 'exited' or state['ExitCode'] != 0
            or state['OOMKilled'] or state['Running'] or state['Dead']
            or read(root/'terminal.json')['exit_code'] != 0):
        raise ValueError('successful archived container terminal required')
    population = read(spec['population'])
    if (identity['config']['split'] != 'test' or population['recordings'] != 3793
            or identity['csv_sha256'] != population['source_csv_sha256']['test']
            or result['examples'] != 3793 or result['predictions'] != 3793 or result['missing'] != 0):
        raise ValueError('complete official reference population required')
    assets = {}
    for name, verification in identity['models'].items():
        model = Path(identity['config'][name])
        for entry in verification['files']:
            assets[str(model/entry['file'])] = entry['sha256']
    if not assets:raise ValueError('archived model identities required')
    check_pins(assets)
    predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    aligned = align(population,predictions)
    if (aligned['alignment'] != full['alignment'] or aligned['speakers'] != 10
            or result['joint_correct'] != prior['previously_verified_joint_correct']):
        raise ValueError('archived audio alignment differs')
    files = {**spec['pins'], **assets}
    for path in (historical, predictions_path, root/'identity.json', root/'result.json',
                 root/'terminal.json', root.parent/(cid+'-terminal.json')):
        files[str(path)] = file_hash(path)
    proof = dict(recordings=3793,speakers=10,reference_container_id=cid,
        runtime_provenance='retained_historical_terminal_and_independent_alignment',
        live_container_inspected=False,model_files_reverified=True,audio_files_reverified=True,
        predictions_sha256=file_hash(predictions_path),files=files)
    return proof, population, read(spec['task']), predictions, receipt


def compare_pair(task, roster, targets, predictions, policies, *, seed=941, replicates=2000):
    """Compare native top1 semantics; sequence outputs have no probability score."""
    if type(seed) is not int or type(replicates) is not int or replicates < 100:
        raise ValueError('fixed cluster bootstrap required')
    if set(policies) != {'initial','selected'}:raise ValueError('exact source-selected pair required')
    commands = {c['id']:command(c['values']) for c in task['commands']}
    if len(commands) != 31 or len(set(commands.values())) != 31:
        raise ValueError('unique native31 command ontology required')
    sources = {r['id']:r for r in roster}; gold = {r['id']:command(r['gold']) for r in targets}
    reference = {'fsc:'+r['example_id']:r for r in predictions}
    if (not sources or len(sources) != len(roster) or len(gold) != len(targets)
            or len(reference) != len(predictions) or sources.keys() != gold.keys()
            or sources.keys() != reference.keys()):
        raise ValueError('complete distinct paired population required')
    groups=defaultdict(list); truth={}; support=Counter(); predicted=Counter(); correct=Counter()
    invalid=0
    for uid,source in sources.items():
        speaker=source['speaker_id']; row=reference[uid]; y=gold[uid]
        if (type(speaker) is not str or not speaker or y not in commands.values()
                or row['audio_sha256'] != source['audio_sha256']):
            raise ValueError('paired gold, speaker or audio differs')
        parsed=parse_prediction(row['raw_prediction'])
        try:p=command(parsed) if parsed is not None else None
        except ValueError:p=None
        valid=p in commands.values(); invalid+=not valid
        groups[speaker].append(uid);support[y]+=1;truth[uid]=p==y
        if valid:predicted[p]+=1
        if truth[uid]:correct[y]+=1
    speakers=sorted(groups);n=len(sources)
    if len(speakers)<2:raise ValueError('speaker clusters required')
    def f1(k):return 2*correct[k]/(support[k]+predicted[k]) if support[k]+predicted[k] else 0.
    baseline=dict(recordings=n,correct=sum(truth.values()),accuracy=sum(truth.values())/n,
        invalid_or_out_of_ontology=invalid,
        macro_f1_full_ontology=sum(f1(k) for k in sorted(commands.values()))/31)
    comparisons={};deltas={}
    for role,policy in policies.items():
        rows={r['id']:r for r in policy['records']}
        if len(rows)!=len(policy['records']) or rows.keys()!=sources.keys():
            raise ValueError('complete unique Jev population required')
        for uid,row in rows.items():
            if (row['gold'] not in commands or row['top1'] not in commands
                    or commands[row['gold']]!=gold[uid]
                    or row['speaker_id']!=sources[uid]['speaker_id']
                    or type(row['top1_correct']) is not bool
                    or row['top1_correct']!=(row['top1']==row['gold'])):
                raise ValueError('Jev gold, speaker or native correctness differs')
        accuracy=sum(r['top1_correct'] for r in rows.values())/n
        if (policy['metrics']['queries']!=n or policy['metrics']['top1_event_accuracy']!=accuracy
                or policy['metrics']['candidate_ids']!=list(commands)):
            raise ValueError('Jev aggregate or candidate order differs')
        deltas[role]={i:int(rows[i]['top1_correct'])-int(truth[i]) for i in sources}
        comparisons[role]=dict(policy_id=policy['policy_id'],jev_accuracy=accuracy,
            reference_accuracy=baseline['accuracy'],accuracy_difference_jev_minus_reference=accuracy-baseline['accuracy'],
            both_correct=sum(rows[i]['top1_correct'] and truth[i] for i in sources),
            both_wrong=sum(not rows[i]['top1_correct'] and not truth[i] for i in sources),
            jev_only_correct=sum(rows[i]['top1_correct'] and not truth[i] for i in sources),
            reference_only_correct=sum(not rows[i]['top1_correct'] and truth[i] for i in sources))
    rng=random.Random(seed);samples={k:[] for k in deltas};reference_samples=[]
    totals={k:{s:sum(v[i] for i in ids) for s,ids in groups.items()} for k,v in deltas.items()}
    base_totals={s:sum(truth[i] for i in ids) for s,ids in groups.items()}
    for _ in range(replicates):
        draw=[rng.choice(speakers) for _ in speakers];den=sum(len(groups[s]) for s in draw)
        reference_samples.append(sum(base_totals[s] for s in draw)/den)
        for role in samples:samples[role].append(sum(totals[role][s] for s in draw)/den)
    baseline['accuracy_percentile95']=interval(reference_samples)
    for role in samples:comparisons[role]['difference_percentile95']=interval(samples[role])
    return dict(reference=baseline,native_comparisons=comparisons,
        bootstrap=dict(seed=seed,replicates=replicates,speaker_clusters=len(speakers)),
        scope='Same-audio descriptive native top1 comparison. FSC-supervised SpeechBrain versus '
              'SLURP-trained frozen configuration transfer; no matched-budget algorithm superiority, '
              'external probability metrics or target checkpoint selection. Speaker bootstrap does not '
              'cover training seeds, repeated commands across speakers or multiplicity.')


def run(spec, output):
    output=Path(output)
    if output.exists():raise ValueError('new comparison directory required')
    check_pins(spec['pins'])
    experiment=Path(spec['experiment']);status=read(experiment/'status.json')
    if (status['phase']!='frozen_transfer_collected_and_scored'
            or status['scores_sha256']!=file_hash(experiment/'scores.json')):
        raise ValueError('completed independently scored transfer required')
    archive,population,task,predictions,receipt=authenticate_archive(spec['reference'])
    scoring_spec=read(experiment/'score-spec.json');contract=scoring_spec['scoring_contract']
    if file_hash(contract['path'])!=contract['sha256']:raise ValueError('scoring contract changed')
    frozen=read(contract['path'])
    if (frozen['population']['sha256']!=file_hash(spec['reference']['population'])
            or frozen['task']['sha256']!=file_hash(spec['reference']['task'])):
        raise ValueError('reference and actual transfer definitions differ')
    output.mkdir(parents=True,exist_ok=False)
    # Re-run actual runtime/input/policy verification before opening target labels.
    full=reaudited_score(scoring_spec,output/'reaudited-scores.json')
    if full.get('runtime_provenance_verified') is not True:raise ValueError('runtime proof required')
    result=compare_pair(task,population['records'],scoring_targets(population,task),predictions,full['policies'])
    if (result['reference']['correct']!=receipt['result']['joint_correct']
            or result['reference']['recordings']!=receipt['result']['examples']):
        raise ValueError('recomputed historical native accuracy differs')
    result.update(reference_archive=archive,transfer_runtime_reaudited=True,
        reaudited_scores_sha256=file_hash(output/'reaudited-scores.json'),
        source_comparison=full['comparison'],comparison_pins=spec['pins'])
    with (output/'comparison.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    result=run(read(args.spec),args.output)
    print(json.dumps(result['native_comparisons'],indent=2))
