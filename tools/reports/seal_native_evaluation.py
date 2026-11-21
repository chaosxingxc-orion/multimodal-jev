"""Seal the three native test populations without dispatch or model selection."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_decision_contract import PREFIX_LAYOUT, render_prefix_context_request
from multimodal_jev.context_evaluation_contract import METRIC_FIELDS, DERIVED_FIELDS
from multimodal_jev.context_execution import map_io
from multimodal_jev.local_checkpoint import commit_json, sync_directory
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.source_snapshot import freeze, inventory, regular, verify
from multimodal_jev.slurp_training import content_hash

SPLITS = {'SLURP_clean_TEST60': 'official_test_clean', 'FSC_native31': 'official_test',
          'MINDS_native14': 'frozen_transfer_upstream_train'}


def publish(config, provenance):
    from tools.reports.context_transfer_macro_inputs import load as load_original
    from tools.reports.context_transfer_portable_inputs import load as load_portable
    from tools.reports.score_context_transfer_macro import fsc_truth, minds_truth
    from tools.reports.context_slurp_test_macro_truth import load as slurp_truth
    loaders = dict(FSC_native31=fsc_truth, MINDS_native14=minds_truth, SLURP_clean_TEST60=slurp_truth)
    output = Path(config['output'])
    workers = config['io_workers']
    if (set(config['datasets']) != set(SPLITS) or config['prompt_layout'] != PREFIX_LAYOUT
            or config['threshold'] != .5 or output.exists()):
        raise ValueError('fresh output and the three registered stable-prefix populations required')
    check_paths({'evaluation_seal': output})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.native-eval-') as work:
        work = Path(work)
        dest = work/'release'
        dest.mkdir()
        source = config['source']
        verify(source['root'], manifest_sha256=source['manifest_sha256'], io_workers=workers)
        staging = work/'engineering'
        shutil.copytree(source['root'], staging, ignore=shutil.ignore_patterns('FILES.json'),
                        copy_function=shutil.copyfile)
        for directory in (staging, *[p for p in staging.rglob('*') if p.is_dir()]):
            directory.chmod(0o755)
        for name in config['engineering_files']:
            target = staging/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(regular(Path(__file__).resolve().parents[2], name), target)
        engineering = freeze(staging, dest/'engineering', patterns=['**/*'], io_workers=workers)
        reports = {}
        audio_jobs = {}
        shared_root = Path(config['shared_root']).resolve()
        published_root = Path(config['published_root'])
        for dataset, spec in config['datasets'].items():
            print('Checking', dataset, flush=True)
            original = spec['original_evaluation']
            admitted = load_original(original)
            portable = load_portable(spec['portable_full_inputs'])
            if any(admitted[k] != portable[k] for k in ('dataset','population','candidate_ids','input_receipts')):
                raise ValueError('portable population differs from native evidence: ' + dataset)
            population, truth = loaders[dataset](original, spec['scoring'], admitted)
            ids = [r['id'] for r in population]
            if ids != list(admitted['requests']) or set(ids) != set(truth):
                raise ValueError('inference and scoring populations differ: ' + dataset)
            panel = None
            rows, labels = [], []
            for row in population:
                uid = row['id']; request = admitted['requests'][uid]; digest = row['audio_sha256']
                system, history, readouts = render_prefix_context_request(request)
                shape = dict(system=system, readouts=readouts,
                             questions=request.model_input()['questions'],
                             requirements=request.requirements, version=request.version)
                if panel is None:
                    panel = shape
                if shape != panel or history or request.history:
                    raise ValueError('expected one frozen audio-only panel per dataset')
                source_audio = Path(request.audio)
                if not source_audio.is_file():
                    raise ValueError('missing evaluation audio: ' + str(source_audio))
                if dataset == 'MINDS_native14':
                    rel = Path('audio')/dataset/(digest+'.wav')
                    physical = published_root/rel
                    audio_jobs[(str(source_audio),digest)] = (source_audio,digest,dest/rel)
                else:
                    if not source_audio.resolve().is_relative_to(shared_root):
                        raise ValueError('expected shared source audio')
                    physical = source_audio
                    audio_jobs[(str(source_audio),digest)] = (source_audio,digest,None)
                rows.append(dict(id=uid, audio=str(physical), audio_sha256=digest))
                target = truth[uid]
                group = (target['language'] if dataset == 'MINDS_native14'
                         else row['speaker_id'] if dataset == 'FSC_native31' else row['source_group'])
                gold = target['intent'] if dataset == 'MINDS_native14' else target
                if gold not in admitted['candidate_ids']:
                    raise ValueError('gold outside the frozen ontology')
                labels.append(dict(id=uid,audio_sha256=digest,gold=gold,group=group))
            (dest/'inference'/dataset).mkdir(parents=True)
            (dest/'scoring'/dataset).mkdir(parents=True)
            commit_json(dest/'inference'/dataset/'panel.json',dict(panel, candidate_ids=admitted['candidate_ids'],
                         prompt_layout=PREFIX_LAYOUT, history=[]))
            commit_json(dest/'inference'/dataset/'rows.json',rows)
            commit_json(dest/'scoring'/dataset/'targets.json',labels)
            reports[dataset] = dict(recordings=len(rows), candidates=len(admitted['candidate_ids']),
                                    split=SPLITS[dataset], input_origin=spec['portable_full_inputs']['manifest_sha256'],
                                    scoring_origin={k:v['sha256'] for k,v in spec['scoring'].items()})
            print(dataset, reports[dataset], flush=True)

        def check_audio(job):
            source, expected, target = job
            if target is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source,target)
            actual = target or source
            if file_hash(actual) != expected:
                raise ValueError('audio changed: ' + str(source))

        # Multiple recordings can legitimately carry identical audio. Read each
        # original; copy each content address once, while preserving every ID.
        copies = {}
        jobs = []
        for source_audio, digest, target in audio_jobs.values():
            if target is not None:
                copies.setdefault(target, (source_audio, digest, target))
            jobs.append((source_audio, digest, None))
        _, timing = map_io(check_audio, [*jobs, *copies.values()], workers=workers)
        paths = [str(path.relative_to(dest)) for path in dest.rglob('*')
                 if path.is_file() and not path.is_relative_to(dest/'engineering')]
        manifest = dict(schema='jev-native-evaluation-seal-v1', datasets=reports, prompt_layout=PREFIX_LAYOUT,
                        threshold=config['threshold'], metrics=[*METRIC_FIELDS,*DERIVED_FIELDS],
                        source_only_policy_selection=True, target_tuning_allowed=False,
                        engineering_manifest_sha256=engineering['manifest_sha256'],
                        files=inventory(dest,paths,io_workers=workers),
                        audio_integrity=dict(verified_unique_path_digest_pairs=len(audio_jobs),
                                             wall_seconds=timing['wall_seconds']),
                        stable_prefix_encoded_cache_ready=False, npu_admitted=False, provenance=provenance)
        commit_json(dest/'MANIFEST.json',manifest)
        os.rename(dest,output)
        sync_directory(output.parent)
    return dict(output=str(output), manifest_sha256=file_hash(output/'MANIFEST.json'), datasets=reports)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--local',required=True)
    args=parser.parse_args()
    c,p=load_config(args.config,args.local)
    print(json.dumps(publish(c,p),indent=2))
