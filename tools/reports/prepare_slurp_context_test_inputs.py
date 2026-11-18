"""Prepare compact label-free clean TEST inputs using the frozen training task.

This consumes existing accepted Omni population identities only. It does not
score, select policies, alter training, run a processor or dispatch inference.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_decision_contract import render_context_request
from multimodal_jev.slurp_training import content_hash
from tools.jev.run_context_baseline_sequence import check_pins


def read(path):return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:return [json.loads(line) for line in stream]


def select_inputs(native, population, learning, selection):
    """Only identity/audio fields are consumed; no test annotation lookup."""
    source={r['id']:r for r in native};keep=population['ids'];excluded=population['excluded_ids']
    if (len(source)!=len(native) or len(set(keep))!=len(keep) or len(set(excluded))!=len(excluded)
            or set(keep)&set(excluded) or set(source)!=set(keep)|set(excluded)):
        raise ValueError('exact exclusion-only unique test population required')
    for row in native:
        if (set(row)!={'id','input','audio_sha256','source_group','split'} or row['split']!='test'
                or set(row['input'])!={'audio','schema_sha256'}
                or row['id']!='slurp:'+Path(row['input']['audio']).name):
            raise ValueError('canonical label-free native test requests required')
    selected=[source[uid] for uid in keep]
    overlaps={}
    for name,records in [('learning',learning),('selection',selection)]:
        ids={r['id'] for r in records};hashes={r['audio_sha256'] for r in records}
        if len(ids)!=len(records):raise ValueError('unique learning/selection input identities required')
        overlaps[name]=dict(recording_ids=len(ids & set(keep)),
            audio_file_hashes=len(hashes & {r['audio_sha256'] for r in selected}))
        if any(overlaps[name].values()):raise ValueError('TEST overlaps source learning or selection')
    return selected,overlaps


def run(spec, output):
    output=Path(output)
    if output.exists():raise ValueError('new immutable input directory required')
    check_pins(spec['pins'])
    for key in ('omni_receipt','training_manifest','experiment'):
        if spec['pins'].get(spec[key])!=file_hash(spec[key]):raise ValueError('pinned source required')
    receipt=read(spec['omni_receipt']);score_path=Path(receipt['result']);score=read(score_path)
    if (file_hash(score_path)!=receipt['result_sha256']
            or not receipt['normalized_predictions_equal_filtered_original']
            or not receipt['clean_native_gold_independently_reconstructed']
            or not receipt['all_official_metric_entries_recomputed']):
        raise ValueError('previously accepted clean reference required')
    check_pins(score['pins'])
    population_path=score_path.parent/'population.json'
    predictions=score_path.parent/'official-predictions.jsonl'
    if (file_hash(population_path)!=score['population_sha256']
            or file_hash(predictions)!=score['normalized_predictions_sha256']
            or score['examples']!=12991 or score['original_examples']!=13078
            or score['split']!='test' or score['full_official_split'] is not False
            or score['full_declared_clean_population'] is not True
            or score['optimization_allowed'] is not False):
        raise ValueError('fixed complete clean TEST identity required')
    paths=[Path(p) for p in score['pins'] if Path(p).name=='test.requests.jsonl']
    if len(paths)!=1:raise ValueError('one accepted native input source required')
    release=Path(spec['training_manifest']).parent;manifest=read(spec['training_manifest'])
    check_pins({str(release/p):h for p,h in manifest['files'].items()})
    experiment=read(spec['experiment'])
    if (experiment['release_sha256']!=file_hash(spec['training_manifest'])
            or experiment['task_semantics']!='exclusive_event'
            or experiment['learning_recordings']!=40513 or experiment['selection_recordings']!=9226):
        raise ValueError('existing frozen exclusive-event experiment required')
    learning=rows(release/'learning.audio.jsonl');selection=rows(release/'selection.audio.jsonl')
    if len(learning)!=40513 or len(selection)!=9226:raise ValueError('complete source population required')
    native=rows(paths[0]);population=read(population_path)
    selected,overlaps=select_inputs(native,population,learning,selection)
    if len(selected)!=12991 or len(native)!=13078:raise ValueError('full accepted population required')
    task=read(release/'tasks.json')['intent']
    candidates=[dict(id=c['id'],definition=c['description']) for c in task['schema']['candidates']]
    if len(candidates)!=60 or len({c['id'] for c in candidates})!=60:
        raise ValueError('complete frozen training candidate menu required')
    output.mkdir(parents=True,exist_ok=False)
    # Definitions are stored once; requests are reconstructed with the pinned
    # compiler and checked against per-record request/rendering hashes.
    definitions=dict(candidates=candidates,requirements=task['thinking']['requirements'],semantics='exclusive_event')
    (output/'definitions.json').write_text(json.dumps(definitions,indent=2)+'\n')
    with (output/'inputs.jsonl').open('x') as stream:
        for row in selected:
            audio=row['input']['audio']
            if file_hash(audio)!=row['audio_sha256']:raise ValueError('test waveform file changed')
            request=compile_candidates(audio,**definitions)
            system,text=render_context_request(request)
            value=dict(id=row['id'],audio=audio,audio_sha256=row['audio_sha256'],
                source_group=row['source_group'],request_sha256=content_hash(request.model_input()),
                rendered_sha256=content_hash(dict(system=system,text=text)))
            stream.write(json.dumps(value,separators=(',',':'))+'\n')
    files={**spec['pins'],**score['pins']}
    for path in (score_path,population_path,predictions,release/'tasks.json',
                 release/'learning.audio.jsonl',release/'selection.audio.jsonl',Path(__file__)):
        files[str(path.resolve())]=file_hash(path)
    report=dict(schema='slurp-clean-context-test-inputs-v1',status='input_only_not_runtime_admitted',
        recordings=len(selected),original_recordings=len(native),excluded_recordings=len(population['excluded_ids']),
        source_groups=len({r['source_group'] for r in selected}),candidate_ids=[c['id'] for c in candidates],
        semantics='exclusive_event',request_version='jev-context-exclusive_event-v1',
        split='official_test_clean_exclusion_subset',source_population_overlap=overlaps,
        overlap_scope='Recording IDs and compressed audio-file hashes; not decoded waveform or semantic duplicate certification.',
        task_source_sha256=file_hash(release/'tasks.json'),training_release_sha256=file_hash(spec['training_manifest']),
        source_files=files,files={p.name:file_hash(p) for p in output.iterdir() if p.is_file()},
        policy_selection='source_SLURP_selection_only_before_target_inference',
        target_labels_opened=False,optimization_allowed=False,dispatch_allowed=False,
        existing_omni_reference=dict(examples=receipt['examples'],joint_accuracy=receipt['joint_accuracy'],
            joint_correct=receipt['joint_correct'],malformed=receipt['malformed'],
            provenance='Previously verified metric; not rescored here.',
            interpretation='Native composite generation with malformed outputs counted wrong; not matched Jev interface.'),
        missing_before_evaluation=['source_selected_policy_identity','full_processor_and_input_cache_validation',
            'same_policy_HF_vLLM_parity_on_predeclared_label_free_panel','full_population_inference_and_independent_scoring'],
        scope='Full clean TEST input preparation with unchanged TRAIN-defined candidate menu. No target labels read, '
              'processor/model forward, TEST model score, full-original-TEST claim or RL improvement. '
              'Does not change or block existing selection/GRPO/RLOO/FSC queues.')
    with (output/'MANIFEST.json').open('x') as stream:json.dump(report,stream,indent=2)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();r=run(read(args.spec),args.output)
    print(json.dumps({k:r[k] for k in ('status','recordings','source_groups','source_population_overlap')},indent=2))
