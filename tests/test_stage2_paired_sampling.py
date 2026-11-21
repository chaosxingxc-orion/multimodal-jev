from copy import deepcopy
import math

import pytest

from multimodal_jev.stage2_training_view import sample
from tools.reports.prepare_stage2_paired_dataset import make_channel_pairs


def test_matched_failures_and_replay_are_independent_of_probability_working_point():
    names = ['left', 'right', 'neutral']
    # Four failed recordings have explicit same-wording controls. Only one
    # replay slot is available at 20%, regardless of the requested class floor.
    gold = {'wrong': 'left', 'tied': 'left', 'correct': 'left', 'same': 'left',
            'also_wrong': 'left', 'another_wrong': 'left',
            'flip_control': 'right', 'unverified': 'neutral', 'protected': 'neutral'}
    rows = [dict(id=uid, intent=intent, audio_sha256=uid,
                 role='learning', official_split='train', known_pending_source=False)
            for uid, intent in gold.items()]
    probability = dict(wrong=[.2,.4,.1], tied=[.2,.2,.1], correct=[.4,.3,.2],
        same=[.4,.3,.2], also_wrong=[.2,.1,.4], another_wrong=[.2,.1,.4],
        flip_control=[.2,.4,.1], unverified=[.5,.1,.4], protected=[.1,.2,.8])
    evidence = dict(eligible_ids=list(gold.keys()-{'protected'}),
        same_words=[[['correct','wrong','also_wrong'], ['same','tied','another_wrong']]],
        local_pairs=[['cue_flip','wrong','flip_control']])
    options = dict(mode='paired_train_v1', seed=7, replay_class_priority=16,
                   easy_fraction=.2, evidence=evidence)
    ids, evidence_rows, summary = sample(rows, names, probability, options)
    flags = {d['id']: d for d in evidence_rows}
    assert summary['selected_difficult'] == 4
    assert summary['selected_easy'] == 1
    assert flags['wrong']['comparison_ids']['cue_flip'] == 'flip_control'
    assert flags['tied']['comparison_ids']['same_words_different_speaker'] == 'correct'
    assert flags['unverified']['bucket'] == 'unverified_rank_candidate'
    assert {'unverified','protected','flip_control'}.isdisjoint(ids)
    shifted = {uid: [1/(1+math.exp(-(math.log(p/(1-p))+2))) for p in ps]
               for uid, ps in probability.items()}
    after, _, changed = sample(list(reversed(rows)), names, shifted, options)
    assert after == ids
    assert summary['probability_quality']['all'] != changed['probability_quality']['all']
    with pytest.raises(ValueError, match='clean TRAIN'):
        sample([dict(rows[0], official_split='test'), *rows[1:]], names, probability, options)
    invalid = deepcopy(options)
    invalid['evidence']['local_pairs'].append(['cue_flip','wrong','protected'])
    with pytest.raises(ValueError, match='permitted training targets'):
        sample(rows, names, probability, invalid)


def test_channel_pair_requires_matching_source_and_text_and_reuses_native_task():
    def row(uid, speaker='s', source='st', group='g', text='turn it on'):
        return dict(id=uid, audio_sha256=uid, speaker=speaker, source=source,
                    group=group, text=text, gold='on')
    m = [row('a-headset.flac'), row('a.flac'), row('b-headset.flac'), row('b.flac', source='other'),
         row('c-headset.flac')]
    pairs, conflicts = make_channel_pairs(m, dict(close_suffix='-headset.flac', distant_suffix='.flac'))
    assert pairs == [['a-headset.flac', 'a.flac']]
    assert conflicts == 1
    eligible = [x['id'] for x in m]
    rows = [dict(id=x['id'], audio_sha256=x['audio_sha256'], source_group=x['group'],
                 role='learning', official_split='train', known_pending_source=False, intent=x['gold']) for x in m]
    probs = dict.fromkeys(eligible, [.8, .2])
    probs['a-headset.flac'] = [.2, .8]
    evidence = dict(schema='jev-paired-train-evidence-v2', channel_pairs=pairs,
        channel_conflicts_excluded=conflicts, holdout_keys=['audio_sha256','source','group','text'],
        eligible_ids=eligible, same_words=[], local_pairs=[['cue_preserve','b.flac','c-headset.flac']])
    opts = dict(mode='paired_train_v1', seed=7, replay_class_priority=16, easy_fraction=.2, evidence=evidence)
    ids, diagnostics, summary = sample(rows, ['on','off'], probs, opts)
    assert summary['hard_evidence_by_kind'] == {'same_utterance_different_microphone': 1}
    assert next(x for x in diagnostics if x['id']=='a-headset.flac')['comparison_ids'] == {
        'same_utterance_different_microphone':'a.flac'}
    assert len(ids) == 2
    rows[0]['source_group'] = 'other'
    with pytest.raises(ValueError, match='same utterance group'):
        sample(rows, ['on','off'], probs, opts)


def test_full_train_keeps_uncorroborated_rank_errors_and_disjoint_probability_tails():
    candidates = ['room_on', 'room_off', 'other_event']
    probabilities = {
        'matched_error': [.1,.8,.2],
        'rank_error': [.2,.1,.7],
        'low_gold': [.35,.1,.1],
        'high_negatives': [.98,.95,.92],
        'matched_correct': [.97,.02,.01],
        'replay': [.97,.02,.01],
    }
    rows = [
        dict(id=uid, audio_sha256=uid, source_group=uid, intent='room_on', scenario='room',
             role='learning', official_split='train', known_pending_source=False)
        for uid in probabilities
    ]
    evidence = dict(schema='jev-paired-train-evidence-v3', eligible_ids=list(probabilities),
        same_words=[[['matched_error','matched_correct'],['replay']]], local_pairs=[],
        channel_pairs=[], holdout_keys=[], channel_conflicts_excluded=0)
    options = dict(mode='paired_train_v1', seed=7, replay_class_priority=16,
                   easy_fraction=.25, evidence=evidence, min_unique_recordings=6,
                   probability_examples_per_type=1)
    ids, diagnostics, summary = sample(rows, candidates, probabilities, options)
    assert set(ids) == {r['id'] for r in diagnostics if r['selected']}
    assert summary['selected_by_type'] == {
        'same_words_different_speaker':1, 'rank_cross_scenario':1,
        'positive_underconfidence':1, 'negative_overconfidence':1, 'easy_replay':2,
    }
    assert summary['selected_native_failures'] == 2
    assert summary['selected_rank_only'] == 1
    with pytest.raises(ValueError, match='minimum unique recordings'):
        sample(rows, candidates, probabilities, dict(options,min_unique_recordings=7))
