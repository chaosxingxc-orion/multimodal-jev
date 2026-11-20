import json
from pathlib import Path

from tools.reports.fsc_annotation_cues import mine
from tools.reports.fsc_keyword_evaluation import command_values, make_panels, matched_pairs, speaker_stats, summary


def test_mining_uses_distinct_official_annotations_and_speaker_support():
    values = {'up': ('increase', 'volume', 'none'), 'down': ('decrease', 'volume', 'none')}
    source = dict(source_split='train', source_text_column='transcription',
                  source_text_kind='publisher_supplied_written_recording_prompt',
                  excluded_words=['turn', 'the'], min_distinct_annotations=2, min_speakers=2,
                  min_precision=.75, min_occurrence_rate_gap=.1, keywords_per_field=3, max_example_words=8)
    rows = []
    for label, texts in {'up': ['Volume up', 'Turn the volume up', 'Louder please'],
                         'down': ['Volume down', 'Turn the volume down', 'Quieter please']}.items():
        action, obj, location = values[label]
        for text in texts:
            for speaker in ('speaker-a', 'speaker-b'):
                rows.append(dict(action=action, object=obj, location=location, speakerId=speaker, transcription=text))
    original = mine(rows, source, values)
    duplicated = mine(list(reversed(rows)) + [rows[-1]]*50, source, values)
    for label in values:
        arm = original['candidates'][label]
        assert arm['fields'] == duplicated['candidates'][label]['fields']
        assert [x['word'] for x in arm['fields']['action']['selected']] == [label]
        assert arm['fields']['location']['selected'] == []
        assert arm['example']['text'] in {r['transcription'] for r in rows if r['action'] == values[label][0]}


def test_panels_only_add_registered_cues_and_keep_readouts():
    root = Path(__file__).resolve().parents[1]
    task = json.loads((root/'configs/tasks/fsc-span-transfer-v1.json').read_text())
    protocol = json.loads((root/'configs/tasks/fsc/keyword-evaluation-v3.json').read_text())
    cues = {'candidates': {key: dict(fields={field: {'selected': [{'word': 'help'}]} for field in
            ('action', 'object', 'location')}, example={'text': 'Official example'})
            for key in command_values(task['commands'])}}
    panels = make_panels(protocol, task, cues)
    base = panels['baseline']
    for arm, p in panels.items():
        for fixed in ('candidate_ids', 'readouts', 'requirements', 'version', 'prompt_layout'):
            assert p[fixed] == base[fixed]
        for q, original in zip(p['questions'], base['questions'], strict=True):
            assert q['id'] == original['id'] and q['definition'].startswith(original['definition'])
            assert ('cue words' in q['definition']) == protocol['arms'][arm]['keywords']
            assert ('Official example' in q['definition']) == protocol['arms'][arm]['examples']


def test_single_field_diagnostic_uses_all_legal_neighbors_and_half_ties():
    values = {'up': ('increase', 'heat', 'none'), 'down': ('decrease', 'heat', 'none'),
              'volume': ('increase', 'volume', 'none'), 'kitchen': ('increase', 'heat', 'kitchen'),
              'other': ('decrease', 'volume', 'kitchen')}
    pairs = matched_pairs(values)
    assert pairs['up'] == [('action', 'down'), ('object', 'volume'), ('location', 'kitchen')]
    rows = [{'id': 'a', 'prob': [.2, .1, .4, .2, .6]}]
    targets = [{'id':'a', 'gold':'up', 'group':'speaker'}]
    result = summary(speaker_stats(rows, targets, list(values), pairs), ['speaker'])
    assert [result['matched_'+k] for k in ('action', 'object', 'location')] == [1., 0., .5]
    assert result['top1_accuracy'] == 0
