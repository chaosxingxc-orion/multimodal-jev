"""Mine word cues from the original written recording prompts in official FSC TRAIN."""
from collections import Counter, defaultdict
import re
import unicodedata

from multimodal_jev.fsc_span_transfer import FIELDS


def words(text):
    return tuple(re.findall(r"[a-z]+", unicodedata.normalize('NFKC', text).lower()))


def mine(train_rows, source, values):
    if (source['source_split'], source['source_text_column'], source['source_text_kind']) != (
            'train', 'transcription', 'publisher_supplied_written_recording_prompt'):
        raise ValueError('Only the publisher written TRAIN prompts are supported')
    by_triple = defaultdict(lambda: defaultdict(set))
    raw_text = defaultdict(Counter)
    labels_per_text = defaultdict(set)
    for row in train_rows:
        triple = tuple(row[field] for field in FIELDS)
        text = row[source['source_text_column']].strip()
        phrase = words(text)
        if not phrase:
            raise ValueError('Missing official TRAIN annotation')
        by_triple[triple][phrase].add(row['speakerId'])
        raw_text[triple, phrase][text] += 1
        labels_per_text[phrase].add(triple)
    if set(by_triple) != set(values.values()):
        raise ValueError('Official TRAIN labels differ from the fixed FSC ontology')

    def pooled(triples):
        pooled_phrases = defaultdict(set)
        for triple in triples:
            for phrase, speakers in by_triple[triple].items():
                pooled_phrases[phrase].update(speakers)
        return pooled_phrases

    def term_stats(phrases):
        counts = Counter()
        speakers = defaultdict(set)
        for phrase, ids in phrases.items():
            for token in set(phrase):
                counts[token] += 1
                speakers[token].update(ids)
        return counts, speakers

    excluded = set(source['excluded_words'])
    candidates = {}
    for label, triple in values.items():
        fields = {}
        cue_terms = set()
        for index, (field, value) in enumerate(zip(FIELDS, triple, strict=True)):
            if value == 'none':
                fields[field] = dict(contrast='unspecified_no_cue', selected=[])
                continue
            near = [t for t in by_triple if t[index] != value and
                    all(a == b for j, (a, b) in enumerate(zip(t, triple, strict=True)) if j != index)]
            # When there is no single-field legal competitor, compare classes of
            # the same action/object context where possible, then the same field globally.
            if near:
                pos, neg = by_triple[triple], pooled(near)
                scope = 'one_field_neighbors'
            else:
                context = (0,) if field == 'object' else (1,) if field == 'action' else ()
                pool = [t for t in by_triple if context and all(t[j] == triple[j] for j in context)]
                if any(t[index] != value for t in pool):
                    # Restrict fallback groups to locations represented on both
                    # sides. For example, "kitchen" cannot serve as evidence for
                    # "lights" just because lamp recordings have no room tag.
                    pos_labels = [t for t in pool if t[index] == value]
                    neg_labels = [t for t in pool if t[index] != value]
                    locations = {t[2] for t in pos_labels} & {t[2] for t in neg_labels}
                    if locations:
                        pos_labels = [t for t in pos_labels if t[2] in locations]
                        neg_labels = [t for t in neg_labels if t[2] in locations]
                    pos, neg = pooled(pos_labels), pooled(neg_labels)
                    scope = 'shared_action_or_object'
                else:
                    pos = pooled(t for t in by_triple if t[index] == value)
                    neg = pooled(t for t in by_triple if t[index] != value)
                    scope = 'all_train_field_values'
            if not pos or not neg:
                fields[field] = dict(contrast='no_contrast', selected=[])
                continue
            positives, speakers = term_stats(pos)
            negatives, _ = term_stats(neg)
            ranked = []
            for token, n in positives.items():
                m = negatives[token]
                precision = n / (n+m)
                gap = n/len(pos) - m/len(neg)
                if (token not in excluded and len(token) >= 2
                        and n >= source['min_distinct_annotations']
                        and len(speakers[token]) >= source['min_speakers']
                        and precision >= source['min_precision']
                        and gap >= source['min_occurrence_rate_gap']):
                    ranked.append(dict(word=token, positive_annotations=n, contrast_annotations=m,
                                       positive_speakers=len(speakers[token]), precision=precision, rate_gap=gap))
            ranked.sort(key=lambda row: (-row['rate_gap'], -row['positive_annotations'],
                                         -row['positive_speakers'], row['word']))
            fields[field] = dict(contrast=scope, positive_pool=len(pos), contrast_pool=len(neg), selected=ranked)
        # Attribute a repeated word to the field where it has the strongest
        # contrast. In a tie prefer location, then action, then object.
        priority = {'object': 0, 'action': 1, 'location': 2}
        owner = {}
        for field, report in fields.items():
            for entry in report['selected']:
                key = (entry['rate_gap'], priority[field], field)
                word = entry['word']
                if word not in owner or key > owner[word]:
                    owner[word] = key
        for field, report in fields.items():
            report['selected'] = [entry for entry in report['selected'] if owner[entry['word']][2] == field][
                :source['keywords_per_field']]
            cue_terms.update(entry['word'] for entry in report['selected'])
        # Select an existing unambiguous written prompt with broad cue coverage;
        # support is only a tie-breaker, so repeats do not dominate word ranking.
        eligible = [p for p in by_triple[triple]
                    if len(labels_per_text[p]) == 1 and len(p) <= source['max_example_words']]
        if not eligible:
            raise ValueError(f'No short unambiguous official TRAIN example for {label}')
        phrase = min(eligible, key=lambda p: (-len(set(p) & cue_terms), len(p),
                                              -len(by_triple[triple][p]), p))
        variants = raw_text[triple, phrase]
        example = min(variants, key=lambda text: (-variants[text], text))
        candidates[label] = dict(fields=fields, example=dict(text=example,
            speakers=len(by_triple[triple][phrase]), recordings=variants.total()))
    return dict(source_text_kind=source['source_text_kind'], split=source['source_split'],
                recordings=len(train_rows), speakers=len({row['speakerId'] for row in train_rows}),
                distinct_annotations=len(labels_per_text), candidates=candidates)
