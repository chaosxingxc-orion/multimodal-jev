#!/usr/bin/env python3
"""Exact-text provenance candidates for learning sources, never new supervision."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_sha(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def annotation_text(value):
    # Only strip bracketed annotation syntax, not punctuation/case/wake words.
    return re.sub(r'\[([^:\[\]]+) : ([^\[\]]+)\]', r'\2', value)


def trace(sources, annotated, original):
    index = defaultdict(dict)
    original_by_id = defaultdict(list)
    for ordinal, row in enumerate(original, 1):
        original_by_id[row['answerid']].append((ordinal, row))
    for ordinal, row in enumerate(annotated, 1):
        forms = {'answer_normalised': row['answer_normalised'],
                 'answer_annotation_stripped': annotation_text(row['answer_annotation'])}
        for field, text in forms.items():
            if text:
                index[text].setdefault(ordinal, {'row': row, 'matched_fields': []})['matched_fields'].append(field)
    ids = [str(s['slurp_id']) for s in sources]
    if len(ids) != len(set(ids)):
        raise ValueError('unique learning source IDs required')
    result = []
    for source in sources:
        matches = []
        for ordinal, item in sorted(index.get(source['sentence'], {}).items()):
            row = item['row']
            # IDs join only within the upstream dataset, never SLURP->upstream.
            originals = []
            for original_ordinal, origin in original_by_id.get(row['answerid'], []):
                equal_fields = [k for k in ('answer', 'answer_normalised')
                                if origin[k] and origin[k] == row[k]]
                originals.append(dict(csv_record_ordinal=original_ordinal,
                    row_sha256=content_sha(origin), original_row=origin,
                    content_equal_fields=equal_fields,
                    linkage_status='id_and_content_match' if equal_fields else 'id_only_unverified'))
            matches.append(dict(csv_record_ordinal=ordinal, row_sha256=content_sha(row),
                matched_fields=item['matched_fields'], annotated_row=row,
                native_pair_agreement=[source['scenario'], source['action']] == [row['scenario'], row['intent']],
                original_candidates=originals))
        result.append(dict(source_id=str(source['slurp_id']), reference=source['sentence'],
            source_metadata_sha256=content_sha(source), native_pair=[source['scenario'], source['action']],
            status='no_exact_match' if not matches else 'one_exact_candidate' if len(matches)==1 else 'multiple_exact_candidates',
            candidates=matches, actor_input_allowed=False, optimization_allowed=False,
            semantic_supervision_admitted=False))
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ('learning', 'bindings', 'metadata', 'fetch', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    learning = [json.loads(x) for x in a.learning.read_text().splitlines()]
    bindings = [json.loads(x) for x in a.bindings.read_text().splitlines()]
    by_record = {x['id']:x for x in bindings}
    if len(by_record)!=len(bindings) or len({x['id'] for x in learning})!=len(learning):
        raise ValueError('duplicate recording IDs')
    source_ids = set()
    for row in learning:
        if row['role']!='learning' or row['official_split']!='train':
            raise ValueError('learning TRAIN only')
        bound = by_record[row['id']]
        if bound['request_sha256']!=row['source_request_sha256']:
            raise ValueError('request identity mismatch')
        source_ids.update(str(x) for x in bound['source_ids'])
    metadata = [json.loads(x) for x in a.metadata.read_text().splitlines()]
    sources = [x for x in metadata if str(x['slurp_id']) in source_ids]
    if {str(x['slurp_id']) for x in sources}!=source_ids:
        raise ValueError('missing learning source')
    fetch = json.loads(a.fetch.read_text())
    upstream = {}
    pins = {str(x):sha(x) for x in (a.learning,a.bindings,a.metadata,a.fetch)}
    for item in fetch['files']:
        path = Path(item['path'])
        data = path.read_bytes()
        if sha(path)!=item['sha256'] or hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=item['git_blob']:
            raise ValueError('upstream identity mismatch')
        pins[str(path)]=item['sha256']
        with path.open() as f:
            upstream[path.name]=list(csv.DictReader(f, delimiter=';'))
    annotated=upstream['NLU-Data-Home-Domain-Annotated-All.csv']
    original=upstream['paraphrases_and_intents_26k_normalised_all.csv']
    traced=trace(sources,annotated,original)
    questions=defaultdict(Counter)
    for row in traced:
        for match in row['candidates']:
            native=match['annotated_row']
            questions['/'.join(row['native_pair'])][native['question']]+=1
    summary=dict(schema='slurp-learning-upstream-provenance-v1', pins=pins,
        implementation_sha256=sha(__file__), upstream_commit=fetch['commit'],
        learning_recordings=len(learning), learning_source_utterances=len(traced),
        source_status_counts=dict(Counter(x['status'] for x in traced)),
        upstream_annotated_records=len(annotated), upstream_original_records=len(original),
        annotated_duplicate_answer_ids=sum(n>1 for n in Counter(x['answerid'] for x in annotated).values()),
        exact_candidate_bindings=sum(len(x['candidates']) for x in traced),
        native_pair_disagreement_bindings=sum(not m['native_pair_agreement'] for x in traced for m in x['candidates']),
        question_counts_by_native_pair={k:dict(v) for k,v in sorted(questions.items())},
        question_count_unit='matched upstream record bindings, not independent sources or recordings',
        matching='literal full reference against answer_normalised or annotation markup stripped; no ID join, no case/punctuation/wake-word normalization',
        restrictions=['Exact text matches are provenance candidates, not certified historical row identity.',
                      'All duplicate text candidates retained; no label-based tie breaking.',
                      'Original linkage within upstream requires content verification; ID-only candidates remain unverified.',
                      'Collection prompts and upstream labels are offline provenance only, never actor inputs or semantic supervision.',
                      'No selection/test examples inspected or inferred as semantic negatives.'],
        training_dispatch_allowed=False)
    a.output.mkdir(parents=True)
    out=a.output/'source-provenance.jsonl'
    out.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in traced))
    summary['files']={out.name:sha(out)}
    (a.output/'report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('pins','question_counts_by_native_pair','restrictions')},indent=2))


if __name__=='__main__':
    main()
