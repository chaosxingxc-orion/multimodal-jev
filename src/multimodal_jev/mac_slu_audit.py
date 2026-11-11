"""TRAIN-only structural evidence for MAC-SLU decision projection.

Empty annotation containers are recorded as unresolved, never negative labels.
No audio transcription or model input is constructed from reference text.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import PurePosixPath


def audit_rows(rows):
    ids = set()
    queries = defaultdict(list)
    intents = Counter()
    relations = Counter()
    flags = defaultdict(list)
    instance_counts = Counter()
    explicit_counts = Counter()
    summaries = []
    for row in rows:
        uid = row.get('id')
        if not isinstance(uid, str) or not uid or uid in ids:
            raise ValueError('unique nonempty string record ids required')
        ids.add(uid)
        if not isinstance(row.get('query'), str) or not isinstance(row.get('split_sens'), list):
            raise ValueError('reference query and split_sens structure differs')
        semantics = row.get('semantics')
        if not isinstance(semantics, dict):
            raise ValueError('semantics mapping required')
        # Audit-only grouping; reference strings never become actor input.
        qhash = hashlib.sha256(row['query'].encode()).hexdigest()
        queries[qhash].append(uid)
        present = []
        local_relations = []
        reasons = set()
        if not semantics:
            reasons.add('empty_semantics_unresolved')
        for label, domains in semantics.items():
            if not isinstance(domains, dict):
                raise ValueError('instance domain mapping required')
            if not domains:
                reasons.add('empty_intent_instance_unresolved')
            if len(domains) > 1:
                reasons.add('multiple_domains_in_instance')
            for domain, slots in domains.items():
                if not isinstance(domain, str) or not isinstance(slots, list):
                    raise ValueError('domain and slot list required')
                if any(not isinstance(s, dict) or not isinstance(s.get('name'), str)
                       or 'value' not in s for s in slots):
                    raise ValueError('named slot with source value required')
                labels = [s['value'] for s in slots if s['name'] == 'intent']
                if len(labels) != 1 or not isinstance(labels[0], str) or not labels[0]:
                    reasons.add('missing_or_multiple_intent_labels')
                    continue
                key = (domain, labels[0])
                present.append(key)
                intents[key] += 1
                slot_names = [s['name'] for s in slots if s['name'] != 'intent']
                if len(slot_names) != len(set(slot_names)):
                    reasons.add('repeated_slot_role')
                for name in set(slot_names):
                    relations[(*key, name)] += 1
                    local_relations.append((*key, name))
        if len(present) != len(set(present)):
            reasons.add('repeated_domain_intent_instances')
        if len(row['split_sens']) != len(semantics):
            reasons.add('split_sentence_instance_count_differs')
        for reason in sorted(reasons):
            flags[reason].append(uid)
        instance_counts[len(semantics)] += 1
        explicit_counts[len(present)] += 1
        summaries.append(dict(id=uid, annotation_instances=len(semantics),
                              explicit_intent_instances=len(present),
                              distinct_domain_intents=len(set(present)),
                              distinct_bound_slot_roles=len(set(local_relations)),
                              unresolved_empty=any('unresolved' in r for r in reasons),
                              flags=sorted(reasons)))
    return dict(records=len(rows), annotation_instance_histogram=dict(instance_counts),
                explicit_intent_histogram=dict(explicit_counts),
                domain_intent_types=len(intents), bound_slot_role_types=len(relations),
                intent_support=[dict(domain=k[0], intent=k[1], instances=v)
                                for k, v in sorted(intents.items())],
                bound_role_support=[dict(domain=k[0], intent=k[1], role=k[2], instances=v)
                                    for k, v in sorted(relations.items())],
                flag_record_counts={k: len(v) for k, v in sorted(flags.items())},
                flagged_ids=dict(sorted(flags.items())),
                exact_query_duplicate_groups=[dict(query_sha256=k, ids=v)
                                               for k, v in sorted(queries.items()) if len(v) > 1],
                records_without_empty_containers=sum(not r['unresolved_empty'] for r in summaries),
                record_summaries=summaries,
                scope='Structural TRAIN annotation audit, not semantic/listening review or training admission. '
                      'Empty containers unresolved. Reference text used only for duplicate grouping, never model input.')


def audit_audio_members(members, expected_ids):
    """Validate the entire TRAIN tar index without extraction or gold filtering."""
    ids = set()
    sizes = []
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('unsafe archive member')
        if member.isdir():
            continue
        if not member.isfile() or len(path.parts) != 2 or path.parts[0] != 'audio_train':
            raise ValueError('unexpected nonregular or misplaced audio member')
        if path.suffix != '.wav' or not path.stem.startswith('id_') or member.size <= 44:
            raise ValueError('unexpected audio filename or empty payload')
        uid = path.stem[3:]
        if uid in ids:
            raise ValueError('duplicate archive audio id')
        ids.add(uid)
        sizes.append(member.size)
    return dict(audio_members=len(ids), total_uncompressed_bytes=sum(sizes),
                missing_audio_ids=sorted(set(expected_ids) - ids),
                unlabelled_audio_ids=sorted(ids - set(expected_ids)),
                exact_id_coverage=ids == set(expected_ids),
                scope='Full gzip/tar stream and member metadata read; no waveform decode or listening review.')
