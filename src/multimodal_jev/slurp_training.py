"""Project individually reviewed clean SLURP decisions into audio training rows.

This module validates review provenance; it cannot perform semantic adjudication.
Pending decisions are omitted, never converted to negative or null supervision.
"""
import hashlib

from .consumption import canonical
from .task_schema import DecisionSchema, JevExample
from .thinking import ThinkingSpec


def content_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def decision_binding(example):
    return content_hash(dict(example_id=example.example_id, split=example.split,
                             source_group=example.metadata['source_group'],
                             audio_sha256=example.metadata['audio_sha256'],
                             input=example.model_input(), target=example.target))


def validate_admission(example):
    """Reject legacy or stale SLURP supervision at optimization entry points."""
    meta = example.metadata
    if example.split != 'train':
        raise ValueError('optimization admits TRAIN only')
    if meta.get('optimization_allowed', True) is not True:
        raise ValueError('optimization explicitly disallowed')
    if meta.get('role', 'learning') != 'learning':
        raise ValueError('optimization admits learning role only')
    marked = ('slurp' in example.example_id.lower() or
              'slurp' in example.schema.id.lower() or
              'slurp' in str(meta.get('dataset', '')).lower() or
              'slurp' in str(meta.get('source_dataset', '')).lower() or
              'slurp' in example.audio.lower() or 'slurp_review' in meta)
    if not marked:
        return
    review = meta.get('slurp_review')
    if not isinstance(review, dict) or review.get('protocol') != 'slurp-decision-admission-v1':
        raise ValueError('SLURP requires reviewed decision admission')
    if (meta.get('optimization_allowed') is not True or meta.get('role') != 'learning'
            or review.get('status') != 'reviewed' or not example.thinking):
        raise ValueError('SLURP reviewed learning decision required')
    for field in ('source_release_sha256', 'request_sha256', 'native_target_sha256',
                  'task_sha256', 'review_sha256', 'decision_sha256'):
        value = review.get(field)
        if (type(value) is not str or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('SLURP review hash required: ' + field)
    if review['decision_sha256'] != decision_binding(example):
        raise ValueError('SLURP reviewed decision changed')


def project(request, target, tasks, reviews, source_release_sha256):
    """Return admitted atomic examples and explicit masks for one recording.

    `tasks` is fixed independently of this source's labels. Review entries bind
    exact source target and task, with evidence for positive AND negative labels.
    Immutable clean source bundles remain pending; this creates a separate layer.
    """
    if (set(request) != {'id', 'official_split', 'role', 'source_group', 'audio_sha256', 'input'}
            or set(request['input']) != {'audio'}):
        raise ValueError('clean audio-only source request required')
    if request['official_split'] != 'train' or request['role'] != 'learning':
        raise ValueError('projection admits learning TRAIN only')
    request_hash = content_hash(request)
    if target['id'] != request['id'] or target['request_sha256'] != request_hash:
        raise ValueError('source target binding mismatch')
    if target['annotation_status'] != 'available':
        raise ValueError('source conflict cannot become supervision')
    if not tasks or set(reviews) - set(tasks):
        raise ValueError('fixed task bank and matching review keys required')
    target_hash = content_hash(target)
    examples, masks = [], {}
    for key, task in tasks.items():
        schema = DecisionSchema.from_dict(task['schema'])
        thinking = ThinkingSpec.from_dict(task['thinking'])
        if task['kind'] == 'intent':
            if schema.decision_type != 'choice':
                raise ValueError('intent requires Choice')
            labels = task['candidate_pairs']
            if (set(labels) != {c.id for c in schema.candidates}
                    or any(type(p) is not list or len(p) != 2 or
                           any(type(v) is not str or not v for v in p) for p in labels.values())
                    or len({tuple(p) for p in labels.values()}) != len(labels)):
                raise ValueError('unique candidate scenario/action mapping required')
            pair = [target['native_target']['scenario'], target['native_target']['action']]
            matches = [uid for uid, value in labels.items() if value == pair]
            if len(matches) != 1:
                raise ValueError('native intent absent from task candidates')
            value = matches[0]
        elif task['kind'] == 'type_presence':
            if schema.decision_type != 'noul' or not task.get('entity_type'):
                raise ValueError('type presence requires Noul and entity type')
            value = any(e['type'] == task['entity_type'] for e in target['native_target']['entities'])
        else:
            raise ValueError('unsupported reviewed projection')
        review = reviews.get(key)
        status = review.get('status') if review else 'pending'
        if status not in {'pending', 'conflict', 'unavailable', 'reviewed'}:
            raise ValueError('unknown review status')
        masks[key] = status
        if status != 'reviewed':
            continue
        if (review.get('request_sha256') != request_hash or
                review.get('native_target_sha256') != target_hash or
                review.get('task_sha256') != content_hash(task)):
            raise ValueError('stale or cross-source review')
        if (type(review.get('value')) is not type(value) or review['value'] != value):
            raise ValueError('review must agree with native label; corrections require separate release')
        if (not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip()
                or not isinstance(review.get('evidence'), list) or not review['evidence']
                or any(not isinstance(e, str) or not e.strip() for e in review['evidence'])):
            raise ValueError('reviewer and evidence references required, including negative decisions')
        meta = dict(dataset='slurp-clean-reviewed', role='learning',
                    source_group=request['source_group'], audio_sha256=request['audio_sha256'],
                    optimization_allowed=True,
                    slurp_review=dict(protocol='slurp-decision-admission-v1', status='reviewed',
                        source_release_sha256=source_release_sha256, request_sha256=request_hash,
                        native_target_sha256=target_hash, task_sha256=content_hash(task),
                        review_sha256=content_hash(review)))
        example = JevExample(request['id']+'::'+key, 'train', request['input']['audio'],
                             schema, value, meta, thinking)
        meta['slurp_review']['decision_sha256'] = decision_binding(example)
        validate_admission(example)
        examples.append(example)
    return examples, masks
