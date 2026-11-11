"""Validate authored type definitions independently of supervision approval."""
from .task_schema import DecisionSchema
from .thinking import ThinkingSpec


def validate_catalog(catalog, expected_types):
    rows=catalog['types']
    names=[r['entity_type'] for r in rows]
    if len(names)!=len(set(names)) or set(names)!=set(expected_types):
        raise ValueError('exact unique type inventory required')
    for row in rows:
        if row['definition_status']!='authored_draft' or row['optimization_allowed'] is not False:
            raise ValueError('draft catalog cannot authorize supervision')
        if type(row['definition']) is not str or not row['definition'].strip():
            raise ValueError('semantic definition required')
        neighbors=row['confusable_types']
        if not neighbors or len(neighbors)!=len(set(neighbors)) or set(neighbors)-set(names) or row['entity_type'] in neighbors:
            raise ValueError('known distinct neighboring types required')
        if not row['boundary_review'].strip() or not row['evidence_source_ids']:
            raise ValueError('evidence and pending boundary review required')
        ThinkingSpec('type-roles-v1',tuple(row['thinking_requirements']))
    if catalog['all_definitions_approved'] is not False:
        raise ValueError('draft approval flag must remain false')
    return catalog


def review_contract(catalog, entity_type):
    """Render a candidate definition, for review only; no target construction.

    Include neighboring definitions so the target card can be inspected with
    its actual proposed boundaries. This does not assert mutually exclusive
    labels or resolve the pending annotation precedence.
    """
    indexed={r['entity_type']:r for r in catalog['types']}
    validate_catalog(catalog,indexed)
    row=indexed[entity_type]
    question=('Does the audio contain evidence for the following semantic role? '
              +row['definition']+' Related role definitions: '
              +' '.join(n+': '+indexed[n]['definition'] for n in row['confusable_types']))
    schema=DecisionSchema('type-role-review','noul',question,allow_unknown=True)
    thinking=ThinkingSpec('type-roles-v1',tuple(row['thinking_requirements']))
    return dict(schema=schema,thinking=thinking,optimization_allowed=False,
                boundary_review=row['boundary_review'])
