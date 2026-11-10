"""Scorer-only release target projection; unknown extra arguments stay unknown."""
import ast

from .audio2tool_score import typed,validate_calls


def literal_call(text):
    try:
        node=ast.parse(text,mode='eval').body
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.args:
            raise ValueError('not a named keyword call')
        args={}
        for kw in node.keywords:
            if kw.arg is None or kw.arg in args:raise ValueError('duplicate/expanded keyword')
            args[kw.arg]=ast.literal_eval(kw.value)
        call=dict(name=node.func.id,arguments=args);validate_calls([call]);return call
    except (SyntaxError,TypeError,ValueError,RecursionError):
        return None


def project(row,tier,official,registry_names):
    primary=dict(name=row['tool_name'],arguments=row['extracted_params'])
    validate_calls([primary])
    if primary['name'] not in registry_names:raise ValueError('unknown primary tool')
    extras=row.get('additional_tool_calls')
    if extras is None:extras=[]
    if type(extras) is not list or any(type(v) is not str or not v for v in extras):
        raise ValueError('null or explicit additional tool strings required')
    canonical={official.normalize_tool_name(n):n for n in registry_names}
    if len(canonical)!=len(registry_names):raise ValueError('ambiguous normalized registry')
    annotations=[];calls=[primary]
    for raw in extras:
        name=official.normalize_tool_name(raw)
        if name not in canonical:raise ValueError('additional tool does not normalize to registry')
        parsed=literal_call(raw)
        if parsed is not None and parsed['name']!=canonical[name]:
            parsed=None
        annotations.append(dict(raw=raw,name=canonical[name],arguments=None if parsed is None else parsed['arguments'],
                                argument_evidence='unavailable' if parsed is None else 'explicit_literal_call'))
        if parsed is not None:calls.append(parsed)
    original=literal_call(row['expected_tool_call'])
    complete=all(a['argument_evidence']=='explicit_literal_call' for a in annotations)
    return dict(id=f"audio2tool:tier{tier}:release:{row['id']}",tier=tier,release_id=row['id'],
        source_query_idx=row['query_idx'],primary=primary,
        official_expected_tools=[primary['name'],*extras],additional_annotations=annotations,
        full_calls=calls if complete else None,
        full_call_argument_coverage='complete' if complete else 'incomplete',
        primary_target_source='tool_name + extracted_params',
        expected_tool_call_raw=row['expected_tool_call'],
        primary_encoding_consistent=None if original is None else typed(original)==typed(primary),
        scope='Scorer-only target; primary structured fields authoritative for official-compatible scoring; no invented additional arguments.')
