"""Input-only decision bank and partial-gold Audio2Tool Jev projection.

Noul follows the existing explicit true/false contract. Typed call lists and
tool sequences are project extensions, not nga's 26-option letter readout.
"""
from copy import deepcopy

from .audio2tool_input import REGISTRY_SHA256,validate
from .audio2tool_inventory import registry
from .audio2tool_inference import digest
from .audio2tool_score import typed,validate_calls,validate_projected_target

PROTOCOL='audio2tool-jev-bundle-v1'


def task_bank(registry_bytes):
    tools=registry(registry_bytes,REGISTRY_SHA256)
    if len(tools)!=152:raise ValueError('complete independent tool registry required')
    decisions=[]
    for tool in tools:
        key='tool.'+tool['tool_id']
        for suffix,question in (
            ('present','At least one call to this tool is required by the final user request.'),
            ('absent','No call to this tool is required by the final user request.')):
            decisions.append(dict(id=key+'.'+suffix,decision_type='noul',tool_name=tool['tool_name'],
                question=question,true_definition=question,false_definition='The stated proposition is false.'))
        decisions.append(dict(id=key+'.calls',decision_type='typed_call_list',tool_name=tool['tool_name'],
            question='Return every call to this tool as a list of argument objects. Preserve repeated calls and their '
                'relative order, parameter types and nested values. Return [] when no call is required. '
                'Follow the full tool signature, defaults and constraints; do not invent missing values.',
            value_schema=dict(type='array',items=dict(type='object')),
            schema_scope='Structural JSON type only; parameter semantics supplied by the unchanged registry definition.'))
    decisions.append(dict(id='tools.sequence',decision_type='typed_tool_sequence',
        question='Return the tool names for the final request, with the primary call first, followed by additional '
            'calls in order. Include repeated tool names once for each call.',
        value_schema=dict(type='array',items=dict(type='string',enum=[t['tool_name'] for t in tools])),
        scoring_scope='Local sequence metric uses primary then published additional-call order; annotation order is not verified execution chronology.'))
    return dict(protocol=PROTOCOL,registry_sha256=REGISTRY_SHA256,tools=tools,decisions=decisions,
        annotation_missing_policy='Missing gold arguments mean unavailable supervision, never an actor-visible unknown label or empty call list.',
        primitive_scope='Noul plus project typed_call_list/typed_tool_sequence extensions; no synthetic ordinal targets.')


def request(native,bank):
    validate(native['input'])
    if native['split']!='external_test' or bank['protocol']!=PROTOCOL or bank['registry_sha256']!=REGISTRY_SHA256:
        raise ValueError('external-test native input and registered task bank required')
    # The task-bank reference is common to every sample: no gold-derived tool
    # filtering, question availability or missing-argument hints enter the input.
    return dict(id=native['id'],tier=native['tier'],split='external_test',protocol=PROTOCOL,
        native_request_sha256=digest(native),
        input=dict(context=deepcopy(native['input']),task_bank_sha256=digest(bank)))


def target(request_row,native_target,bank,official):
    validate_projected_target(native_target,official)
    if (native_target['id']!=request_row['id'] or native_target['tier']!=request_row['tier']
            or request_row['input']['task_bank_sha256']!=digest(bank)):
        raise ValueError('native target and Jev request binding differ')
    return dict(id=request_row['id'],request_sha256=digest(request_row),
        native_target=deepcopy(native_target),
        unavailable_decisions=[d['id'] for d in bank['decisions'] if d['decision_type']=='typed_call_list'
            and any(a['name']==d['tool_name'] and a['argument_evidence']=='unavailable'
                    for a in native_target['additional_annotations'])])


def answer(gold,decision):
    """Return supervision status separately from a valid JSON null argument."""
    native=gold['native_target']
    entries=[dict(name=native['primary']['name'],arguments=native['primary']['arguments'],argument_evidence='structured_primary'),
             *native['additional_annotations']]
    if decision['decision_type']=='typed_tool_sequence':
        return dict(status='available',value=[e['name'] for e in entries])
    selected=[e for e in entries if e['name']==decision['tool_name']]
    if decision['decision_type']=='noul':
        present=bool(selected)
        return dict(status='available',value=present if decision['id'].endswith('.present') else not present)
    if decision['decision_type']!='typed_call_list':raise ValueError('registered decision type required')
    if any(e['argument_evidence']=='unavailable' for e in selected):
        return dict(status='unavailable',reason='source_additional_arguments_missing',value=None)
    return dict(status='available',value=[deepcopy(e['arguments']) for e in selected])


def reconstruct(bank,answers):
    """Predictions only: require pair/list consistency and restore cross-tool order."""
    if set(answers)!={d['id'] for d in bank['decisions']}:raise ValueError('complete decision answers required')
    lists={};positions={}
    for tool in bank['tools']:
        base='tool.'+tool['tool_id'];calls=answers[base+'.calls']
        if type(calls) is not list:raise ValueError('complete typed call list required')
        validate_calls([dict(name=tool['tool_name'],arguments=a) for a in calls])
        if (type(answers[base+'.present']) is not bool or type(answers[base+'.absent']) is not bool
                or answers[base+'.present']!=bool(calls) or answers[base+'.absent']==bool(calls)):
            raise ValueError('explicit Noul and call list conflict')
        lists[tool['tool_name']]=calls;positions[tool['tool_name']]=0
    sequence=answers['tools.sequence']
    if type(sequence) is not list or any(type(n) is not str or n not in lists for n in sequence):
        raise ValueError('ordered registry tool names required')
    result=[]
    for name in sequence:
        index=positions[name]
        if index>=len(lists[name]):raise ValueError('sequence multiplicity exceeds predicted calls')
        result.append(dict(name=name,arguments=deepcopy(lists[name][index])));positions[name]+=1
    if any(positions[n]!=len(calls) for n,calls in lists.items()):raise ValueError('sequence omits predicted calls')
    return result


def verify_gold(request_row,gold,bank,official):
    if (gold['id']!=request_row['id'] or gold['request_sha256']!=digest(request_row)
            or request_row['input']['task_bank_sha256']!=digest(bank)):
        raise ValueError('Jev request/target/bank identity differs')
    if gold!=target(request_row,gold['native_target'],bank,official):raise ValueError('gold coverage changed')
    answers={d['id']:answer(gold,d) for d in bank['decisions']}
    missing=[key for key,value in answers.items() if value['status']=='unavailable']
    if missing!=gold['unavailable_decisions']:raise ValueError('explicit missing supervision differs')
    if not missing:
        rebuilt=reconstruct(bank,{k:v['value'] for k,v in answers.items()})
        if typed(rebuilt)!=typed(gold['native_target']['full_calls']):raise ValueError('original complete call sequence lost')
    elif gold['native_target']['full_calls'] is not None:
        raise ValueError('partial arguments cannot claim full sequence gold')
    return answers
