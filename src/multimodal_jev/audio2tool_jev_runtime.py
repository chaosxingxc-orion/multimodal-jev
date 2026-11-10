"""Single-decision audio-only rendering for the full Audio2Tool Jev bank."""
import json

from .audio2tool_input import TASK,messages as native_messages,system_text,validate
from .audio2tool_inference import digest
from .audio2tool_jev import PROTOCOL,task_bank

INSTRUCTION=(
    'Listen directly to the supplied audio and answer only the current decision about the final user request. '
    'Use the complete available tool definitions and any supplied conversation history. '
    'Preserve repeated calls, parameter types, nested values and call order. '
    'Follow tool defaults and constraints; do not invent missing values. '
    'Return only the requested JSON value, without a transcript, explanation or Markdown.\n\n')


def messages(request,decision_id,waves,bank,registry_bytes):
    if (set(request)!={'id','tier','split','protocol','native_request_sha256','input'}
            or request['protocol']!=PROTOCOL or request['split']!='external_test'
            or set(request['input'])!={'context','task_bank_sha256'}
            or bank!=task_bank(registry_bytes) or request['input']['task_bank_sha256']!=digest(bank)):
        raise ValueError('unaltered input-only external-test Jev request and registry bank required')
    inputs=request['input']['context'];validate(inputs)
    selected=[d for d in bank['decisions'] if d['id']==decision_id]
    if len(selected)!=1:raise ValueError('one registered decision required')
    decision=selected[0];result=native_messages(inputs,waves,registry_bytes)
    definition=system_text(registry_bytes)
    if not definition.startswith(TASK):raise ValueError('registered complete tool definition rendering required')
    formats={'noul':'Output exactly one JSON boolean: true or false.',
             'typed_call_list':'Output exactly one JSON array of argument objects for this tool; [] means no call.',
             'typed_tool_sequence':'Output exactly one JSON array of tool-name strings; preserve repetitions.'}
    context=('Use the whole supplied conversation to answer about the final user request. '
             if inputs['mode']=='dialogue_last_user' else '')
    if inputs['mode']=='foreground_mixture':context+='Follow only the foreground speaker; ignore overlapping background speech. '
    result[0]=dict(role='system',content=INSTRUCTION+context+'\nAvailable tools:\n'+definition[len(TASK):]
        +'\n\nCurrent decision:\n'+json.dumps(decision,ensure_ascii=False,sort_keys=True)
        +'\n\n'+formats[decision['decision_type']])
    return result
