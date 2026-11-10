"""Input-only, single-decision Omni adapter for the SFC-Jev bundle protocol."""
import hashlib
import json

from .sfc_definitions import definitions
from .sfc_jev import native_from_request, request_from_native
from .sfc_omni import messages as native_messages

# Reviewed instruction prefixes from TRAIN only. Unknown text must not be deleted.
PREFIX_HASHES = frozenset('''
baab94ba1237c396569ec9bd1bb2903eef7c37d851b614d643cdd7f25e92446a
f6011aaedd9408a849d31415f9c521bba6598cb18debdae26ed04420b1ee5f8f
1f73c39882be6c922d4b91168a86311e30293dacc2b3a1f792507de1952091cb
a4e102160c92ad7caf07c4e516a8edb3d6c6669f90fbbb5f1187ef671aaeaed6
760094b2d0d79274bb3b7b72d7f36a10d32aa708ccadc2f9332dad1e96370309
3f6bcd6cfaac19675128cd0e861ed3b41e67ed79c21e310d4fcae39d4732d26f
a5251832ffa6a49e47d7878847df760ca619fadaf672b5a6d1a010c1e826f6d8
a5fe294af29c2c7e896c04e5b740a5b7ad5ed9a65f2b8bd5d7ea9907a1ec3952
774573a905b65086f7e9012d25e9f7e8863c77335026f265a481179afa6c1b8e
9b28215e2804037634d436a8ba5795537e2550c849d0375f0883e3b144733049
'''.split())
FOLLOWUP = ('Based on the conversation history and the supplemental information in the following audio, '
            'please provide the complete and final list of all required function calls, '
            'ensuring all parameters are correct.\n<audio>')
INSTRUCTION = (
    'Listen to the supplied audio and evaluate the current decision using the available functions '
    'and environment. For a follow-up, use both audio turns and the supplied assistant history '
    'to determine the complete final request. The supplied history is context, not the required '
    'format for your current answer. If a required parameter is missing, use the literal string '
    '"NAN". Preserve all parameter types, repeated calls, array order and argument bindings. '
    'Answer only the current decision, with no explanation or Markdown.\n\n'
)


def validate_bundle(request):
    if request != request_from_native(native_from_request(request)):
        raise ValueError('unaltered input-only SFC-Jev bundle required')


def messages(request, decision_id, audios):
    validate_bundle(request)
    inputs = request['input']['context']
    parsed = definitions(inputs)
    prefix = parsed['instruction_prefix']
    if hashlib.sha256(prefix.encode()).hexdigest() not in PREFIX_HASHES:
        raise ValueError('unreviewed original instruction prefix')
    selected = [d for d in request['input']['decisions'] if d['id'] == decision_id]
    if len(selected) != 1:
        raise ValueError('one declared decision required')
    decision = selected[0]
    result = native_messages(inputs, audios)
    # Preserve source function/environment substrings exactly, including menu order.
    source_sections = inputs['messages'][0]['content'][len(prefix):]
    contract = ('Output exactly one JSON boolean: true or false.' if decision['decision_type'] == 'noul'
                else 'Output exactly one JSON array of argument objects for the specified function. '
                     'Use [] when no call is required. Do not output function-call strings.')
    result[0] = dict(role='system', content=INSTRUCTION + source_sections
                    + '\n\n### Current decision:\n' + json.dumps(decision, ensure_ascii=False, sort_keys=True)
                    + '\n\n' + contract)
    if len(inputs['messages']) == 3:
        if inputs['messages'][2]['content'] != FOLLOWUP:
            raise ValueError('unreviewed original follow-up instruction')
        result[-1]['content'][0] = dict(type='text', text=(
            'Use the conversation history and this supplemental audio to answer the current '
            'decision about the complete final request.\n<audio>'))
    return result


def vllm_request(processor, request, decision_id, audios):
    prompt = processor.apply_chat_template(messages(request, decision_id, audios),
                                          tokenize=False, add_generation_prompt=True)
    return dict(prompt=prompt, multi_modal_data={'audio': [(wave, 16000) for wave in audios]})
