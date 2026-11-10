"""Single-token typed readout over explicitly supplied audio-task alternatives."""
import json
import math
from .consumption import validate_request

LABELS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ASR_TASK_PREFIX = 'language English<asr_text>'


def audio_messages(waveform, system, output_prefix=''):
    """Optional fixed protocol header, never an audio-derived transcript."""
    if output_prefix not in ('', ASR_TASK_PREFIX):
        raise ValueError('only registered fixed ASR header is permitted')
    messages = [{'role': 'system', 'content': [{'type': 'text', 'text': 'English\n' + system}]},
                {'role': 'user', 'content': [{'type': 'audio', 'audio': waveform}]}]
    if output_prefix:
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': output_prefix}]})
    return messages, ({'continue_final_message': True} if output_prefix
                      else {'add_generation_prompt': True})


def alternatives(request):
    validate_request(request); schema = request['input']['schema']
    kind = schema['decision_type']
    if kind == 'choice':
        options = [(c['id'], c['description']) for c in schema['candidates']]
    elif kind == 'noul':
        options = [(False, 'No: the proposition in the question is false.'),
                   (True, 'Yes: the proposition in the question is true.')]
    elif kind == 'score':
        options = [(v, 'Ordinal level ' + str(v)) for v in schema['levels']]
    else:
        raise ValueError('finite direct readout cannot replace open_slot or full native SLURP')
    if schema['allow_unknown']:
        options.append((None, 'Unknown: evidence is insufficient or none of the listed alternatives is valid.'))
    if not 2 <= len(options) <= len(LABELS): raise ValueError('readout requires 2 to 36 alternatives')
    return [{'label': LABELS[i], 'value': value, 'description': description}
            for i, (value, description) in enumerate(options)]


def prompt(request, instruction):
    options = alternatives(request)
    # Request/source IDs and truth-bearing view suffixes never enter the prompt.
    definition = {'question': request['input']['schema']['question'],
                  'decision_type': request['input']['schema']['decision_type'],
                  'alternatives': options}
    return instruction + '\nTask definition:\n' + json.dumps(definition, ensure_ascii=False, sort_keys=True), options


def label_tokens(tokenizer, options):
    tokens = []
    for option in options:
        ids = tokenizer.encode(option['label'], add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids, skip_special_tokens=False) != option['label']:
            raise ValueError('readout label must be exactly one reversible token')
        tokens.append(ids[0])
    if len(set(tokens)) != len(tokens): raise ValueError('label tokens collide')
    return tokens


def decision(options, logits):
    if len(options) != len(logits) or not logits or any(not math.isfinite(x) for x in logits):
        raise ValueError('finite complete alternative logits required')
    peak = max(logits); weights = [math.exp(x-peak) for x in logits]; total = sum(weights)
    probabilities = [x/total for x in weights]
    index = max(range(len(logits)), key=lambda i: logits[i])
    return {'value': options[index]['value'], 'selected_index': index,
            'probabilities': probabilities, 'candidate_logits': logits}
