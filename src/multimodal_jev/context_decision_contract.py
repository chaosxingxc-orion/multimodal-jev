"""Input-only applicability questions and marker-self compilation for Jev.

This independent protocol does not reinterpret old categorical SpanRequest files.
All definitions precede all decision markers. History is supplied context, never
gold labels or an implicit cache of earlier model calls.
"""
from dataclasses import asdict, dataclass
import json

DECISION_TOKEN = '<|jev_decision|>'
RESERVED_TIMESTAMP_TOKEN = '<|jev_probe_reserved_id_151705|>'


@dataclass(frozen=True)
class ApplicabilityQuestion:
    id: str
    definition: str
    question: str

    def __post_init__(self):
        if any(type(x) is not str or not x.strip() for x in (self.id, self.definition, self.question)):
            raise ValueError('nonempty question ID, definition and question required')


@dataclass(frozen=True)
class ContextDecisionRequest:
    audio: str
    questions: tuple[ApplicabilityQuestion, ...]
    requirements: tuple[str, ...]
    history: tuple[str, ...] = ()
    version: str = 'jev-audio-context-applicability-v1'

    def __post_init__(self):
        if type(self.audio) is not str or not self.audio.strip() or not self.questions or not self.requirements:
            raise ValueError('audio, questions and requirements required')
        if any(not isinstance(q, ApplicabilityQuestion) for q in self.questions):
            raise ValueError('applicability questions required')
        if len({q.id for q in self.questions}) != len(self.questions):
            raise ValueError('duplicate question ID')
        if any(type(x) is not str or not x.strip() for x in (*self.requirements, *self.history, self.version)):
            raise ValueError('nonempty supplied context and version required')
        if any(token in json.dumps(asdict(self), ensure_ascii=False)
               for token in (DECISION_TOKEN, RESERVED_TIMESTAMP_TOKEN)):
            raise ValueError('reserved decision marker in request')

    def model_input(self):
        return asdict(self)


def render_context_request(request):
    """All rules, questions and history are available to the first readout."""
    system = ('English\nEvaluate each supplied applicability question using the audio and supplied context. '
              'Return probabilities through decision readouts. Do not generate text.')
    context = json.dumps(dict(version=request.version, requirements=request.requirements,
                              history=request.history,
                              questions=[asdict(q) for q in request.questions]),
                         ensure_ascii=False, separators=(',', ':'))
    readouts = [json.dumps({'decision_id': q.id}, ensure_ascii=False, separators=(',', ':'))
                + '\n' + DECISION_TOKEN for q in request.questions]
    return system, context + '\n' + '\n'.join(readouts)


def encode_context_request(processor, request, audio, *, decision_token_id, audio_token_id, max_tokens):
    """Use a previously audited tokenizer; never allocate or initialize tokens here."""
    import torch
    if (type(decision_token_id) is not int or type(audio_token_id) is not int
            or min(decision_token_id, audio_token_id) < 0 or decision_token_id == audio_token_id):
        raise ValueError('distinct registered decision and audio IDs required')
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError('positive explicit context limit required')
    tokenizer = processor.tokenizer
    if (tokenizer.get_vocab().get(DECISION_TOKEN) != decision_token_id
            or DECISION_TOKEN not in tokenizer.all_special_tokens
            or tokenizer.encode(DECISION_TOKEN, add_special_tokens=False) != [decision_token_id]):
        raise ValueError('audited single special decision token required')
    system, content = render_context_request(request)
    messages = [{'role': 'system', 'content': [{'type': 'text', 'text': system}]},
                {'role': 'user', 'content': [{'type': 'audio', 'audio': audio}]},
                {'role': 'assistant', 'content': content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if system not in text or content not in text or text.count(DECISION_TOKEN) != len(request.questions):
        raise ValueError('template lost or duplicated context/readouts')
    batch = dict(processor(text=[text], audio=[audio], sampling_rate=16000, return_tensors='pt'))
    ids = batch['input_ids']
    if ids.dtype != torch.long or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError('one complete expanded request required')
    if ids.shape[1] > max_tokens:
        raise ValueError('context limit exceeded; truncation forbidden')
    reserved_id=tokenizer.get_vocab().get(RESERVED_TIMESTAMP_TOKEN)
    if reserved_id is not None and (ids==reserved_id).any():
        raise ValueError('timestamp reservation cannot appear in decision input')
    positions = (ids[0] == decision_token_id).nonzero().flatten()
    audio_positions = (ids[0] == audio_token_id).nonzero().flatten()
    if (len(positions) != len(request.questions) or not len(audio_positions)
            or int(audio_positions.max()) >= int(positions[0])):
        raise ValueError('all audio must precede exactly one marker per question')
    mask = batch['attention_mask']
    if mask.shape != ids.shape or not ((mask == 0) | (mask == 1)).all() or not mask[0, positions].all():
        raise ValueError('valid complete marker attention mask required')
    return batch, positions[None, :]
