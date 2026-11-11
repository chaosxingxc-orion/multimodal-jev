"""Input-only contract for one-forward, finite typed decision spans."""
from dataclasses import asdict, dataclass
import json
from .task_schema import DecisionSchema


@dataclass(frozen=True)
class SpanQuestion:
    id: str
    schema: DecisionSchema

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError('span id required')
        if self.schema.decision_type not in {'noul', 'choice', 'score'}:
            raise ValueError('finite typed spans required')

    @property
    def values(self):
        s = self.schema
        values = ((False, True) if s.decision_type == 'noul' else
                  tuple(c.id for c in s.candidates) if s.decision_type == 'choice' else s.levels)
        return values + ((None,) if s.allow_unknown else ())


@dataclass(frozen=True)
class SpanRequest:
    audio: str
    questions: tuple[SpanQuestion, ...]
    final_id: str
    requirements: tuple[str, ...]
    version: str = 'jev-audio-spans-v1'

    def __post_init__(self):
        if not self.audio or not self.questions or not self.requirements:
            raise ValueError('audio, questions and thinking requirements required')
        if len({q.id for q in self.questions}) != len(self.questions):
            raise ValueError('duplicate span id')
        if self.questions[-1].id != self.final_id:
            raise ValueError('final question must occupy last span')
        if any(type(r) is not str or not r.strip() for r in self.requirements):
            raise ValueError('nonempty thinking requirements required')
        if '[JEV_SPAN_' in json.dumps(asdict(self), ensure_ascii=False):
            raise ValueError('reserved span marker in input')

    def model_input(self):
        return asdict(self)


def span_labels(tokenizer, questions):
    """Exact single-token symbols shared by prompt, LM-head readout and targets."""
    import string
    labels = string.ascii_uppercase + string.ascii_lowercase + string.digits
    result = []
    for q in questions:
        if len(q.values) > len(labels):
            raise ValueError('too many values for native symbol readout')
        symbols = list(labels[:len(q.values)])
        ids = [tokenizer.encode(s, add_special_tokens=False) for s in symbols]
        if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != len(ids):
            raise ValueError('decision symbols must be distinct single tokens')
        result.append({'symbols':symbols,'token_ids':[x[0] for x in ids],'values':q.values})
    return result


def encode_span_request(processor, request, audio):
    """Locate readouts AFTER audio expansion; no targets or answer prefixes.

    Use exact offsets in the fully rendered text, then require each marker token
    sequence to occur uniquely in the processor's expanded sequence. Fail closed
    on tokenizer/template mismatch instead of guessing positions.
    """
    import torch
    system = ('English\nFill all typed decision spans from the audio in one forward pass. '
              'Intermediate and final questions are provided. Do not generate text.\n'
              + json.dumps({'version': request.version, 'requirements': request.requirements}, ensure_ascii=False))
    mappings = span_labels(processor.tokenizer, request.questions)
    parts = []
    for i, q in enumerate(request.questions):
        parts.append(json.dumps({'id':q.id,'final':q.id==request.final_id,
                                 'schema':asdict(q.schema),
                                 'symbols':dict(zip(mappings[i]['symbols'],q.values))}, ensure_ascii=False)
                     + f'\n[JEV_SPAN_{i+1}]')
    # Stock ASR templates discard user text, but retain an assistant context.
    # This is a fully supplied masked context, never a generated response.
    messages = [{'role':'system','content':[{'type':'text','text':system}]},
                {'role':'user','content':[{'type':'audio','audio':audio}]},
                {'role':'assistant','content':'\n'.join(parts)}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if system not in text or any(part not in text for part in parts):
        raise ValueError('template dropped task definitions or thinking requirements')
    raw = processor.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    batch = processor(text=[text], audio=[audio], sampling_rate=16000, return_tensors='pt')
    expanded = batch['input_ids'][0].tolist()
    positions = []
    for i in range(len(request.questions)):
        marker = f'[JEV_SPAN_{i+1}]'
        if text.count(marker) != 1:
            raise ValueError('template lost or duplicated span marker')
        start = text.index(marker); end = start+len(marker)
        hits = [j for j,(a,b) in enumerate(raw['offset_mapping']) if a < end and b > start]
        if not hits or hits != list(range(hits[0],hits[-1]+1)):
            raise ValueError('invalid marker token offsets')
        ids = raw['input_ids'][hits[0]:hits[-1]+1]
        matches = [j for j in range(len(expanded)-len(ids)+1) if expanded[j:j+len(ids)] == ids]
        if len(matches) != 1:
            raise ValueError('marker does not align uniquely after audio expansion')
        if matches[0] < 1:
            raise ValueError('no causal predecessor for decision marker')
        positions.append(matches[0]-1)
    if positions != sorted(set(positions)):
        raise ValueError('span positions are not strictly ordered')
    # A causal readout before audio would have no acoustic evidence.
    audio_id = processor.tokenizer.convert_tokens_to_ids('<|audio_pad|>')
    audio_positions = [i for i,v in enumerate(expanded) if v == audio_id]
    if not audio_positions or max(audio_positions) >= positions[0]:
        raise ValueError('all audio tokens must precede span readouts')
    return dict(batch), torch.tensor([positions], dtype=torch.long)
