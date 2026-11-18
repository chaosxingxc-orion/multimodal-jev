"""Audit the exact Qwen-ASR decision-token extension without allocating tokens.

The timestamp guard is a reservation only: it must never be supplied in a Jev
request. This protocol is for decision pooling, not forced-alignment decoding.
Runtime/probability acceptance and immutable policy export remain separate.
"""
import json
from pathlib import Path

from .baseline import file_hash
from .context_decision_contract import DECISION_TOKEN,RESERVED_TIMESTAMP_TOKEN


def validate_processor_extension(base,processor):
    base,processor=Path(base),Path(processor)
    original=json.loads((base/'tokenizer.json').read_text())
    extended=json.loads((processor/'tokenizer.json').read_text())
    config=json.loads((base/'config.json').read_text())
    reserved={}
    def walk(value,prefix=''):
        if not isinstance(value,dict):return
        for k,v in value.items():
            if k.endswith(('_token_id','_token_ids')):
                for i in v if isinstance(v,list) else [v]:
                    if type(i) is int:reserved.setdefault(i,[]).append(prefix+k)
            walk(v,prefix+k+'.')
    walk(config)
    if config.get('timestamp_token_id')!=151705 or 151706 in reserved:
        raise ValueError('reviewed timestamp/decision reservation differs')
    if {k:v for k,v in original.items() if k!='added_tokens'}!={k:v for k,v in extended.items() if k!='added_tokens'}:
        raise ValueError('original tokenizer model or normalization changed')
    old={x['id']:x for x in original['added_tokens']}
    new={x['id']:x for x in extended['added_tokens']}
    if len(old)!=len(original['added_tokens']) or len(new)!=len(extended['added_tokens']):
        raise ValueError('duplicate token ID')
    if any(new.get(i)!=v for i,v in old.items()) or set(new)-set(old)!={151705,151706}:
        raise ValueError('existing special tokens or extension IDs differ')
    for i,text in ((151705,RESERVED_TIMESTAMP_TOKEN),(151706,DECISION_TOKEN)):
        expected=dict(id=i,content=text,single_word=False,lstrip=False,rstrip=False,normalized=False,special=True)
        if new[i]!=expected:raise ValueError('marker attributes differ')
    vocab=original['model']['vocab']
    if (max([*vocab.values(),*old])!=151704 or
            any(v in (151705,151706) for v in vocab.values()) or
            config['text_config']['vocab_size']<=151706):
        raise ValueError('existing embedding/token vocabulary boundary differs')
    return dict(schema='qwen3-asr-jev-token-extension-v1',decision_id=151706,
        timestamp_reserved_id=151705,original_tokenizer_preserved=True,
        files={str(p):file_hash(p) for p in (base/'config.json',base/'tokenizer.json',processor/'tokenizer.json')},
        scope='Tokenizer JSON extension and config reservations only; processor roundtrip, embedding row and runtime admission are separate.')
