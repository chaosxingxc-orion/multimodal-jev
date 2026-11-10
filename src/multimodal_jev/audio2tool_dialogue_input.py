"""Audio-only Tier7 decision inputs, projected from the audited asset inventory.

No annotation loader. Retain post-decision audio separately for provenance;
the actor renderer accepts only the allowlisted input object.
"""
from pathlib import PurePosixPath
import re

from .audio2tool_inventory import REVISION


def validate_audio(audio, query_id, position):
    if set(audio) != {'path', 'sha256', 'bytes', 'turn', 'role'}:
        raise ValueError('unexpected audio field; transcripts and targets forbidden')
    p = PurePosixPath(audio['path'])
    if (p.parts[:3] != ('public', 'tier7_multiturn_audios', query_id)
            or len(p.parts) != 4 or p.is_absolute() or '..' in p.parts
            or type(audio['turn']) is not int or audio['turn'] != position
            or audio['role'] not in ('user', 'agent')
            or not p.name.startswith(f"turn_{position:02d}_{audio['role']}_")
            or p.suffix != '.wav' or not re.fullmatch('[0-9a-f]{64}', audio['sha256'])
            or type(audio['bytes']) is not int or audio['bytes'] <= 0):
        raise ValueError('invalid pinned ordered Tier7 audio')


def decision_input(row, registry_sha256):
    if (set(row) != {'tier', 'query_id', 'audio_semantics', 'audios'} or row['tier'] != 7
            or row['audio_semantics'] != 'ordered_conversation_turns'
            or not re.fullmatch(r'query_\d{5}', row['query_id'])
            or not re.fullmatch('[0-9a-f]{64}', registry_sha256)
            or not row['audios']):
        raise ValueError('audited Tier7 inventory and independent registry pin required')
    for i,audio in enumerate(row['audios']):
        validate_audio(audio,row['query_id'],i)
    users = [i for i,a in enumerate(row['audios']) if a['role']=='user']
    if not users or row['audios'][0]['role'] != 'user':
        raise ValueError('dialogue must contain and start with user audio')
    end = users[-1]+1
    return dict(schema='audio2tool-tier7-decision-input-v1', revision=REVISION,
        split='external_test', id='audio2tool:tier7:'+row['query_id'],
        input=dict(query_id=row['query_id'], registry_sha256=registry_sha256,
                   decision_boundary='after_last_user',
                   audios=[dict(a) for a in row['audios'][:end]]),
        provenance=dict(post_decision_audios=[dict(a) for a in row['audios'][end:]],
                        full_dialogue_audio_count=len(row['audios'])))


def messages(actor_input, waves, system_text, registry_sha256):
    """Render only audio+role evidence; caller supplies separately pinned task/schema.

    Registry hash binds the caller's task/schema construction. This renderer does
    not certify arbitrary system_text or waveform content: callers must verify
    registry bytes, audio hashes and resampling before supplying those objects.
    """
    if (set(actor_input) != {'query_id','registry_sha256','decision_boundary','audios'}
            or actor_input['registry_sha256'] != registry_sha256
            or not re.fullmatch('[0-9a-f]{64}',registry_sha256)
            or not re.fullmatch(r'query_\d{5}',actor_input['query_id'])
            or actor_input['decision_boundary'] != 'after_last_user'
            or not actor_input['audios'] or len(waves) != len(actor_input['audios'])
            or not isinstance(system_text,str) or not system_text.strip()):
        raise ValueError('strict decision input, full audio prefix and pinned task required')
    audios = actor_input['audios']
    if audios[0]['role']!='user' or audios[-1]['role']!='user':
        raise ValueError('decision prefix must start/end with user')
    content=[]
    for i,(audio,wave) in enumerate(zip(audios,waves,strict=True)):
        validate_audio(audio,actor_input['query_id'],i)
        content.extend([dict(type='text',text=f"Conversation turn {i}, speaker role {audio['role']}:"),
                        dict(type='audio',audio=wave)])
    return [dict(role='system',content=system_text),dict(role='user',content=content)]
