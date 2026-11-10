"""Unified audio-only Audio2Tool native-task input and local JSON protocol.

This is an explicit project baseline contract, not the historical paper prompt.
No per-example targets, transcripts or candidate filtering are accepted.
"""
from pathlib import PurePosixPath
import re

from .audio2tool_inventory import registry
from .audio2tool_dialogue_input import messages as dialogue_messages

REGISTRY_SHA256='7c50b0e8b49f514b169f4040c38ac67ca69ebaf361f754f1f3e994c7cb09d0a4'
TASK=('Listen directly to the audio and return the tool calls needed for the user\'s final request. '
      'Use the complete available tool definitions below. Respect corrections and conversation history. '
      'Return only a JSON array of objects with exactly two keys: "name" (tool name) and '
      '"arguments" (parameter object). Preserve the requested call order and repeated calls. '
      'Use parameter types and defaults from the tool definitions. Do not invent missing values. '
      'Do not include a transcript, explanation or Markdown.\n\nAvailable tools:\n')
MODES={'single_audio','dialogue_last_user','foreground_mixture'}


def system_text(registry_bytes):
    tools=registry(registry_bytes,REGISTRY_SHA256)
    if len(tools)!=152:raise ValueError('complete independent tool registry required')
    lines=[]
    for tool in tools:
        lines.extend([f"- {tool['signature']}: {tool['description']}",
                      f"  Defaults: {tool['argument_defaults']}",
                      f"  Constraints: {tool['argument_constraints']}"])
    return TASK+'\n'.join(lines)


def validate(inputs):
    if (set(inputs)!={'mode','registry_sha256','audios'} or inputs['mode'] not in MODES
            or inputs['registry_sha256']!=REGISTRY_SHA256 or not inputs['audios']):
        raise ValueError('strict audio input fields and fixed registry required')
    audios=inputs['audios'];mode=inputs['mode']
    if mode=='dialogue_last_user':
        # Delegate ordered prefix and role/path checks to the Tier7 contract.
        query=PurePosixPath(audios[0]['path']).parent.name
        dialogue_messages(dict(query_id=query,registry_sha256=REGISTRY_SHA256,
                               decision_boundary='after_last_user',audios=audios),
                          [None]*len(audios),'validation',REGISTRY_SHA256)
    else:
        if len(audios)!=1:raise ValueError('one original waveform per independent variant')
        a=audios[0]
        if set(a)!={'path','sha256','bytes'}:raise ValueError('unexpected audio field')
        p=PurePosixPath(a['path'])
        if (p.is_absolute() or '..' in p.parts or len(p.parts)!=4 or p.parts[0]!='public'
                or not re.fullmatch(r'tier[1-68]_[a-z_]+_audios',p.parts[1])
                or not re.fullmatch(r'query_\d{5}',p.parts[2])
                or not re.fullmatch('[0-9a-f]{64}',a['sha256'])
                or type(a['bytes']) is not int or a['bytes']<=0):
            raise ValueError('invalid pinned single audio path')
        mixture=p.parts[1]=='tier8_intent_blending_audios'
        if mixture!=(mode=='foreground_mixture'):
            raise ValueError('mixture protocol differs from source tier')
        if (mixture and p.name!='mixed.wav') or (not mixture and not re.fullmatch(r'speaker_\d+_.+\.wav',p.name)):
            raise ValueError('original variant or mixed waveform required')


def messages(inputs,waves,registry_bytes):
    validate(inputs)
    if len(waves)!=len(inputs['audios']):raise ValueError('all ordered waveforms required')
    system=system_text(registry_bytes)
    if inputs['mode']=='dialogue_last_user':
        query=PurePosixPath(inputs['audios'][0]['path']).parent.name
        return dialogue_messages(dict(query_id=query,registry_sha256=REGISTRY_SHA256,
            decision_boundary='after_last_user',audios=inputs['audios']),waves,
            system+'\nRespond to the final user turn using the preceding conversation.',REGISTRY_SHA256)
    instruction=('Follow only the foreground speaker and ignore the overlapping background speech.'
                 if inputs['mode']=='foreground_mixture' else 'Respond to this spoken request.')
    return [dict(role='system',content=system),dict(role='user',content=[
        dict(type='text',text=instruction),dict(type='audio',audio=waves[0])])]
