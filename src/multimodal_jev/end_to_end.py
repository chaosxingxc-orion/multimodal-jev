"""Enforce the user's audio-to-task execution scope before allocating models."""


def require_audio_task_route(config):
    if config.get('route') != 'audio-direct':
        raise ValueError('end-to-end audio-direct route required; ASR-to-text cascade is retired')
    for key in ('text_model', 'asr_evidence', 'reference_transcripts', 'transcript_path', 'oracle'):
        if config.get(key) is not None:
            raise ValueError('end-to-end route forbids separate text model or transcript evidence: ' + key)
    if not config.get('audio_model'):
        raise ValueError('end-to-end audio model required')
