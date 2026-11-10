#!/usr/bin/env python3
"""Explicit same-feedback revision after a truncated reasoning-only response."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json,file_hash
from configure_omni_teacher import configure


def prepare(source, output, executor):
    result=json.loads((source/'response/result.json').read_text())
    prior=json.loads((source/'prepared.json').read_text())
    cfg=json.loads((source/'executor-config.private.json').read_text())
    if (result['state']!='response_received' or result['http_status']!=200
            or result['finish_reasons']!=['length']
            or file_hash(source/'response/response.sse')!=result['response_sha256']
            or cfg['thinking']!={'type':'enabled'} or cfg['max_tokens']!=32768
            or file_hash(source/'messages.private.json')!=prior['messages_sha256']
            or cfg['messages']['sha256']!=prior['messages_sha256']
            or Path(cfg['messages']['path']).resolve()!=(source/'messages.private.json').resolve()
            or prior['selection_or_test_feedback_used'] is not False
            or (source/'candidates').exists()):
        raise ValueError('bound truncated unaccepted TRAIN teacher response required')
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    path=output/'messages.private.json'
    path.write_bytes((source/'messages.private.json').read_bytes());path.chmod(0o600)
    receipt={**prior,'source_request':str(source),'previous_usage':result['usage'],
        'previous_response_sha256':result['response_sha256'],
        'scope':'explicit budget revision after truncation; identical TRAIN messages; thinking disabled and max_tokens8192; no automatic replay'}
    atomic_json(output/'prepared.json',receipt)
    configure(output,source/'executor-config.private.json',executor,thinking='disabled',max_tokens=8192)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','output','executor'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output,a.executor),indent=2))
