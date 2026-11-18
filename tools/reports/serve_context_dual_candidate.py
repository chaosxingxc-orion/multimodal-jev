"""Versioned entrypoint derived from the pinned bulk-service lifecycle."""
import hashlib
from pathlib import Path

SOURCE_SHA='528fc1c57a727c9d4f261e1b5de325ad92629e5519fb1d5e383796ca77449860'


def service_namespace():
    p=Path(__file__).with_name('serve_context_batched_candidate.py')
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:
        raise ValueError('bulk service lifecycle changed')
    source=raw.decode()
    replacements=[
        ('from tools.reports.context_batched_owner_candidate import owner_class, PROFILE',
         'from tools.reports.context_dual_rollout_candidate import owner_class, RUNTIME as PROFILE'),
        ("PROTOCOL = 'bulk_segmented_candidate_v1'", "PROTOCOL = 'bulk_dual_logits_candidate_v1'")]
    for before,after in replacements:
        if source.count(before)!=1:raise ValueError('unexpected service source')
        source=source.replace(before,after)
    namespace={'__name__':__name__+'.derived','__file__':__file__}
    exec(compile(source,'<dual-bulk-service>','exec'),namespace)
    return namespace


if __name__=='__main__':
    service_namespace()['main']()
