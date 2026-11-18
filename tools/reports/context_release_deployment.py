"""Verify every immutable release pin is supplied by a read-only bind.

No inference, writes, relocation, or mutation of the dataset publication.
"""
import copy
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from tools.reports.context_cached_storage_operations import mounts


def inherited(c):
    result={k:c[k] for k in ('host','lease','shared_root','driver_root')}
    if 'release_evidence' in c:result['release_evidence']=copy.deepcopy(c['release_evidence'])
    return result


def audit(argv,config):
    root=Path(config['release']);manifest=root/'MANIFEST.json'
    if file_hash(manifest)!=config['release_sha256']:raise ValueError('release identity differs')
    m=json.loads(manifest.read_text());publication=root/'publication-config.json'
    # v2 pins are release-relative; use its own complete bundle verifier.
    if m['schema']=='slurp-official-clean-intent-v2':
        from multimodal_jev.slurp_portable_release import config as portable_config
        portable_config(root,config['release_sha256'])
        return dict(status='self_contained_release_no_external_publication_pins',external_pins=0)
    if m['schema']!='slurp-official-clean-intent-v1':raise ValueError('known publication contract required')
    if m['files'].get('publication-config.json')!=file_hash(publication):
        raise ValueError('release publication config differs')
    pins=json.loads(publication.read_text())['pins'];bindings=mounts(argv)
    for name,digest in pins.items():
        matches=[b for b in bindings if Path(name).is_relative_to(b[1])]
        if not matches:raise ValueError('publication dependency not mounted: '+name)
        source,dest,readonly=max(matches,key=lambda b:len(Path(b[1]).parts))
        if not readonly:raise ValueError('publication dependency writable: '+name)
        actual=Path(source)/Path(name).relative_to(dest)
        if not actual.is_file() or file_hash(actual)!=digest:
            raise ValueError('mounted publication dependency differs: '+name)
    return dict(status='complete_readonly_release_publication_pins_verified',external_pins=len(pins),
        release_sha256=config['release_sha256'],actual_remote_mounts_pending=True)
