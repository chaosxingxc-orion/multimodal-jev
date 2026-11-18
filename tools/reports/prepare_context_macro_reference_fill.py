"""Seal a fixed-reference precompute package from an already sealed training package.

The fill reuses the training package's Python sources byte for byte, so the
retained cache namespace it populates is exactly the one training will read.
Only the client attempt id, the retained reference cache and the fill binding
change; no source is regenerated and no dataset is copied.
"""
import argparse
import json
from pathlib import Path
import shutil

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.prepare_context_macro_replay import write
from tools.reports.run_context_batch_probe_candidate import verify

SCHEMA='context-fixed-reference-fill-v1'
ENTRY_COUNT=40513
SEALED_FILES=('FILES.json','training.json','fill.json','REPLAY.json','MACRO-AUDIT.json')
HOST_STATUS=('macro_training_prefix_prepared_only','macro_efficiency_prepared_only')


def prepare(c, provenance):
    prepared=Path(c['training_prepared']);source=prepared/'package'
    receipt=json.loads((prepared/'PREPARED.json').read_text())
    if receipt['status'] not in HOST_STATUS:
        raise ValueError('sealed training package required as the fill host')
    if verify(source)!=c['training_manifest_sha256'] or receipt['files_sha256']!=c['training_manifest_sha256']:
        raise ValueError('sealed training package changed')
    if type(c['expected_entries']) is not int or c['expected_entries']!=ENTRY_COUNT:
        raise ValueError('frozen TRAIN population size required')
    original=json.loads((source/'training.json').read_text())
    if original['start_update']!=0 or original.get('resume_root'):
        raise ValueError('fill host must start from the sealed initialization')
    cache=Path(c['reference_cache'])
    output=Path(c['output']);remote=Path(c['remote_root']);ipc=remote/'ipc'
    if (output.exists() or output.resolve().is_relative_to(source.resolve())
            or not remote.is_absolute() or '..' in remote.parts or ':' in str(remote)
            or remote.is_relative_to(Path(receipt['remote_root']))
            or Path(receipt['remote_root']).is_relative_to(remote)
            or not cache.is_absolute() or '..' in cache.parts or ':' in str(cache)
            or cache.is_relative_to(remote) or remote.is_relative_to(cache)
            or len(str(ipc).encode())+8>107):
        raise ValueError('disjoint fill root and retained cache required')
    sources={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*.py') if p.is_file()}
    if sources!=json.loads((source/'RUNTIME-SOURCES.json').read_text()):
        raise ValueError('fill host Python sources must match its runtime manifest')
    output.mkdir(parents=True);dest=output/'package';shutil.copytree(source,dest)
    for name in SEALED_FILES:
        (dest/name).unlink(missing_ok=True)
    t=relocate(original,receipt['remote_root'],str(remote))
    t.update(reference_cache=str(cache),reference_condition='precomputed_retained',
        reference_index=t['source_root']+'/REFERENCE-INDEX.json')
    t['macro_client']['attempt_id']='reference-fill-v1'
    fill=dict(schema=SCHEMA,source_root=t['source_root'],source_policy=t['source_policy'],
        source_policy_id=t['source_policy_id'],base_id=t['base_id'],source_contract=t['source_contract'],
        reference_adapter=t['reference_adapter'],dual_runtime_identity=t['dual_runtime_identity'],
        dual_runtime_manifest=t['dual_runtime_manifest'],encoded_audio_cache=t['encoded_audio_cache'],
        audio_mirror_root=t['audio_mirror_root'],reference_cache=t['reference_cache'],
        audio_token_id=t['audio_token_id'],
        expected_entries=c['expected_entries'],macro_client=t['macro_client'],
        output=str(remote/'run'))
    # Keep immutable data pins; every internal code pin is re-derived from the
    # copied bytes, and files written below are never self-pinned.
    skipped=set(SEALED_FILES)|{'service-0.json','service-1.json'}
    external={p:h for p,h in t['pins'].items() if not Path(p).is_relative_to(receipt['remote_root'])}
    internal={str(remote/'package'/p.relative_to(dest)):file_hash(p) for p in dest.rglob('*')
        if p.is_file() and str(p.relative_to(dest)) not in skipped}
    pins=dict(external,**internal)
    fill['pins']=pins
    write(dest/'fill.json',fill)
    for i in range(2):
        service=relocate(json.loads((source/f'service-{i}.json').read_text()),
                         receipt['remote_root'],str(remote))
        service['pins']=internal
        write(dest/f'service-{i}.json',service)
    write(dest/'FILES.json',{str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()})
    result=dict(status='reference_fill_prepared_only',image=receipt['image'],files_sha256=verify(dest),
        remote_root=str(remote),ipc_root=str(ipc),runtime_identity=original['dual_runtime_identity'],
        reference_cache=str(cache),device_scope=[[6],[7]],
        training_manifest_sha256=c['training_manifest_sha256'],
        runtime_sources_sha256=file_hash(dest/'RUNTIME-SOURCES.json'),config_provenance=provenance,
        training_dispatch_allowed=False,finite_npu_dispatch_allowed=False,
        pending=['retained-cache fill on cards 6/7','independent fill journal audit',
                 'locked training seal bound to the audited index'])
    write(output/'PREPARED.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(prepare(c,provenance),indent=2))
