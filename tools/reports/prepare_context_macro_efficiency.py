"""Seal matched micro8/16/32 efficiency arms after actual DP2 replay admission.

All arms use one refreshed source tree, the same full TRAIN prefix and fresh
reference caches. No automatic dispatch or model copies.
"""
import argparse
import json
from pathlib import Path
import shutil

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports.audit_context_macro_training import audit_collected
from tools.reports.prepare_context_macro_replay import write
from tools.reports.prepare_context_dual_replay import relocate
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.context_macro_trainer_candidate import generate, experiment_window
from tools.reports.context_macro_training_operations import validate

REFRESH = (
    'tools/reports/context_macro_trainer_candidate.py',
    'src/multimodal_jev/context_policy_snapshot.py',
    'src/multimodal_jev/context_checkpoint_policy.py',
    'src/multimodal_jev/context_policy_artifacts.py',
)


def prepare(c, provenance):
    if type(c['steps']) is not int or not 3 <= c['steps'] <= 100:
        raise ValueError('explicit 3..100 global steps per arm required')
    replay, _ = load_config(c['replay_config'], c['replay_local'])
    admission = audit_collected(replay)
    if (admission['status'] != 'finite_dp2_replay_audited'
            or admission['fresh_process_replay_verified'] is not True):
        raise ValueError('actual independently audited fresh-process replay required')
    binding = json.loads((Path(replay['prepared'])/'package/REPLAY.json').read_text())
    parent = binding['parent_operations']
    source = Path(parent['prepared'])/'package'
    if verify(source) != binding['parent_manifest_sha256']:
        raise ValueError('original prefix package changed')
    output, remote, project = Path(c['output']), Path(c['remote_root']), Path(c['project'])
    old = Path(parent['remote_root'])
    if (output.exists() or output.resolve().is_relative_to(source.resolve())
            or not remote.is_absolute() or '..' in remote.parts or ':' in str(remote)
            or remote.is_relative_to(old) or old.is_relative_to(remote)
            or remote.is_relative_to(replay['remote_root']) or Path(replay['remote_root']).is_relative_to(remote)):
        raise ValueError('new disjoint efficiency paths required')
    original = json.loads((source/'training.json').read_text())
    if original['start_update'] != 0 or original['stop_update'] != 2 or original.get('resume_root'):
        raise ValueError('original two-step initialization required')
    output.mkdir(parents=True)
    arms = {}
    runtime_hash = None
    for micro in (8, 16, 32):
        name = 'micro'+str(micro)
        dest = output/name/'package';root = remote/name
        if len(str(root/'ipc').encode())+8 > 107:
            raise ValueError('bounded Unix socket path required')
        shutil.copytree(source, dest)
        (dest/'FILES.json').unlink()
        for rel in REFRESH:
            (dest/rel).parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(project/rel, dest/rel)
        generate(dest)
        sources = {str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*.py')}
        write(dest/'RUNTIME-SOURCES.json',sources)
        digest = file_hash(dest/'RUNTIME-SOURCES.json')
        if runtime_hash is not None and digest != runtime_hash:
            raise ValueError('efficiency arms must execute identical source bytes')
        runtime_hash = digest
        t = relocate(original,str(old),str(root))
        t.update(microbatch=micro,stop_update=c['steps'],experiment_kind='efficiency')
        t['dual_runtime_identity']['source_manifest_sha256'] = digest
        t['macro_client']['runtime_identity'] = t['dual_runtime_identity']
        t['macro_client']['attempt_id'] = 'efficiency-v1'
        # Keep dataset/cache pins; replace every internal code pin with current bytes.
        pins = {p:h for p,h in t['pins'].items() if not Path(p).is_relative_to(root/'package')}
        internal = {str(root/'package'/p.relative_to(dest)):file_hash(p) for p in dest.rglob('*')
            if p.is_file() and p.name not in ('training.json','service-0.json','service-1.json')}
        t['pins'] = dict(pins,**internal)
        experiment_window(t)
        write(dest/'training.json',t)
        for i in range(2):
            service = relocate(json.loads((source/f'service-{i}.json').read_text()),str(old),str(root))
            service['dual_runtime_identity'] = t['dual_runtime_identity']
            service['pins'] = internal
            write(dest/f'service-{i}.json',service)
        write(dest/'FILES.json',{str(p.relative_to(dest)):file_hash(p) for p in dest.rglob('*') if p.is_file()})
        r = json.loads((source.parent/'PREPARED.json').read_text())
        r.update(status='macro_efficiency_prepared_only',files_sha256=verify(dest),remote_root=str(root),
            ipc_root=str(root/'ipc'),runtime_identity=t['dual_runtime_identity'],
            config_provenance=provenance,efficiency=dict(microbatch=micro,steps=c['steps'],
                reference_condition='empty_at_start',replay_admission=admission),
            pending=['actual bounded run','independent every-step audit','matched efficiency comparison'])
        write(dest.parent/'PREPARED.json',r)
        operations = dict(parent,prepared=str(dest.parent),remote_root=str(root),
            manifest_sha256=r['files_sha256'],operations_root=str(Path(c['operations_root'])/name),
            tag=c['tag_prefix']+'-'+name)
        validate(operations)
        # Already-resolved machine bindings stay in local preparation output.
        write(dest.parent/'operations.json',operations)
        arms[name] = dict(prepared=str(dest.parent),remote_root=str(root),microbatch=micro,
            manifest_sha256=r['files_sha256'],operations=str(dest.parent/'operations.json'))
    result = dict(schema='context-macro-efficiency-arms-v1',status='prepared_only',arms=arms,
        steps_per_arm=c['steps'],runtime_sources_sha256=runtime_hash,
        same=['initial_policy','algorithm','credit_arithmetic','full_TRAIN_prefix','global128',
              'DP2_dualTP1','empty_reference_cache','checkpoint_every_update','source_bytes'],
        changed=['microbatch','DP2_length_partition','isolated_output_paths'],
        training_dispatch_allowed=False,efficacy_claim=False)
    write(output/'ARMS.json',result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    import torch
    torch.set_num_threads(1)
    print(json.dumps(prepare(c,provenance),indent=2))
