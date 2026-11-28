"""Package sample-specific Stage1 using immutable admitted layouts and caches."""
from pathlib import Path
import shutil
import sys
import tarfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.source_snapshot import freeze
from tools.reports.card_scoring_benchmark import pinned
from tools.reports.prepare_tpo_windows import macros


def package(c):
    previous = Path(c['layout_package'])
    prior = pinned(dict(path=previous/'operations-local.json',sha256=c['layout_package_sha256']))
    prior = prior['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    root = Path(c['output'])
    config = dict(c['deployment'])
    if any(config[k] != prior[k] for k in ('host','image','lease','devices','prepared_input','read_only_mounts')):
        raise ValueError('the new Stage1 must retain the admitted layout and environment')
    if config['remote_root'] == prior['remote_root'] or config['operations'] == prior['operations']:
        raise ValueError('the new Stage1 requires fresh output directories')
    audit = pinned(c['audit'])
    manifest = Path(c['audit']['path']).with_name('manifest.json')
    data = pinned(dict(path=manifest,sha256=audit['manifest_sha256']))
    if (audit['schema'] != 'jev-audited-instance-negative-manifest-v1'
            or data['schema'] != 'jev-stage1-instance-negatives-v1'
            or audit['epoch'] != c['epoch'] or data['epoch'] != c['epoch']
            or audit['reference'] != data['reference']
            or prior['entry_pins']['speech-partners.json'] != data['reference']['semantic_sha256']
            or len(data['rows']) != c['population'] or audit['recordings'] != c['population']):
        raise ValueError('instance manifest does not match the requested training epoch')
    entry = root/'entry'
    entry.mkdir(parents=True)
    old = previous/'entry'
    roles = ('prepare','finite','formal')
    changed = {f'{r}.json' for r in roles}
    for rel, sha in prior['entry_pins'].items():
        if rel not in changed and rel != 'source/FILES.json':
            if Path(rel).parent != Path('.'):
                raise ValueError('unexpected layout entry path')
            shutil.copyfile(old/rel,entry/rel)
            if file_hash(entry/rel) != sha:
                raise ValueError(f'layout dependency changed: {rel}')
    shutil.copyfile(manifest,entry/'instance-negatives.json')
    if file_hash(entry/'instance-negatives.json') != audit['manifest_sha256']:
        raise ValueError('instance manifest changed during package assembly')
    pin = dict(path='/jev-entry/instance-negatives.json',sha256=audit['manifest_sha256'])
    snap = freeze(Path(__file__).resolve().parents[2],entry/'source',patterns=c['source_patterns'],
                  io_workers=c['io_workers'])
    resume = None
    if c['epoch']:
        proof = pinned(c['resume_receipt'])
        if (not proof['passed'] or proof['container_exited'] is not True or not proof['completed_window']
                or proof['host'] != config['host'] or proof['stage1_instance_epoch'] != c['epoch']-1
                or proof['checkpoint_sha256'] != data['reference']['checkpoint_sha256']
                or proof['policy_id'] != data['reference']['policy_id']):
            raise ValueError('second epoch reference and exact resume checkpoint must match')
        path = Path(proof['policy']).parent
        config['read_only_mounts'] = [*config['read_only_mounts'],str(path)+':/jev-instance-resume']
        resume = dict(root='/jev-instance-resume',step=proof['updates'],sha256=proof['checkpoint_sha256'])
    from multimodal_jev.card_contrast import Directions
    for role in roles:
        name = role+'.json'
        ops = pinned(dict(path=old/name,sha256=prior['entry_pins'][name]))['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
        objective = Directions.from_objective(ops['triplet_objective'])
        if (objective.positives, objective.cards, objective.recordings) != tuple(c['negative_manifest_slots']):
            raise ValueError('sampling manifest slot geometry differs from the admitted training objective')
        del ops['class_logit_ema'], ops['class_scoring']
        ops.update(source=dict(root='/jev-entry/source',sha256=snap['manifest_sha256']),
                   instance_negative_manifest=pin,instance_epoch=c['epoch'])
        if resume:
            ops['instance_resume'] = resume
            ops.pop('finite_pressure',None)
        commit_json(entry/name,macros('NATIVE_SEAL_CONTAINER_CONFIG',ops))
    archive = root/'entry.tar'
    with tarfile.open(archive,'x') as f:
        f.add(entry,arcname='entry')
    pins = [*entry.glob('*.json'),entry/'source/FILES.json']
    config.update(archive=str(archive),entry_pins={str(p.relative_to(entry)):file_hash(p) for p in sorted(pins)},
                  entry_archive_sha256=file_hash(archive),chains=prior['chains'],device_group_lease=True,
                  wait_for_lease=True,idle_devices=config['devices'])
    commit_json(root/'operations-local.json',macros('NATIVE_SEAL_OPERATIONS_CONFIG',config))


if __name__ == '__main__':
    package(load_config(*sys.argv[1:])[0])
