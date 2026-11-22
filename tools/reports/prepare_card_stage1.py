"""Build the knowledge-card Stage1 package for the experiment ledger."""
import json
from pathlib import Path
import shutil
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.source_snapshot import freeze
from tools.reports.freeze_card_assignments import compile_plan, read


def load(path, sha):
    return json.loads(read(Path(path),sha))


def run(c):
    root=Path(c['output'])
    entry=root/'entry'
    entry.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2]
    design=load(c['design'],c['design_sha256'])
    for stage in ('stage1','stage2'):
        design[stage]['lora'].update(c['lora'])
    design.update(status='registered',model=c['model_description'])
    commit_json(entry/'design.json',design)
    compile_plan(c,entry,file_hash(entry/'design.json'))
    parent=load(c['parent'],c['parent_sha256'])
    native=load(c['native_cache'],c['native_cache_sha256'])
    if native['source_identity']!=parent['release_sha256'] or len(native['templates']['both']['suffix'])!=len(native['templates']['native']['suffix']) or len(native['templates']['both']['prefix'])!=len(native['templates']['native']['prefix']):
        raise ValueError('historical lengths must share the native token count')
    def block(path):
        return [(r['id'],r['input_tokens']) for r in json.loads(path.read_bytes())['records'] if r['weight']]
    lengths={}
    with ThreadPoolExecutor(c['io_workers']) as pool:
        for values in pool.map(block, sorted(Path(c['batches']).glob('batch-*.json'))):
            for uid,length in values:
                if lengths.setdefault(uid,length)!=length:
                    raise ValueError('inconsistent acoustic lengths')
    if len(lengths)!=native['full_verified'] or max(lengths.values())>native['max_tokens']:
        raise ValueError('full source lengths differ from cache proof')
    commit_json(entry/'native-lengths.json',lengths)
    commit_json(entry/'native-cache.json',native)
    init=load(c['init'],c['init_sha256'])
    commit_json(entry/'init.json',init)
    # Carry only authenticated immutable inputs consumed from the historical release.
    old_source=Path(parent['source_root'])
    old_policy=Path(parent['source_policy'])
    parent['pins']={p:h for p,h in parent['pins'].items() if (not Path(p).is_relative_to(old_source)
        or Path(p).is_relative_to(parent['schedule']) or Path(p).is_relative_to(parent['parent_schedule']))
        and not Path(p).is_relative_to(old_policy)}
    commit_json(entry/'training.json',parent)
    frozen=freeze(repo,entry/'source',patterns=c['source_patterns'],io_workers=c['io_workers'])
    shutil.copyfile(repo/'configs/portable/operations/native-seal-container.json',entry/'native-seal-container.json')
    deploy=c['deployment'].copy()
    devices=parent['physical_devices']
    groups=[]
    sample=json.loads((Path(parent['release'])/'learning.jsonl').open().readline())
    for role in ('prepare','finite','formal'):
        spec=dict(phase='card_stage1', role=role, entry='/jev-entry',work='/jev-work',source=dict(
            root='/jev-entry/source',sha256=frozen['manifest_sha256']),sample=dict(
            path=str(Path(parent['audio_mirror_root'])/'audio'/(sample['audio_sha256']+'.audio')),sha256=sample['audio_sha256']),
            io_workers=c['io_workers'],**c['window'])
        commit_json(entry/(role+'.json'),dict(schema='jev-local-macros-v1',macros=dict(NATIVE_SEAL_CONTAINER_CONFIG=spec)))
        groups.append([dict(name=role,devices=[] if role=='prepare' else devices,
            collect=role+'/receipt.json',**c['resources'])])
    archive=root/'entry.tar'
    with tarfile.open(archive,'x') as tar:
        tar.add(entry,arcname='entry')
    deploy.update(archive=str(archive),entry_archive_sha256=file_hash(archive),
        entry_pins={str(p.relative_to(entry)):file_hash(p) for p in [*entry.glob('*.json'),entry/'source/FILES.json']},
        chains=[groups],device_group_lease=True,wait_for_lease=True,idle_devices=devices)
    commit_json(root/'operations-local.json',dict(schema='jev-local-macros-v1',macros=dict(NATIVE_SEAL_OPERATIONS_CONFIG=deploy)))


if __name__=='__main__':
    run(load_config(*sys.argv[1:])[0])
