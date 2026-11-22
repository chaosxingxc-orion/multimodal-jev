"""Bind a terminal card policy and its frozen example bank to the evaluation queue."""
import json
from pathlib import Path
import shutil
import sys
import tarfile
import unicodedata

from multimodal_jev.baseline import file_hash
from multimodal_jev.intent_card_template import compile_intent_card_request
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.source_snapshot import freeze
from tools.reports.prepare_tpo_evaluation import queue_jobs
from tools.reports.prepare_tpo_windows import macros


def read(path):
    return json.loads(Path(path).read_bytes())


def pinned(path, sha):
    if file_hash(path) != sha:
        raise ValueError('changed input: '+str(path))
    return read(path)


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def compile_cards(c, pkg, entry):
    bank = pinned(pkg/'sampled-bank.json', c['bank_sha'])
    native = pinned(pkg/'native-cache.json', c['native_sha'])
    release = Path(c['release'])
    release_manifest = pinned(release/'MANIFEST.json', bank['pins']['clean_manifest'])
    task = pinned(release/'tasks.json',release_manifest['files']['tasks.json'])['intent']
    roster = {}
    for name in ('learning.jsonl','selection.jsonl'):
        if file_hash(release/name) != release_manifest['files'][name]:
            raise ValueError('source groups changed')
        for line in (release/name).open():
            row = json.loads(line)
            roster[row['id']] = (row['source_group'], row['audio_sha256'])
    identities = {}
    for source in c['annotations']:
        if file_hash(source['path']) != source['sha256']:
            raise ValueError('annotation identity changed')
        for line in Path(source['path']).open():
            ann = json.loads(line)
            for rec in ann['recordings']:
                identities['slurp:'+rec['file']] = (str(ann['slurp_id']),norm(ann['sentence']))
    test = c['test_roster']
    for row in pinned(test['path'],test['sha256']):
        roster[row['id']] = (None,row['audio_sha256'])
    cards = bank['cards']
    def keys(uid):
        group, digest = roster[uid]
        original, text = identities[uid]
        return (group,digest,original,text)
    excluded = {}
    for i,card in enumerate(cards):
        for j,example in enumerate(card['examples']):
            for k,v in enumerate(keys(example['id'])):
                excluded.setdefault((k,v),set()).add((i,j))
    template = read(c['template'])
    panels = {'native':native['templates']['native']['panel']}
    variants, assignments = {}, {}
    for uid in roster:
        denied = set().union(*(excluded.get((k,v),()) for k,v in enumerate(keys(uid)) if v is not None))
        indices = tuple(next((j for j in range(len(card['examples'])) if (i,j) not in denied),None)
                        for i,card in enumerate(cards))
        if indices not in variants:
            arm = 'support-'+str(len(variants))
            variants[indices] = arm
            panel = compile_intent_card_request('audio',task,bank,template,'definitions_rules_example',
                example_indices={card['id']:j for card,j in zip(cards,indices,strict=True)}).model_input()
            del panel['audio']
            panels[arm] = panel
        assignments[uid] = variants[indices]
    encoding = {k:native['encoding'][k] for k in ('audio_token_id','decision_token_id','max_tokens','prompt_layout')}
    witness = read(pkg/'prepare.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']['sample']
    commit_json(entry/'cards.json',dict(panels=panels,assignments=assignments,encoding=encoding,witness=witness))
    return dict(path='/jev-entry/cards.json',sha256=file_hash(entry/'cards.json')),encoding


def run(c):
    root = Path(c['output'])
    entry = root/'entry'
    entry.mkdir(parents=True,exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    old, pkg = Path(c['evaluation_base'])/'entry', Path(c['stage1_package'])/'entry'
    plan = pinned(old/'block-plan.json',c['plan_sha'])['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    view, enc = compile_cards(c,pkg,entry)
    policy = pinned(c['manifest'],c['manifest_sha'])
    if policy['policy_id'] != c['policy_id']:
        raise ValueError('incorrect terminal checkpoint')
    parent = read(pkg/'training.json')
    parent.update(source_policy=c['snapshot'],source_policy_id=policy['policy_id'],source_contract=policy['contract'],
                  audio_token_id=enc['audio_token_id'],view_execution=dict(max_input_tokens=enc['max_tokens']))
    commit_json(entry/'parent.json',parent)
    frozen = freeze(repo,entry/'source',patterns=c['source_patterns'],io_workers=c['io_workers'])
    source = dict(root='/jev-entry/source',sha256=frozen['manifest_sha256'])
    deploy = dict(c['deployment'])
    ports = plan['endpoints']
    for dataset in c['card_datasets']:
        ports[dataset]['card_view'] = view
    common = dict(source=source, io_workers=c['io_workers'], parent='/jev-entry/parent.json',
                  parent_sha=file_hash(entry/'parent.json'), batch=c['batch'], arms=['stage1'])
    queue_config = dict(c,sample_blocks=dict(devices=c['devices'],batches_per_block=plan['queue']['batches_per_block'],
                                              prefetch=plan['queue']['prefetch']))
    datasets = {k:dict(logical_partitions=v) for k,v in plan['queue']['partitions'].items()}
    inferred = {k:['stage1'] for k in datasets}
    prep,workers,queue = queue_jobs(queue_config,entry,common,datasets,ports,inferred,set(),deploy['image'])
    cal_cfg = pinned(old/'calibration-0.json',c['calibration_config_sha'])['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    cal = dict(cal_cfg['calibration'], policy_ids={'stage1':policy['policy_id']})
    if file_hash(old/'calibration-roster.json') != cal['roster']['sha256']:
        raise ValueError('calibration roster changed')
    shutil.copyfile(old/'calibration-roster.json',entry/'calibration-roster.json')
    def spec(name, value):
        commit_json(entry/(name+'.json'),macros('NATIVE_SEAL_CONTAINER_CONFIG',value))
    calibration, shards = [], []
    for i,device in enumerate(c['devices']):
        name = 'calibration-'+str(i)
        shards.append('/jev-work/'+name)
        spec(name,dict(common, phase='tpo_eval',endpoint=cal_cfg['endpoint'],calibration=cal,card_view=view,
                       devices=[device],shard=i,shards=len(c['devices']),output=shards[-1],prefetch=cal['prefetch']))
        calibration.append(dict(name=name,devices=[device],**c['eval_container']))
    spec('calibration-fit',dict(common,phase='tpo_score',calibration=cal,calibration_fit=True,
                                shards=shards,output='/jev-work/thresholds.json'))
    spec('score',dict(common,phase='tpo_score',queue=queue,endpoints=ports,inferred_arms=inferred,
                      probability_panels={k:dict(v,comparisons=[]) for k,v in plan['probability_panels'].items()},
                      thresholds='/jev-work/thresholds.json',calibration_roster_sha256=cal['roster']['sha256'],
                      output='/jev-work/metrics.json'))
    def cpu(name,path):
        return dict(name=name,devices=[],collect=path,**c['cache_container'])
    shutil.copyfile(repo/'configs/portable/operations/native-seal-container.json',entry/'native-seal-container.json')
    archive = root/'entry.tar'
    with tarfile.open(archive,'x') as tar:
        tar.add(entry,arcname='entry')
    paths={c['snapshot'],*(v['cache']['output'] for v in ports.values()),
           *(v['cache']['seal']['root'] for v in ports.values())}
    deploy.update(archive=str(archive),entry_archive_sha256=file_hash(archive),
                  entry_pins={str(p.relative_to(entry)):file_hash(p) for p in [*entry.glob('*.json'),entry/'source/FILES.json']},
                  chains=[[[prep],calibration,[cpu('calibration-fit','thresholds.json')],workers,[cpu('score','metrics.json')]]],
                  read_only_mounts=[p+':'+p for p in sorted(paths) if not Path(p).is_relative_to(deploy['shared_root'])],
                  idle_devices=c['devices'],wait_for_lease=True,device_group_lease=True)
    commit_json(root/'operations-local.json',macros('NATIVE_SEAL_OPERATIONS_CONFIG',deploy))


if __name__ == '__main__':
    run(load_config(*sys.argv[1:])[0])
