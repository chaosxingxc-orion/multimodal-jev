"""Candidate attribute schedule and resume identity; no training dispatch.

Reuses the frozen full-population order, length planner and checkpoint format.
Publication requires the complete independent processor audit, not anchors.
"""
import copy
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_training_schedule import load_schedule
from multimodal_jev.span_length_groups import length_group_indices
from tools.reports.context_attribute_actor_candidate import generate, digest
from tools.reports.verify_context_attribute_population import verify

DEPENDENCIES = {
    'src/multimodal_jev/context_training_schedule.py': '3cc6cf54e45e00facfde1e89b20091419a33ad9c06e479ebd3a93aba284684f0',
    'src/multimodal_jev/span_length_groups.py': 'f3e09f884e8b4aa1f3a304cfc460a31935e11e4c12893f9da6a275403b762329',
    'src/multimodal_jev/context_training_checkpoint.py': '73633e00c3ea7d822c68e0450cc4f7326efde875d4888dea9d366a2caa375130',
}


def authenticate(root):
    for name, expected in DEPENDENCIES.items():
        if file_hash(Path(root)/name) != expected:
            raise ValueError('Frozen training dependency changed: '+name)


def regroup(batches, records, inputs, *, arm):
    """Pure transformation; caller authenticates source schedule and audit."""
    if arm not in ('direct', 'q_input', 'q_credit'):
        raise ValueError('Unknown comparison arm')
    assets = {r['id']: r for r in inputs}
    if not assets or len(assets) != len(inputs):
        raise ValueError('Distinct complete input population required')
    lengths = {}
    input_arm = 'direct' if arm == 'direct' else 'q_input'
    for record in records:
        uid = record['id']
        if uid not in assets or uid in lengths or record['audio_sha256'] != assets[uid]['audio_sha256']:
            raise ValueError('Processor population/audio differs')
        rows = record['records']
        if [r['arm'] for r in rows] != ['direct', 'q_input']:
            raise ValueError('Exact processor arm pair required')
        row = next(r for r in rows if r['arm'] == input_arm)
        shape = row['tensors']['input_features']['shape']
        tokens = row['tokens']
        if (type(tokens) is not int or not 0 < tokens <= 4096 or len(shape) != 3
                or shape[0] != 1 or any(type(x) is not int or x < 1 for x in shape)):
            raise ValueError('Bounded actual token/feature shape required')
        lengths[uid] = [tokens, shape[-1]]
    if lengths.keys() != assets.keys():
        raise ValueError('Incomplete processor population')
    result = copy.deepcopy(batches)
    for batch in result:
        if len(batch['slots']) != 128:
            raise ValueError('Frozen global128 required')
        batch['rank_indices'] = length_group_indices(
            [lengths[s['id']] for s in batch['slots']], microbatch=8,
            world_size=4, order='length_grouped')
    return result, dict(input_arm=input_arm, lengths_sha256=digest(lengths),
        query_order_sha256=digest([dict(epoch=b['epoch'], index=b['index'], slots=b['slots']) for b in batches]),
        batches_sha256=digest(result))


def prepare_schedule(*, source_root, original_schedule, original_sha256,
                     collection, package, transport, anchors, arm, output):
    """Publish only after full real TRAIN processor evidence is validated."""
    authenticate(source_root)
    output = Path(output)
    if output.exists():
        raise ValueError('Never replace an existing candidate schedule')
    audit = verify(collection, package, transport, anchors)
    package = Path(package)
    candidate = json.loads((package/'candidate-manifest.json').read_text())
    inputs = [json.loads(s) for s in (package/'inputs.jsonl').read_text().splitlines()]
    menu = json.loads((package/'catalog.json').read_text())
    _, actor = generate(menu, arm, source_root=source_root)
    batches, original = load_schedule(original_schedule, original_sha256,
        release_sha256=candidate['release_sha256'], population={r['id']: r for r in inputs})
    if len(batches) != 634 or len(inputs) != 40513:
        raise ValueError('Full two-epoch TRAIN schedule required')
    with (Path(collection)/'run/records.jsonl').open() as stream:
        derived, identity = regroup(batches, map(json.loads, stream), inputs, arm=arm)
    manifest = dict(schema='context-attribute-training-schedule-v1', **identity,
        actor_contract_id=actor['contract_id'], catalog_sha256=actor['catalog_sha256'],
        source_schedule_sha256=original_sha256, release_sha256=candidate['release_sha256'],
        candidate_manifest_sha256=file_hash(package/'candidate-manifest.json'),
        processor_records_sha256=audit['records_sha256'],
        processor_transport_sha256=audit['transport_sha256'],
        recordings=40513, updates=634, global_batch=128, world_size=4, microbatch=8,
        accumulation=4, training_dispatch_allowed=False)
    output.mkdir(parents=True)
    (output/'batches.json').write_text(json.dumps(derived, separators=(',', ':'))+'\n')
    manifest['batches_file_sha256'] = file_hash(output/'batches.json')
    (output/'MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def load_attribute_schedule(root, manifest_sha256, *, original_schedule, population,
                            menu, arm, source_root):
    """Load a sealed derived schedule without trusting a caller's in-memory dict."""
    authenticate(source_root)
    root=Path(root)
    if file_hash(root/'MANIFEST.json') != manifest_sha256:
        raise ValueError('Attribute schedule identity changed')
    manifest=json.loads((root/'MANIFEST.json').read_text())
    _,actor=generate(menu,arm,source_root=source_root)
    if (manifest['schema'] != 'context-attribute-training-schedule-v1'
            or manifest['actor_contract_id'] != actor['contract_id']
            or manifest['catalog_sha256'] != actor['catalog_sha256']
            or manifest['input_arm'] != ('direct' if arm=='direct' else 'q_input')
            or file_hash(root/'batches.json') != manifest['batches_file_sha256']):
        raise ValueError('Attribute schedule arm or bytes differ')
    original,_=load_schedule(original_schedule,manifest['source_schedule_sha256'],
        release_sha256=manifest['release_sha256'],population=population)
    derived=json.loads((root/'batches.json').read_text())
    if len(original)!=634 or len(population)!=40513 or len(derived)!=634 or digest(derived)!=manifest['batches_sha256']:
        raise ValueError('Complete attribute schedule required')
    order=[dict(epoch=b['epoch'],index=b['index'],slots=b['slots']) for b in original]
    if digest(order)!=manifest['query_order_sha256']:
        raise ValueError('Source query order changed')
    for before,after in zip(original,derived,strict=True):
        if ({k:v for k,v in before.items() if k!='rank_indices'} !=
                {k:v for k,v in after.items() if k!='rank_indices'}
                or len(after['rank_indices'])!=4 or any(len(r)!=32 for r in after['rank_indices'])
                or any(type(i) is not int for r in after['rank_indices'] for i in r)
                or sorted(i for r in after['rank_indices'] for i in r)!=list(range(128))):
            raise ValueError('Exposure or DP4 slot assignment changed')
    return derived,manifest


def checkpoint_contract(config, menu, arm, schedule_manifest, *, source_root):
    """Bind reward/arm/data/arithmetic to the original exact-resume validator."""
    authenticate(source_root)
    _, actor = generate(menu, arm, source_root=source_root)
    schedule = copy.deepcopy(schedule_manifest)
    if (config['source_policy_id'] != actor['reference_policy_id']
            or config['algorithm'] not in ('grpo_v1', 'rloo_v1')
            or config['stage'] != 'task' or config['global_batch'] != 128 or config['microbatch'] != 8
            or schedule['schema'] != 'context-attribute-training-schedule-v1'
            or schedule['actor_contract_id'] != actor['contract_id']
            or schedule['catalog_sha256'] != actor['catalog_sha256']
            or schedule['release_sha256'] != config['release_sha256']
            or schedule['input_arm'] != ('direct' if arm == 'direct' else 'q_input')
            or [schedule[k] for k in ('recordings','updates','global_batch','world_size','microbatch','accumulation')]
            != [40513,634,128,4,8,4]):
        raise ValueError('Exact attribute training configuration required')
    design = json.loads((Path(source_root)/'configs/experiments/context-native-attribute-comparison-design-v1.json').read_text())
    result = {k:copy.deepcopy(config[k]) for k in ('base_id','source_policy_id','source_contract',
        'runtime_profile','release_sha256','global_batch','microbatch','stage','algorithm','behavior_bounds')}
    result.update(schema='context-attribute-checkpoint-contract-v1', attribute=actor,
        schedule=schedule, optimizer=dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.),
        objective={k:copy.deepcopy(design[k]) for k in ('policy_loss','auxiliary','process_auxiliary','kl')},
        reward=design['arms'][arm]['reward'], grad_clip=1., world_size=4)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source-root','original-schedule','original-sha256','collection','package','transport','anchors','output'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--arm',required=True,choices=['direct','q_input','q_credit'])
    print(json.dumps(prepare_schedule(**vars(parser.parse_args())),indent=2))
