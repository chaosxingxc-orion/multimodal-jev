"""Finish full TRAIN input audit and three schedules after the original collector.

Local files only: no remote collection, model invocation or training dispatch.
The caller waits for the existing owner to finish; this module rejects missing
or changed terminal evidence and never substitutes an anchor-only population.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from tools.reports.verify_context_attribute_population_v2 import verify
from tools.reports.prepare_context_attribute_schedule_v2 import prepare
from tools.reports.context_attribute_training_contract import load_attribute_schedule,checkpoint_contract
from tools.reports.context_attribute_training_candidate import load_dataset,DATA_SHA
from tools.reports.context_attribute_actor_candidate import digest

ARMS=('direct','q_input','q_credit')


def verify_terminal_binding(transport,launch,collection):
    """Bind successful complete input output to the exact existing CPU job."""
    t=json.loads(Path(transport).read_text());l=json.loads(Path(launch).read_text())
    i=json.loads((Path(collection)/'inspect.json').read_text())
    if (t['container_id']!=l['container_id'] or i['Id']!=l['container_id']
            or t['image']!=l['image'] or i['Image']!=l['image']
            or t['state']!=i['State'] or t['state']['Status']!='exited'
            or t['state']['ExitCode']!=0 or t['state']['OOMKilled']
            or t['terminal']['exit_code']!=0 or t['terminal']['completed']!=40513
            or t['source_manifest_sha256']!=l['source_manifest_sha256']
            or i['HostConfig']['Devices'] or i['HostConfig']['NetworkMode']!='none'):
        raise ValueError('Exact successful full CPU input job required')
    return t


def finish(*,source_root,collection,package,transport,launch,anchors,dataset,
           original_schedule,original_sha256,base_config,output):
    output=Path(output)
    if output.exists():raise ValueError('New output required; preserve partial attempts')
    verify_terminal_binding(transport,launch,collection)
    audit=verify(collection,package,transport,anchors,source_root=source_root,role='learning')
    menu,inputs,targets,data=load_dataset(dataset)
    if file_hash(Path(package)/'candidate-manifest.json')!=DATA_SHA:
        raise ValueError('Audited population differs from target candidate')
    config=json.loads(Path(base_config).read_text())
    if config['release_sha256']!=data['release_sha256'] or config['schedule_sha256']!=original_sha256:
        raise ValueError('Same release and full exposure schedule required')
    population={r['id']:r for r in inputs}
    output.mkdir(parents=True)
    (output/'TRAIN-independent-v2.json').write_text(json.dumps(audit,indent=2)+'\n')
    arms={}
    for arm in ARMS:
        schedule=prepare(source_root=source_root,original_schedule=original_schedule,
            original_sha256=original_sha256,collection=collection,package=package,
            transport=transport,anchors=anchors,arm=arm,output=output/arm)
        sha=file_hash(output/arm/'MANIFEST.json')
        batches,loaded=load_attribute_schedule(output/arm,sha,original_schedule=original_schedule,
            population=population,menu=menu,arm=arm,source_root=source_root)
        if loaded!=schedule:raise ValueError('Published schedule differs on reload')
        contract=checkpoint_contract(config,menu,arm,loaded,source_root=source_root)
        (output/arm/'checkpoint-contract.json').write_text(json.dumps(contract,indent=2)+'\n')
        arms[arm]=dict(manifest_sha256=sha,batches_sha256=digest(batches),
            query_order_sha256=loaded['query_order_sha256'],lengths_sha256=loaded['lengths_sha256'],
            checkpoint_contract_sha256=file_hash(output/arm/'checkpoint-contract.json'))
    if (arms['q_input']['batches_sha256']!=arms['q_credit']['batches_sha256']
            or arms['q_input']['lengths_sha256']!=arms['q_credit']['lengths_sha256']
            or len({r['query_order_sha256'] for r in arms.values()})!=1
            or len({r['checkpoint_contract_sha256'] for r in arms.values()})!=3):
        raise ValueError('Matched input/exposure or distinct arm identity differs')
    result=dict(schema='context-attribute-full-input-publication-v2',training_dispatch_allowed=False,
        source_container_id=audit['container_id'],recordings=40513,updates=634,arms=arms,
        full_audit_sha256=file_hash(output/'TRAIN-independent-v2.json'),
        scope='Actual complete TRAIN inputs and derived schedules; no Qwen/NPU model acceptance or training dispatch',
        files={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file()})
    (output/'PUBLICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source-root','collection','package','transport','launch','anchors','dataset',
                'original-schedule','original-sha256','base-config','output'):
        p.add_argument('--'+key,required=True)
    print(json.dumps(finish(**vars(p.parse_args())),indent=2))
