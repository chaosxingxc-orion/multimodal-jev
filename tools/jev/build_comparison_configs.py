#!/usr/bin/env python3
"""Write fixed comparison schedules without starting training or allocating devices."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.comparison_schedule import source_batches,comparison_configs
from multimodal_jev.task_schema import JevExample


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--initializer-identity',type=Path,required=True)
    parser.add_argument('--rl-template',type=Path,required=True)
    parser.add_argument('--sft-template',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--export-root',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('comparison configs must not be overwritten')
    initializer=json.loads(args.initializer_identity.read_text())
    rl=json.loads(args.rl_template.read_text());sft=json.loads(args.sft_template.read_text())
    rl['audio_input_mode']='actor_features';rl['actor']['response_logits_only']=False
    if file_hash(rl['manifest'])!=initializer['manifest_sha256']:
        raise ValueError('initializer manifest changed')
    ids=initializer['example_ids'];wanted=set(ids);examples=[]
    with Path(rl['manifest']).open() as handle:
        for line in handle:
            row=json.loads(line)
            if row['example_id'] in wanted:examples.append(JevExample.from_record(row))
    # The registered study uses the existing31-record,186-view initializer pool.
    if len(examples)!=186 or len({e.metadata['source_group'] for e in examples})!=31:
        raise ValueError('registered31-record initializer selection required')
    configs={}
    for seed in (17,29,43):
        batches=source_batches(examples,ids,seed,6)
        for name,config in comparison_configs(rl,sft,batches,seed,args.export_root/str(seed)).items():
            configs[f'{seed}-{name}.json']=config
    args.output.mkdir(parents=True)
    for name,config in configs.items():atomic_json(args.output/name,config)
    print(json.dumps({'configs':len(configs),'output':str(args.output),'scope':'static schedules only; runtime preflight and matched budget design still required'}))


if __name__=='__main__':main()
