#!/usr/bin/env python3
"""Persist real vLLM audio rollouts; executed only on an allocated device host."""
import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import time
from transformers import Qwen3ASRProcessor
from multimodal_jev.task_schema import JevExample
from multimodal_jev.audio_policy import load_audio
from multimodal_jev.rollout import VllmAudioRollout,validate_sampling
from multimodal_jev.baseline import file_hash,atomic_json
from multimodal_jev.policy_identity import checkpoint_identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    config=json.loads(args.config.read_text());validate_sampling(config['sampling'])
    if config.get('audio_input_mode','waveform') not in {'waveform','actor_features'}:
        raise ValueError('unsupported audio_input_mode')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'result.json').exists():raise FileExistsError('completed rollout must not be overwritten')
        try:
            checkpoint=Path(config['checkpoint'])
            # Hash the checkpoint, tokenizer and processor files as one policy identity.
            policy_id,files=checkpoint_identity(checkpoint)
            identity={'policy_id':policy_id,'checkpoint_files':files,'config':config,
                      'manifest_sha256':file_hash(config['manifest'])}
            if (args.output/'identity.json').exists() and json.loads((args.output/'identity.json').read_text())!=identity:
                raise ValueError('rollout identity mismatch')
            atomic_json(args.output/'identity.json',identity)
            wanted=set(config['example_ids']);examples=[]
            with Path(config['manifest']).open() as handle:
                for line in handle:
                    row=json.loads(line)
                    if row['example_id'] in wanted:examples.append(JevExample.from_record(row))
            if {ex.example_id for ex in examples}!=wanted or len(examples)!=len(wanted):
                raise ValueError('example selection missing or duplicated')
            processor=Qwen3ASRProcessor.from_pretrained(checkpoint,local_files_only=True)
            for ex in examples:
                if file_hash(ex.audio)!=ex.metadata['audio_sha256']:raise ValueError('audio hash changed')
            audios=[load_audio(ex.audio) for ex in examples]
            atomic_json(args.output/'progress.json',{'phase':'engine_load','heartbeat_unix':time.time()})
            rollout=VllmAudioRollout(str(checkpoint),processor,policy_id,config['engine'],
                                     config.get('audio_input_mode','waveform'))
            atomic_json(args.output/'progress.json',{'phase':'sampling','heartbeat_unix':time.time()})
            records=rollout.sample(examples,audios,config['sampling'],config.get('audio_input_mode','waveform'))
            with (args.output/'rollouts.jsonl').open('w') as handle:
                for record in records:handle.write(json.dumps(asdict(record))+'\n')
            atomic_json(args.output/'result.json',{'status':'passed','policy_id':policy_id,
                'prompts':len(examples),'rollouts':len(records),'rollouts_sha256':file_hash(args.output/'rollouts.jsonl'),
                'scope':'vLLM sampling only; actor parity gate and optimizer update not yet run'})
        except BaseException as exc:
            atomic_json(args.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()})
            raise
        atomic_json(args.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})


if __name__=='__main__':main()
