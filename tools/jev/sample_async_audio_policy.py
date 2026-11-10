#!/usr/bin/env python3
"""Persist an actual AsyncLLM audio probe in the existing rollout format."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


async def run(config, output):
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.async_audio_rollout import AsyncAudioRollout, create_async_engine
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.task_schema import JevExample

    if config['audio_input_mode'] != 'actor_features': raise ValueError('exact feature input required')
    policy_id, files = checkpoint_identity(config['checkpoint'])
    identity = {'policy_id': policy_id, 'checkpoint_files': files, 'config': config,
                'manifest_sha256': file_hash(config['manifest'])}
    atomic_json(output/'identity.json', identity)
    wanted = config['example_ids']
    if len(set(wanted)) != len(wanted): raise ValueError('duplicate selection')
    lookup = {}
    for line in Path(config['manifest']).open():
        row = json.loads(line)
        if row['example_id'] in wanted:
            if row['example_id'] in lookup: raise ValueError('duplicate manifest example')
            lookup[row['example_id']] = JevExample.from_record(row)
    if set(lookup) != set(wanted): raise ValueError('missing example')
    examples = [lookup[uid] for uid in wanted]
    for ex in examples:
        if ex.split != 'train' or file_hash(ex.audio) != ex.metadata['audio_sha256']:
            raise ValueError('TRAIN audio identity required')
    processor = Qwen3ASRProcessor.from_pretrained(config['checkpoint'], local_files_only=True)
    audios = [load_audio(ex.audio) for ex in examples]
    atomic_json(output/'progress.json', {'phase':'async_engine_load', 'heartbeat_unix':time.time()})
    start = time.monotonic()
    engine = create_async_engine(config['checkpoint'], config['engine'])
    load_seconds = time.monotonic()-start
    try:
        rollout = AsyncAudioRollout(engine, processor, policy_id, config['engine']['max_model_len'])
        atomic_json(output/'progress.json', {'phase':'async_sampling', 'heartbeat_unix':time.time()})
        records = await rollout.sample(examples, audios, config['sampling'])
        with (output/'rollouts.jsonl').open('x') as handle:
            for record in records: handle.write(json.dumps(asdict(record))+'\n')
        atomic_json(output/'result.json', {'status':'passed','policy_id':policy_id,
            'prompts':len(examples),'rollouts':len(records),'rollouts_sha256':file_hash(output/'rollouts.jsonl'),
            'generation_budget':rollout.last_generation_budget,'request_ids':rollout.last_request_ids,
            'expanded_prompt_ids_match_actor':True,'engine_load_seconds':load_seconds,
            'scope':'actual vLLM AsyncLLM audio generation; not native verl Ray manager/server, actor probability parity, training or efficacy'})
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try: asyncio.run(run(json.loads(args.config.read_text()), args.output))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', {'exit_code':1,'error':repr(exc),'finished_unix':time.time()})
        raise
    atomic_json(args.output/'terminal.json', {'exit_code':0,'finished_unix':time.time()})


if __name__ == '__main__': main()
