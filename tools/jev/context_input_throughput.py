"""Offline Stage1 exposure/timing summary from independently collected ledgers.

No training changes. Token counts are unpadded input exposure, not FLOPs or
generated tokens. Rank wall times run concurrently, so use max, never sum.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('finite nonnegative measurement required')
    return value


def summarize_step(update, batch, ranks):
    rows = batch['records']
    if len(rows) != 128 or len(ranks) != 4:
        raise ValueError('complete global128 DP4 ledgers required')
    indices = batch['rank_indices']
    if (len(indices) != 4 or any(len(x) != 32 for x in indices)
            or sorted(i for group in indices for i in group) != list(range(128))):
        raise ValueError('exact rank partition required')
    real_tokens = slot_tokens = real_audio = slot_audio = real = 0
    for row in rows:
        weight = row['weight']
        tokens = row['input_tokens']
        receipt = row['audio_receipt']
        samples, rate = receipt['output_samples'], receipt['sample_rate']
        if (type(weight) not in (int, float) or weight not in (0, 1)
                or type(tokens) is not int or tokens <= 0
                or type(samples) is not int or samples <= 0
                or type(rate) is not int or rate <= 0):
            raise ValueError('binary real weights and positive input lengths required')
        real += int(weight)
        slot_tokens += tokens
        real_tokens += int(weight) * tokens
        slot_audio += samples / rate
        real_audio += weight * samples / rate
    if not real:
        raise ValueError('empty effective update')
    service = ranks[0]['input_service']
    for key in ('worker_seconds', 'wait_seconds', 'bind_seconds'):
        number(service[key])
    if type(service['cache_hits']) is not int or not 0 <= service['cache_hits'] <= 128:
        raise ValueError('invalid cache count')
    residuals = []
    for row in ranks:
        if (row['update'] != update or row['real_queries'] != real
                or type(row['checkpoint']) is not bool
                or row['checkpoint'] != ranks[0]['checkpoint']
                or row['input_service'] != service):
            raise ValueError('rank update/count/save/input timing disagreement')
        total, prep, actor = (number(row[k]) for k in
                             ('total_seconds', 'preparation_seconds', 'actor_seconds'))
        if total <= 0 or total + 1e-6 < prep + actor:
            raise ValueError('total must contain serial preparation and actor')
        if type(row['peak_allocated']) is not int or row['peak_allocated'] <= 0:
            raise ValueError('positive memory measurement required')
        residuals.append(max(0., total - prep - actor))
    return dict(update=update, checkpoint=ranks[0]['checkpoint'], real_queries=real,
                padding_slots=128-real, real_input_tokens=real_tokens,
                slot_input_tokens=slot_tokens, real_audio_seconds=real_audio,
                slot_audio_seconds=slot_audio,
                **{k: max(row[k] for row in ranks) for k in
                   ('total_seconds', 'actor_seconds', 'preparation_seconds', 'peak_allocated')},
                maximum_rank_other_seconds=max(residuals), input_service=service)


def aggregate(steps):
    if not steps:
        return None
    seconds = sum(row['total_seconds'] for row in steps)
    sums = {key: sum(row[key] for row in steps) for key in
            ('real_queries', 'padding_slots', 'real_input_tokens', 'slot_input_tokens',
             'real_audio_seconds', 'slot_audio_seconds')}
    def distribution(key):
        values = sorted(row[key] for row in steps)
        return dict(mean=sum(values)/len(values), p50=values[math.ceil(.5*len(values))-1],
                    p95=values[math.ceil(.95*len(values))-1])
    return dict(updates=len(steps), **sums, measured_step_seconds=seconds,
                real_queries_per_second=sums['real_queries']/seconds,
                real_input_tokens_per_second=sums['real_input_tokens']/seconds,
                real_audio_seconds_per_second=sums['real_audio_seconds']/seconds,
                phases={key: distribution(key) for key in
                        ('total_seconds', 'actor_seconds', 'preparation_seconds',
                         'maximum_rank_other_seconds')},
                foreground_wait_seconds=sum(row['input_service']['wait_seconds'] for row in steps),
                background_worker_seconds=sum(row['input_service']['worker_seconds'] for row in steps),
                cache_hits=sum(row['input_service']['cache_hits'] for row in steps),
                peak_allocated_bytes=max(row['peak_allocated'] for row in steps))


def run(collection, proof_path):
    root, proof_path = Path(collection), Path(proof_path)
    proof = json.loads(proof_path.read_text())
    state = proof['state']
    if state['Status'] != 'exited' or state['ExitCode'] != 0 or state['OOMKilled']:
        raise ValueError('successful independent terminal collection required')
    consumed = {}
    def read(name):
        raw = (root/name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if proof['files'].get(name) != digest:
            raise ValueError('unbound or changed collected ledger: '+name)
        consumed[name] = digest
        return json.loads(raw)
    config = read('config.json')
    start, stop = config['start_update'], config['stop_update']
    if (config['stage'] != 'instruction' or config['global_batch'] != 128
            or config['microbatch'] != 8 or config['physical_devices'] != [4,5,6,7]
            or type(start) is not int or type(stop) is not int or not 0 <= start < stop <= 634):
        raise ValueError('frozen Stage1 DP4 range required')
    for rank in range(4):
        result = read(f'run/rank-{rank}/result.json')
        if (result['passed'] is not True or result['start_update'] != start
                or result['stop_update'] != stop):
            raise ValueError('complete matching rank range required')
    steps = [summarize_step(step, read(f'run/batch-{step-1:04d}.json'),
             [read(f'run/rank-{rank}/update-{step:04d}.json') for rank in range(4)])
             for step in range(start+1, stop+1)]
    normal = [row for row in steps if row['update'] >= 11 and not row['checkpoint']]
    return dict(schema='context-input-throughput-v1', container_id=proof['container_id'],
                source_proof_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),
                consumed_files=consumed, steps=steps, all_steps=aggregate(steps),
                checkpoint_steps=aggregate([row for row in steps if row['checkpoint']]),
                normal_after_warmup=aggregate(normal),
                first32_normal=aggregate(normal[:32]) if len(normal)>=32 else None,
                last32_normal=aggregate(normal[-32:]) if len(normal)>=32 else None,
                scope='Offline collected Stage1 exposure only. Max rank step durations exclude initialization, '
                'collection and evaluation. Phase maxima are not additive. Background input overlaps '
                'training. Other time includes audits/synchronization/save and is not isolated I/O. '
                'Unpadded input tokens are not executed FLOPs or decoding throughput. No RL timing claim.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--collection', required=True)
    parser.add_argument('--proof', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args=parser.parse_args()
    report=run(args.collection, args.proof)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
