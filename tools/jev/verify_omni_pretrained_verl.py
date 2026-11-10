#!/usr/bin/env python3
"""Offline acceptance of the complete frozen CPU verl comparison; no model load."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, *, positive=False):
    return type(value) in (float, int) and math.isfinite(value) and (not positive or value > 0)


def check_probabilities(reference, microbatches, comparison, result):
    """Recompute every comparison from six ordered two-token raw responses."""
    require(len(microbatches) == 2, 'exact two microbatch conditions required')
    for values in [reference] + [m['log_probs'] for m in microbatches]:
        require(len(values) == 6 and all(len(row) == 2 for row in values), 'complete six-by-two probabilities required')
        require(all(finite(v) and v <= 0 for row in values for v in row), 'finite nonpositive log probabilities required')
    def error(a, b):
        return max(abs(x-y) for ra, rb in zip(a, b, strict=True) for x, y in zip(ra, rb, strict=True))
    unpadded = [error(m['log_probs'], reference) for m in microbatches]
    between = error(microbatches[0]['log_probs'], microbatches[1]['log_probs'])
    require(max(unpadded + [between]) <= .1, 'predeclared 0.1 threshold failed')
    for m, actual in zip(microbatches, unpadded, strict=True):
        require(finite(m['seconds'], positive=True), 'positive microbatch timing required')
        require(finite(m['max_error_from_unpadded']) and abs(m['max_error_from_unpadded']-actual) <= 1e-6,
                'microbatch reported error differs')
    for report in (comparison, result):
        require(report['atol'] == .1, 'original tolerance required')
        errors = report['errors']
        require(len(errors['unpadded']) == 2 and finite(errors['microbatch'])
                and abs(errors['microbatch']-between) <= 1e-6
                and all(finite(v) and abs(v-a) <= 1e-6 for v, a in zip(errors['unpadded'], unpadded, strict=True)),
                'reported comparison differs from raw probabilities')
    require(comparison['passed'] is True, 'successful comparison required')
    return dict(microbatch=between, unpadded=unpadded)


def check_compact_contract(argv, identity, sources, original, microbatches):
    """Bind the opt-in actor to the fully accepted candidate and actual calls."""
    flags = ('--compact-keys-root', '--compact-keys-inspection', '--compact-keys-inspection-hash')
    enabled = any(flag in argv for flag in flags)
    if not enabled:
        require(identity.get('compact_keys_enabled', False) is False
                and identity.get('compact_candidate') is None
                and all(m.get('compact_attention_records') is None for m in microbatches),
                'compact execution cannot be accepted as the default actor')
        return None
    require(all(argv.count(flag) == 1 and argv.index(flag)+1 < len(argv) for flag in flags),
            'complete unique compact candidate arguments required')
    require(identity.get('compact_keys_enabled') is True, 'compact actor flag required')
    from multimodal_jev.omni_compact_acceptance import require_candidate
    candidate = require_candidate(*(argv[argv.index(flag)+1] for flag in flags))
    require(identity.get('compact_candidate') == candidate, 'actual accepted compact candidate binding differs')
    require(sources.get('src/multimodal_jev/omni_compact_attention.py') == candidate['adapter_sha256'],
            'actual actor adapter differs from accepted candidate')
    order = [2, 0, 5, 1, 3, 4]
    lengths = [original['responses'][i]['prompt_tokens'] + 2 for i in order]
    require(all(type(n) is int and 0 < n <= 1230 for n in lengths) and max(lengths) == 1230,
            'original full prompt lengths required')
    for size, micro in zip((1, 2), microbatches, strict=True):
        expected = []
        for offset in range(0, 6, size):
            keys = lengths[offset:offset+size]
            for layer in range(48):
                event = dict(layer=layer, mode='unchanged_no_mask' if all(n == 1230 for n in keys)
                             else 'compact_masked_keys', query_length=1230, key_lengths=keys)
                if event['mode'] == 'compact_masked_keys': event['original_key_length'] = 1230
                expected.append(event)
        require(micro.get('compact_attention_records') == expected,
                'actual compact attention layers or microbatch row coverage differs')
    return candidate


def accept(root, launch_hash, inspection, inspection_hash):
    require(sha(root/'launch.json') == launch_hash and sha(inspection) == inspection_hash, 'registered evidence changed')
    launch = read(root/'launch.json'); intent = read(root/'intent.json'); state = read(inspection)
    require(sha(root/'intent.json') == launch['intent_sha256'], 'launch intent changed')
    argv = intent['argv']; image = intent['image']; host = state['HostConfig']
    require(state['Id'] == launch['container_id'] and state['Image'] == image == launch['image']
            and state['State']['Status'] == 'exited' and state['State']['ExitCode'] == 0
            and not state['State'].get('OOMKilled') and not host.get('Devices') and not host.get('Privileged')
            and host['NetworkMode'] == 'none' and host['Memory'] == 160*1024**3
            and host['NanoCpus'] == 16000000000, 'exact successful bounded CPU container required')
    require(intent['automatic_retry'] is False and intent['wall_seconds'] == 14400
            and intent['atol'] == .1 and intent['micro_batch_sizes'] == [1, 2]
            and intent['weight_updates'] == 0 and intent['source_commit'] == launch['source_commit'],
            'registered comparison contract differs')
    require(state['Config']['Cmd'] == argv[argv.index(image)+1:] and '--preflight-only' not in argv,
            'executed full comparison command differs')
    def argument(name):
        require(argv.count(name) == 1, 'unique argument required: '+name)
        return argv[argv.index(name)+1]
    require(argument('--output') == str(root/'run') and '14400s' in argv, 'bounded output binding differs')
    source = root/'source'
    require({str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()} == set(intent['source_files']),
            'complete source roster differs')
    for name, pin in intent['source_files'].items():
        require(sha(source/name) == pin, 'source snapshot changed: '+name)
    prior = Path(argument('--prior')); acceptance = Path(argument('--acceptance'))
    require(sha(acceptance) == argument('--acceptance-hash') == intent['acceptance_sha256'], 'prior acceptance changed')
    accepted = read(acceptance); original = read(prior/'run/result.json'); previous = read(prior/'run/identity.json')
    require(accepted['schema'] == 'omni-pretrained-cpu-forward-acceptance-v1' and accepted['forwards'] == 6
            and accepted['weight_updates'] == 0 and accepted['result_sha256'] == sha(prior/'run/result.json')
            and accepted['launch_sha256'] == sha(prior/'launch.json')
            and accepted['intent_sha256'] == sha(prior/'intent.json')
            and original['identity_sha256'] == sha(prior/'run/identity.json'), 'accepted original forward required')
    require(sha(root/'inspection.json') == argument('--inspection-hash') == intent['inspection_sha256'],
            'captured original inspection changed')
    require(argument('--inspection') == str(root/'inspection.json'), 'original inspection path differs')
    original_state = read(root/'inspection.json')
    require(original_state['Id'] == accepted['container_id'] and original_state['Image'] == image
            and original_state['State']['Status'] == 'exited' and original_state['State']['ExitCode'] == 0,
            'original successful container binding differs')
    run = root/'run'; identity = read(run/'identity.json'); result = read(run/'result.json')
    order = [2, 0, 5, 1, 3, 4]
    require([r['audio_count'] for r in original['responses']] == [1, 1, 2, 1, 1, 2], 'original audio roster differs')
    sources = {k:v for k,v in intent['source_files'].items()
               if k.startswith(('src/multimodal_jev/', 'tools/jev/')) and k.count('/') == 2 and k.endswith('.py')}
    require(identity['order'] == order and identity['keys'] == [previous['selected_keys'][i] for i in order]
            and identity['response_ids'] == [3849, 151645]
            and all(r['response_token_ids'] == identity['response_ids'] for r in original['responses'])
            and identity['config'] == previous['config'] == read(prior/'config.json')
            and identity['model_files'] == previous['model_files'] and identity['source_files'] == sources
            and identity['prior_acceptance_sha256'] == intent['acceptance_sha256']
            and identity['inspection_sha256'] == intent['inspection_sha256']
            and identity['targets_read'] is False and identity['synthetic_old_log_probs'] is True,
            'complete original input/model/source binding differs')
    preflight = root/'preflight-evidence.json'
    require(sha(preflight) == intent['preflight_evidence_sha256'], 'preflight evidence changed')
    require(read(preflight)['result'] == dict(status='passed', preflight_only=True, model_loaded=False,
            tensor_shape=[6,1230], order=order, weight_updates=0), 'complete real-processor preflight required')
    loading = read(run/'loading-info.json')
    require(all(k in loading for k in ('missing_keys', 'mismatched_keys', 'error_msgs', 'unexpected_keys'))
            and not any(loading.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs'))
            and all(k.startswith(('talker.', 'code2wav.')) for k in loading.get('unexpected_keys', [])),
            'complete pretrained Thinker load required')
    require(result['status'] == 'passed' and result['preflight_only'] is False and result['model_loaded'] is True
            and result['weight_updates'] == 0 and read(run/'terminal.json')['exit_code'] == 0
            and finite(result['elapsed_seconds'], positive=True) and result['elapsed_seconds'] <= 14400
            and type(result['peak_rss_kib']) is int and result['peak_rss_kib'] > 0, 'complete timed frozen run required')
    require(sorted(p.name for p in run.glob('microbatch-*.json')) == ['microbatch-1.json', 'microbatch-2.json'],
            'exact microbatch artifact roster required')
    microbatches = [read(run/f'microbatch-{i}.json') for i in (1, 2)]
    candidate = check_compact_contract(argv, identity, sources, original, microbatches)
    if candidate is not None:
        require(set(host.get('Binds', [])) == {argv[i+1] for i,v in enumerate(argv) if v == '-v'}
                and all(argv[i+1] in state['Config'].get('Env', []) for i,v in enumerate(argv) if v == '-e'),
                'compact actor mounts or environment differ')
    require(sum(m['seconds'] for m in microbatches) <= result['elapsed_seconds'], 'inconsistent elapsed time')
    errors = check_probabilities([original['responses'][i]['response_log_probs'] for i in order],
                                microbatches, read(run/'comparison.json'), result)
    return dict(schema='omni-pretrained-verl-comparison-acceptance-v1', container_id=state['Id'], image=image,
        source_commit=launch['source_commit'], launch_sha256=launch_hash, inspection_sha256=inspection_hash,
        artifact_sha256={p.name:sha(p) for p in run.glob('*.json')}, recomputed_errors=errors,
        atol=.1, forwards_per_condition=6, micro_batch_sizes=[1,2], weight_updates=0,
        compact_keys_enabled=candidate is not None, compact_candidate=candidate,
        elapsed_seconds=result['elapsed_seconds'], peak_rss_kib=result['peak_rss_kib'], verifier_sha256=sha(__file__),
        scope='Offline artifact/source/container acceptance of CPU pretrained verl comparison. '
              'Controlled false+EOS responses, explicit CPU adapters; no NPU/vLLM parity or task quality.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for key in ('root','inspection','output'):parser.add_argument('--'+key,type=Path,required=True)
    for key in ('launch-hash','inspection-hash'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    value=accept(args.root,args.launch_hash,args.inspection,args.inspection_hash)
    with args.output.open('x') as handle:json.dump(value,handle,indent=2);handle.write('\n')
