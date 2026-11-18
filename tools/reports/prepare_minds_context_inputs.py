"""Prepare full native-descriptor MINDS requests without opening gold annotations."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_decision_contract import render_context_request
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.slurp_training import content_hash


def definitions_for(protocol):
    candidates = protocol['candidates']
    if (protocol['schema'] != 'minds14-native-descriptor-transfer-v1'
            or protocol['descriptor_mode'] != 'native_label_underscore_to_space_only'
            or protocol['task_semantics'] != 'exclusive_event'
            or not protocol['input_preparation_allowed']
            or protocol['training_dispatch_allowed'] or protocol['evaluation_dispatch_allowed']
            or len(candidates) != 14 or len({c['id'] for c in candidates}) != 14
            or any(set(c) != {'id', 'definition'} or c['definition'] != c['id'].replace('_', ' ')
                   for c in candidates)):
        raise ValueError('Frozen native-label descriptor preparation protocol required')
    return dict(candidates=candidates, requirements=protocol['requirements'], semantics='exclusive_event')


def reconstruct(row, definitions):
    if set(row) != {'id', 'audio', 'audio_sha256', 'request_sha256', 'rendered_sha256'}:
        raise ValueError('Only label-free consumer fields accepted')
    if (not row['id'].startswith('minds14:')
            or Path(row['audio']).name != row['id'].split(':')[1] + '.wav'):
        raise ValueError('Opaque audio identity required')
    request = compile_candidates(row['audio'], **definitions)
    system, text = render_context_request(request)
    if (len(request.questions) != 14 or content_hash(request.model_input()) != row['request_sha256']
            or content_hash(dict(system=system, text=text)) != row['rendered_sha256']):
        raise ValueError('Complete fixed input representation differs')
    return request


def prepare(protocol_path, assets, verification_path, output):
    protocol_path, assets, verification_path, output = map(Path, (protocol_path, assets, verification_path, output))
    if output.exists():
        raise ValueError('New immutable output directory required')
    storage = check_paths({'input_package': output})
    protocol = json.loads(protocol_path.read_text())
    definitions = definitions_for(protocol)
    p = protocol['population']
    m = json.loads((assets / 'MANIFEST.json').read_text())
    v = json.loads(verification_path.read_text())
    if (file_hash(assets / 'MANIFEST.json') != p['assets_manifest_sha256']
            or file_hash(verification_path) != p['audio_verification_sha256']
            or v['manifest_sha256'] != p['assets_manifest_sha256']
            or Path(v['root']).resolve() != assets.resolve()
            or v['recordings'] != 8168 or p['recordings'] != 8168 or m['recordings'] != 8168
            or not all(v[k] for k in ('full_encoded_bytes_equal_audited_original',
                       'full_decoded_waveforms_equal_audit', 'exact_consumer_schema_no_labels_or_transcripts'))
            or file_hash(assets / 'inputs.jsonl') != m['inputs_sha256']
            or v['inputs_sha256'] != m['inputs_sha256']):
        raise ValueError('Independently verified complete audio assets required')
    rows = [json.loads(line) for line in (assets / 'inputs.jsonl').read_text().splitlines()]
    if len(rows) != 8168 or len({r['id'] for r in rows}) != 8168:
        raise ValueError('Complete unique target population required')
    output.mkdir(parents=True)
    commit_json(output / 'definitions.json', definitions)
    rendered = set()
    with (output / 'inputs.jsonl').open('x') as stream:
        for row in rows:
            if set(row) != {'id', 'audio', 'audio_sha256', 'sample_rate', 'channels', 'samples', 'waveform_sha256'}:
                raise ValueError('Unexpected asset fields')
            safe = 'audio/' + row['id'].removeprefix('minds14:') + '.wav'
            path = assets / row['audio']
            if (row['audio'] != safe or path.is_symlink()
                    or not path.resolve().is_relative_to((assets / 'audio').resolve())
                    or file_hash(path) != row['audio_sha256']):
                raise ValueError('Unsafe or changed audio asset')
            request = compile_candidates(str(path.resolve()), **definitions)
            system, text = render_context_request(request)
            value = dict(id=row['id'], audio=str(path.resolve()), audio_sha256=row['audio_sha256'],
                         request_sha256=content_hash(request.model_input()),
                         rendered_sha256=content_hash(dict(system=system, text=text)))
            reconstruct(value, definitions)
            rendered.add(value['rendered_sha256'])
            stream.write(json.dumps(value, sort_keys=True) + '\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())
    if len(rendered) != 1:
        raise ValueError('Every recording must use identical rendered context')
    result = dict(schema='minds14-native-descriptor-context-inputs-v1', recordings=8168, decisions=14,
                  protocol=dict(path=str(protocol_path.resolve()), sha256=file_hash(protocol_path)),
                  assets_manifest_sha256=p['assets_manifest_sha256'], audio_verification_sha256=file_hash(verification_path),
                  files={n: file_hash(output / n) for n in ('inputs.jsonl', 'definitions.json')},
                  candidate_ids=[c['id'] for c in definitions['candidates']],
                  distinct_rendered_contexts=len(rendered), rendered_sha256=next(iter(rendered)),
                  storage=storage, source_pins={str(Path(f).resolve()):file_hash(f) for f in (
                    __file__, 'src/multimodal_jev/context_candidate_task.py',
                    'src/multimodal_jev/context_decision_contract.py', 'src/multimodal_jev/slurp_training.py')},
                  target_annotations_opened=False, training_dispatch_allowed=False, evaluation_dispatch_allowed=False,
                  scope='Complete native-label descriptor requests only; no processor, inference, target tuning or semantic-definition certification.')
    commit_json(output / 'MANIFEST.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--verification', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = prepare(a.protocol, a.assets, a.verification, a.output)
    print(json.dumps(result, indent=2))
