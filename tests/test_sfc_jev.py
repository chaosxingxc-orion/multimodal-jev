import copy
import hashlib
import json

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc import parse_calls, source_identity
from multimodal_jev.sfc_consumption import SPLITS, split_record
from multimodal_jev.sfc_definitions import definitions
from multimodal_jev.sfc_jev import (export, native_from_request, reconstruct_calls,
                                    request_from_native, restore_bundle, split_bundle)


def function(name):
    return dict(name=name, description='A function defined by the supplied business task.',
                parameters=dict(type='object', properties={
                    'x': {'type': 'array', 'items': {'type': 'string'}},
                    'n': {'type': 'integer'}, 'missing': {'type': 'string'},
                    'extra': {'type': 'object'}, 'flag': {'type': 'boolean'}}, required=['x']))


def text(funcs=None):
    return ('Official instruction\n### Available Functions:\n' + json.dumps(funcs or [function('f'), function('g')])
            + '\n\n### Environment:\n{"date":"2026-11-10"}'
            + "\n\n### User's Audio Request:\n<audio>")


def record(split='train_id', stage=2):
    paths = [f'audio/final_data_audio/level_3_1/{split}/id_1_{i}.wav' for i in range(1, stage+1)]
    solution = json.dumps(['f(x=["a","b"],n=2,missing="NAN",extra={"a":[1,True,None]},flag=False)',
                           'g(x=[])', 'f(x=["a","b"],n=2,missing="NAN",extra={"a":[1,True,None]},flag=False)'])
    return dict(example_id='sfc:'+paths[-1],
        input=dict(messages=[{'role':'user','content':text()},
                             {'role':'assistant','content':'supplied prior assistant'},
                             {'role':'user','content':'follow-up <audio>'}][:2*stage-1],
                   audios=['/dataset/'+p for p in paths]),
        supervision=dict(official_solution=solution, calls=parse_calls(solution)),
        metadata={**source_identity(paths), 'relative_audios':paths,
                  'audio_sha256':[hashlib.sha256(p.encode()).hexdigest() for p in paths]})


def test_definition_parser_preserves_order_environment_and_types():
    parsed = definitions(record()['input'])
    assert parsed['functions'] == [function('f'), function('g')]
    assert parsed['environment'] == {'date': '2026-11-10'}
    assert parsed['instruction_prefix'] == 'Official instruction\n'


@pytest.mark.parametrize('change', ['duplicate_key', 'duplicate_name', 'unknown_required',
                                  'duplicate_required', 'bad_properties', 'missing_description',
                                  'ambiguous_section', 'bad_audio'])
def test_definition_parser_rejects_ambiguous_or_invalid_definitions(change):
    funcs = [function('f'), function('g')]
    if change == 'duplicate_name': funcs[1]['name'] = 'f'
    if change == 'unknown_required': funcs[0]['parameters']['required'] = ['undeclared']
    if change == 'duplicate_required': funcs[0]['parameters']['required'] = ['x', 'x']
    if change == 'bad_properties': funcs[0]['parameters']['properties']['x'] = 'string'
    if change == 'missing_description': del funcs[0]['description']
    content = text(funcs)
    if change == 'duplicate_key': content = content.replace('"date":', '"date":"duplicate","date":')
    if change == 'ambiguous_section': content += '\n\n### Environment:\n{}'
    if change == 'bad_audio': content += ' other'
    with pytest.raises(ValueError): definitions({'messages':[{'content':content}]})


def test_bundle_is_input_only_and_preserves_complete_native_record():
    row = record(); native, target = split_record(row)
    request, gold = split_bundle(native, target)
    assert native_from_request(request) == native
    assert restore_bundle(request, gold) == row
    assert request['input']['context']['messages'][1]['content'] == 'supplied prior assistant'
    assert len(request['input']['context']['audios']) == 2
    assert len(request['input']['decisions']) == 6
    assert gold['answers']['f00.present'] is True and gold['answers']['f00.absent'] is False
    args = gold['answers']['f00.calls']
    assert len(args) == 2 and args[0] == args[1]
    assert args[0]['extra'] == {'a':[1, True, None]}
    assert type(args[0]['n']) is int and args[0]['missing'] == 'NAN'
    # Gold multiplicity/values must not influence question inventory or model inputs.
    changed = copy.deepcopy(row)
    changed['supervision'] = dict(official_solution='[]', calls=[])
    n2, t2 = split_record(changed)
    r2, g2 = split_bundle(n2, t2)
    assert r2 == request
    assert g2['answers']['f00.calls'] == [] and g2['answers']['f00.absent'] is True
    assert request_from_native(native) == request


@pytest.mark.parametrize('change', ['duplicate_lost', 'array_order', 'type_changed', 'binding',
                                  'missing_answer', 'nonboolean', 'inconsistent', 'nonfinite', 'definition'])
def test_bundle_rejects_loss_and_tampering(change):
    request, gold = split_bundle(*split_record(record()))
    if change == 'duplicate_lost': gold['answers']['f00.calls'].pop()
    if change == 'array_order': gold['answers']['f00.calls'][0]['x'].reverse()
    if change == 'type_changed': gold['answers']['f00.calls'][0]['n'] = '2'
    if change == 'binding': request['input']['context']['audios'].reverse()
    if change == 'missing_answer': del gold['answers']['f01.calls']
    if change == 'nonboolean': gold['answers']['f00.present'] = 1
    if change == 'inconsistent': gold['answers']['f00.absent'] = True
    if change == 'nonfinite': gold['answers']['f00.calls'][0]['n'] = float('nan')
    if change == 'definition': request['input']['decisions'][0]['function']['description'] = 'changed'
    with pytest.raises(ValueError): restore_bundle(request, gold)


def native_release(tmp_path):
    root = tmp_path/'native'; root.mkdir(); pins = {}
    for split in SPLITS:
        pairs = [split_record(record(split, stage)) for stage in (1, 2)]
        pin = dict(requests=2, source_groups=1)
        for i, kind in enumerate(('requests','targets')):
            path = root/f'{split}.{kind}.jsonl'
            path.write_bytes(b''.join(canonical(pair[i]) for pair in pairs))
            pin[kind+'_sha256'] = file_hash(path)
        pins[split] = pin
    (root/'MANIFEST.json').write_bytes(canonical(dict(splits=pins)))
    return root


def test_export_complete_splits_roundtrip_and_immutable_release(tmp_path):
    source = native_release(tmp_path)
    path, receipt = export(source, tmp_path/'out')
    for split in SPLITS:
        req = list(map(json.loads, (path/f'{split}.requests.jsonl').read_text().splitlines()))
        gold = list(map(json.loads, (path/f'{split}.targets.jsonl').read_text().splitlines()))
        assert [restore_bundle(r,g) for r,g in zip(req,gold)] == [record(split,i) for i in (1,2)]
        assert receipt['splits'][split]['decisions'] == 12
    with pytest.raises(FileExistsError): export(source, tmp_path/'out')


def test_export_rejects_modified_source_and_cleans_stage(tmp_path):
    source = native_release(tmp_path)
    with (source/'train_id.requests.jsonl').open('a') as f: f.write('{}\n')
    with pytest.raises(ValueError, match='changed'): export(source, tmp_path/'out')
    assert not list((tmp_path/'out').iterdir())


def load_tool(name):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/'tools/jev'/f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_independent_verifier_catches_rebound_parameter_value(tmp_path):
    source = native_release(tmp_path)
    release, receipt = export(source, tmp_path/'out')
    verifier = load_tool('verify_sfc_jev_release')
    assert verifier.verify(release)['splits']['test_id']['bundles'] == 2
    path = release/'test_id.targets.jsonl'
    rows = list(map(json.loads,path.read_text().splitlines()))
    rows[0]['answers']['f00.calls'][0]['n'] = '2'
    path.write_bytes(b''.join(map(canonical,rows)))
    receipt['splits']['test_id']['targets_sha256'] = file_hash(path)
    (release/'MANIFEST.json').write_bytes(canonical(receipt))
    with pytest.raises(AssertionError): verifier.verify(release)


def test_definition_audit_detects_membership_and_type_without_dropping_calls(tmp_path):
    source = native_release(tmp_path)
    audit = load_tool('audit_sfc_jev_definition').audit
    report = audit(source)
    assert report['splits']['train_id']['counts']['identical_duplicate_call_records'] == 2
    assert report['splits']['train_id']['counts']['literal_NAN'] == 4
    assert not report['splits']['train_id']['first_30_schema_anomalies']
    row = record('train_id',1)
    row['supervision'] = dict(official_solution=json.dumps(['f(x="wrong",extra_arg=1)', 'unknown(x=[])']),
                              calls=parse_calls(json.dumps(['f(x="wrong",extra_arg=1)', 'unknown(x=[])'])))
    req,gold = split_record(row)
    manifest=json.loads((source/'MANIFEST.json').read_text())
    for kind,value in (('requests',req),('targets',gold)):
        path=source/f'train_id.{kind}.jsonl'
        lines=path.read_bytes().splitlines(keepends=True); lines[0]=canonical(value); path.write_bytes(b''.join(lines))
        manifest['splits']['train_id'][kind+'_sha256']=file_hash(path)
    (source/'MANIFEST.json').write_bytes(canonical(manifest))
    counts=audit(source)['splits']['train_id']['counts']
    assert counts['gold_function_not_in_input'] == 1
    assert counts['gold_parameter_not_in_input'] == 1
    assert counts['parameter_type_mismatch'] == 1
