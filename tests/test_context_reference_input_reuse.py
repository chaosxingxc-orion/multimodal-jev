from concurrent.futures import ThreadPoolExecutor
import json
import os

import pytest

from tools.reports import context_reference_input_reuse as m


def setup(monkeypatch):
    engine = m.InputReuseEngine(io_workers=2)
    calls = []
    def bind(proof, stream, **arguments):
        calls.append(1)
        if stream != 'valid':
            raise ValueError('invalid stream')
        return dict(rows=arguments['rows']), dict(checked=True)
    monkeypatch.setattr(m.inputs, 'bind', bind)
    args = dict(rows=[dict(id='sample', audio='/a.wav')], task=dict(rule='native'),
                release_sha256='r', processor_sha256='p')
    return engine, calls, args


def test_concurrent_pure_input_cache_and_result_isolation(monkeypatch):
    engine, calls, args = setup(monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fs = [pool.submit(engine.bind, {}, 'valid', **args) for _ in range(2)]
        values = [f.result() for f in fs]
    assert len(calls) == 1 and engine.input_store.hits == 1
    values[0][0]['rows'][0]['id'] = 'changed'
    assert engine.bind({}, 'valid', **args)[0]['rows'][0]['id'] == 'sample'


@pytest.mark.parametrize('changed', ['rows', 'task', 'release_sha256', 'processor_sha256', 'proof', 'code'])
def test_all_semantic_dependencies_invalidate(monkeypatch, changed):
    engine, calls, args = setup(monkeypatch)
    engine.bind({}, 'valid', **args)
    proof = {}
    if changed == 'proof':
        proof = dict(state='changed')
    elif changed == 'code':
        monkeypatch.setattr(engine, 'code_identity', lambda: {'changed': 'f'*64})
    elif changed == 'rows':
        args['rows'][0]['audio'] = '/other.wav'
    elif changed == 'task':
        args['task']['rule'] = 'other'
    else:
        args[changed] = 'new'
    engine.bind(proof, 'valid', **args)
    assert len(calls) == 2


def test_outer_loader_rehashes_same_stat_receipts(tmp_path, monkeypatch):
    engine, calls, args = setup(monkeypatch)
    stream = tmp_path/'stream'; stream.write_text('valid')
    proof = tmp_path/'proof.json'
    proof.write_text(json.dumps(dict(independent_receipts_sha256=m.file_hash(stream))))
    config = dict(input_proof=str(proof), input_proof_sha256=m.file_hash(proof),
                  input_receipts=str(stream), input_receipts_sha256=m.file_hash(stream))
    engine.load_inputs(config, **args)
    before = stream.stat(); stream.write_text('wrong')
    os.utime(stream, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match='evidence changed'):
        engine.load_inputs(config, **args)
    assert len(calls) == 1


def test_input_failure_is_not_retried(monkeypatch):
    engine, calls, args = setup(monkeypatch)
    for _ in range(2):
        with pytest.raises(ValueError, match='invalid stream'):
            engine.bind({}, 'wrong', **args)
    assert len(calls) == 1


def test_mutation_during_reconstruction_rejected(monkeypatch):
    engine, calls, args = setup(monkeypatch)
    original = m.inputs.bind
    def mutate(proof, stream, **arguments):
        result = original(proof, stream, **arguments)
        arguments['task']['rule'] = 'mutated'
        return result
    monkeypatch.setattr(m.inputs, 'bind', mutate)
    with pytest.raises(ValueError, match='changed during reconstruction'):
        engine.bind({}, 'valid', **args)


def test_full_replay_routes_only_explicit_components(tmp_path, monkeypatch):
    from tools.reports import audit_context_reference_dataset as original
    from multimodal_jev.cached_context_preparation import processor_cache_identity
    from multimodal_jev.context_reference_dataset import digest
    engine = m.InputReuseEngine(io_workers=2)
    before = dict(original.__dict__)
    build = dict(source_policy_id='policy', source_contract={}, base_id='base',
                 release='release', release_sha256='release-hash', stage1_commit='commit',
                 pins={'commit':'commit-hash'}, macro_client=dict(runtime_identity={},
                 max_queries=128, rollout_queries=32, replicas=[{}, {}]))
    bindings = {'sample':{'policy_id':'policy'}}
    population = {k:digest(v) for k,v in bindings.items()}
    identity = dict(schema='jev-reference-dataset-v1', policy_id='policy', base_id='base',
                    contract={}, runtime_identity={}, release_sha256='release-hash',
                    candidate_ids=['a'], population_sha256=digest(population))
    result = dict(identity=identity, population=population, recordings=1, manifest_sha256='manifest')
    paths = dict(build_config=tmp_path/'build.json', stage1_commit=tmp_path/'commit.json',
                 input_receipts=tmp_path/'receipts', build_output=tmp_path/'out',
                 stage1_policy=tmp_path/'policy')
    paths['build_output'].mkdir(); (paths['build_output']/'dataset').mkdir()
    paths['stage1_policy'].mkdir()
    (paths['stage1_policy']/'MANIFEST.json').write_text(json.dumps({'files':{}}))
    paths['build_config'].write_text(json.dumps(build))
    paths['stage1_commit'].write_text(json.dumps(dict(joint_policy_manifest_sha256='manifest')))
    (paths['build_output']/'result.json').write_text(json.dumps(result))
    (paths['build_output']/'terminal.json').write_text(json.dumps({'exit_code':0}))
    (paths['build_output']/'dataset/records.json').write_text('{}')
    paths['input_receipts'].write_text('receipts')
    c = dict({k:str(v) for k,v in paths.items()}, build_config_sha256='build-hash',
             stage1_commit_sha256='commit-hash', input_receipts_sha256='receipt-hash',
             input_proof_sha256='proof-hash', journal='journal', output='output')
    hashes={str(paths[k]):c[k+'_sha256'] for k in ('build_config','stage1_commit','input_receipts')}
    hashes[str(paths['stage1_policy']/'MANIFEST.json')]='manifest'
    monkeypatch.setattr(original, 'file_hash', lambda p:hashes[str(p)])
    monkeypatch.setattr(original, 'read_execution_bundle', lambda *a,**kw:{'policy_id':'policy'})
    monkeypatch.setattr(original, 'validate_endpoint', lambda *a:None)
    monkeypatch.setattr(original, 'load', lambda *a,**kw:([{'id':'sample'}],
        {'schema':{'candidates':[{'id':'a','description':'A'}]}},{}))
    monkeypatch.setattr(engine, 'load_inputs', lambda *a,**kw:({}, {'input':'audit'}))
    monkeypatch.setattr(m.inputs, 'bindings_from_records', lambda *a,**kw:bindings)
    monkeypatch.setattr(engine, 'audit', lambda *a,**kw:{'status':'leaf-proof'})
    import multimodal_jev.cached_context_preparation as processor
    monkeypatch.setattr(processor, 'processor_cache_identity', lambda *a:'processor')
    patched = dict(original.__dict__)
    proof = engine.replay(c, {'source':'test'})
    assert proof['status']=='leaf-proof' and proof['input_audit']=={'input':'audit'}
    assert [p['phase'] for p in engine.phases] == [
        'endpoint_bundle','release_population','processor','independent_inputs','record_bindings','journal']
    assert all(original.__dict__[k] is v for k,v in patched.items())
    assert original.run is before['run']
