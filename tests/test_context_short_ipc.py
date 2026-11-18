import ast
import json
import os
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from multimodal_jev.local_ipc import check_ipc_directory
from tools.jev.prepare_context_short_ipc import prepare, sha, transform
from tools.jev.prepare_context_shared_online import RUNNER


def test_actual_runner_role_environments_override_long_parent_paths():
    original = '/data/xingchao/multimodal-jev/simple-final-v1/context-online-shared-npu-v2'
    with pytest.raises(ValueError, match='125 bytes'):
        check_ipc_directory(original+'/service-cache')
    directories = []
    class FakePath(PurePosixPath):
        def mkdir(self, **kwargs):
            directories.append(str(self))
    tree = ast.parse(transform(RUNNER))
    loop = next(n for n in tree.body if isinstance(n, ast.For))
    inherited = dict(TMPDIR=original+'/cache', VLLM_RPC_BASE_PATH=original+'/cache',
                     ASCEND_RT_VISIBLE_DEVICES='wrong')
    namespace = dict(Path=FakePath, r=FakePath(original), envs={},
                     os=SimpleNamespace(environ=inherited))
    exec(compile(ast.Module(body=[loop], type_ignores=[]), '<runner-env>', 'exec'), namespace)
    for role in ('actor', 'service'):
        env = namespace['envs'][role]
        expected = '/jev-tmp/'+role
        assert env['VLLM_RPC_BASE_PATH'] == env['TMPDIR'] == expected
        assert check_ipc_directory(expected)['maximum_socket_path_bytes'] <= 107
        for key in ('HF_HOME', 'XDG_CACHE_HOME', 'TORCHINDUCTOR_CACHE_DIR',
                    'TRITON_CACHE_DIR', 'ASCEND_WORK_PATH', 'ASCEND_CACHE_PATH'):
            assert env[key] == expected
    assert namespace['envs']['service']['ASCEND_RT_VISIBLE_DEVICES'] == '3'
    assert 'ASCEND_RT_VISIBLE_DEVICES' not in namespace['envs']['actor']
    assert inherited['ASCEND_RT_VISIBLE_DEVICES'] == 'wrong'
    assert directories == ['/jev-tmp/actor', '/jev-tmp/service']


def fixture_package(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    remote = '/data/xingchao/multimodal-jev/simple-final-v1/old'
    (source/'run.py').write_text(RUNNER)
    (source/'source').mkdir()
    trainer = source/'source/trainer.py'
    trainer.write_text('# untouched numerical code\n')
    c = dict(engineering_prefix=True, training_dispatch_allowed=False, start_update=0,
             stop_update=2, source_policy='/immutable/policy',
             pins={remote+'/source/trainer.py': sha(trainer)}, output=remote+'/run')
    for name in ('config.json', 'service-config.json'):
        (source/name).write_text(json.dumps(c))
    manifest = dict(remote_root=remote, files={str(p.relative_to(source)): sha(p)
                    for p in source.rglob('*') if p.is_file()})
    (source/'PREPARED.json').write_text(json.dumps(manifest))
    return dict(source=source, source_manifest_sha256=sha(source/'PREPARED.json'),
                output=tmp_path/'out',
                remote_root='/data/xingchao/multimodal-jev/simple-final-v1/new')


def test_package_preserves_original_and_numerical_sources(tmp_path):
    args = fixture_package(tmp_path)
    before = {str(p.relative_to(args['source'])): sha(p) for p in args['source'].rglob('*') if p.is_file()}
    manifest = prepare(**args)
    assert before == {str(p.relative_to(args['source'])): sha(p) for p in args['source'].rglob('*') if p.is_file()}
    assert sha(args['output']/'source/trainer.py') == before['source/trainer.py']
    c = json.loads((args['output']/'config.json').read_text())
    assert c['source_policy'] == '/immutable/policy'
    assert c['output'] == args['remote_root']+'/run'
    assert c['pins'] == {args['remote_root']+'/source/trainer.py': before['source/trainer.py']}
    assert all(sha(args['output']/name) == digest for name, digest in manifest['files'].items())
    assert c['engineering_prefix'] and not c['training_dispatch_allowed']


@pytest.mark.parametrize('change', ['hash', 'extra', 'source', 'root', 'nested', 'guard'])
def test_rejects_unauthenticated_or_unsafe_packages(tmp_path, change):
    args = fixture_package(tmp_path)
    if change == 'hash': args['source_manifest_sha256'] = '0'*64
    if change == 'extra': (args['source']/'untracked').write_text('extra')
    if change == 'source': (args['source']/'source/trainer.py').write_text('changed')
    if change == 'root': args['remote_root'] = '/mnt/cephfs/run'
    if change == 'nested': args['output'] = args['source']/'nested'
    if change == 'guard':
        path = args['source']/'config.json'
        config = json.loads(path.read_text()); config['stop_update'] = 634
        path.write_text(json.dumps(config))
        mpath = args['source']/'PREPARED.json'
        m = json.loads(mpath.read_text()); m['files']['config.json'] = sha(path)
        mpath.write_text(json.dumps(m)); args['source_manifest_sha256'] = sha(mpath)
    with pytest.raises(ValueError): prepare(**args)
    assert not args['output'].exists()


def test_rejects_double_patch():
    with pytest.raises(ValueError): transform(transform(RUNNER))
