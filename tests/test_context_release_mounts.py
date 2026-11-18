import ast
import json
from pathlib import Path

import pytest

from tools.jev import context_release_mounts as mod
from tools.jev import launch_context_rl_evaluation_v2 as launcher


def fixture_files(tmp_path, monkeypatch):
    release = tmp_path/'release';release.mkdir()
    target = '/home/project/review.json'
    source = '/mnt/cephfs/xingchao/evidence/review.json'
    publication = dict(pins={target:'expected'})
    (release/'publication-config.json').write_text(json.dumps(publication))
    (release/'MANIFEST.json').write_text(json.dumps(dict(files={'publication-config.json':'config'})))
    donor = tmp_path/'donor.json'
    donor.write_text(json.dumps(dict(argv=['docker','create','-v','/data/run:/data/run',
                                          '-v',source+':'+target+':ro'])))
    digests={str(release/'MANIFEST.json'):'manifest',str(release/'publication-config.json'):'config',
             source:'expected',target:'expected'}
    monkeypatch.setattr(mod,'sha',lambda p:digests[str(p)])
    return release, donor, digests, source, target


def test_mounts_cover_authenticated_publication_dependencies(tmp_path, monkeypatch):
    r,d,_,s,t = fixture_files(tmp_path,monkeypatch)
    assert mod.publication_mounts(r,'manifest',d)==[s+':'+t+':ro']


@pytest.mark.parametrize('problem',['missing','writable','duplicate','wrong_source','wrong_local'])
def test_rejects_missing_writable_and_changed_dependencies(tmp_path, monkeypatch,problem):
    r,d,digests,s,t = fixture_files(tmp_path,monkeypatch)
    value=json.loads(d.read_text())
    if problem=='missing':value['argv']=value['argv'][:-2]
    if problem=='writable':value['argv'][-1]=s+':'+t
    if problem=='duplicate':value['argv']+=value['argv'][-2:]
    if problem=='wrong_source':digests[s]='bad'
    if problem=='wrong_local':digests[t]='bad'
    d.write_text(json.dumps(value))
    with pytest.raises(ValueError):mod.publication_mounts(r,'manifest',d)


def test_formal_launcher_includes_readonly_dependencies(monkeypatch):
    mount='/mnt/cephfs/xingchao/evidence/a:/home/project/a:ro'
    monkeypatch.setattr(mod,'publication_mounts',lambda:[mount])
    command=launcher.command(dict(remote_root=launcher.PREFIX+'context-rl-eval-test'),
        dict(encoded_audio_cache=launcher.PREFIX+'context-selection-input-cache-v1',physical_devices=[7]))
    assert mount in [command[i+1] for i,x in enumerate(command) if x=='-v']
    assert command[command.index('--entrypoint')+1] == '/opt/multimodal-jev/runtime-entrypoint.sh'
    assert command[command.index(launcher.IMAGE)+1] == 'python'


def test_stage1_generated_runner_preflights_before_model():
    path=Path('.local/decision-objective-research/launch_context_stage1_selection_v2.py')
    tree=ast.parse(path.read_text())
    namespace={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='runner' for t in node.targets):
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    runner=namespace['runner']
    ast.parse(runner)
    assert runner.index("load(c['release']") < runner.index("for i,phase")
    assert 'assert len(rows)==9226' in runner
