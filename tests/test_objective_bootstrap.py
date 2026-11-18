import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import os
import sys

import pytest

from tools.reports.context_objective_bootstrap import MODULES,OBJECTIVE,verify_files

ROOT=Path(__file__).resolve().parents[1]


def setup(tmp_path):
    source=tmp_path/'source';extension=tmp_path/'objective-extension'
    source.mkdir();extension.mkdir()
    # Tiny regular forward package fixture. Subprocess installation separately
    # exercises real plugin modules against the full inherited source tree.
    (source/'forward.py').write_text('VALUE = 1\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=tmp_path/'RUNTIME-SOURCES.json'
    manifest.write_text(json.dumps({'forward.py':sha(source/'forward.py')}))
    for name in [p for _,p in MODULES]+[OBJECTIVE]:
        dest=extension/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest)
    shutil.copy2(ROOT/'tools/reports/context_objective_bootstrap.py',extension/'bootstrap.py')
    files={str(p.relative_to(extension)):sha(p) for p in extension.rglob('*') if p.is_file()}
    return dict(source_root=str(source),dual_runtime_manifest=str(manifest),
        dual_runtime_identity=dict(source_manifest_sha256=sha(manifest)),
        objective_extension=dict(root=str(extension),files=files,forward_manifest_sha256=sha(manifest)),
        objective_plugin=dict(path=str(extension/OBJECTIVE),sha256=files[OBJECTIVE],options={}),
        pins={str(extension/n):h for n,h in files.items()})


def test_disjoint_extension_leaves_forward_identity_unchanged(tmp_path):
    c=setup(tmp_path)
    root,source=verify_files(c)
    assert root!=source and c['objective_extension']['forward_manifest_sha256']==c['dual_runtime_identity']['source_manifest_sha256']


@pytest.mark.parametrize('fault',['overlap','source','extra','missing','pin','objective','manifest','symlink'])
def test_bootstrap_rejects_identity_and_shadow_changes(tmp_path,fault):
    c=setup(tmp_path);root=Path(c['objective_extension']['root']);source=Path(c['source_root'])
    if fault=='overlap':c['objective_extension']['root']=str(source)
    elif fault=='source':(source/'forward.py').write_text('VALUE = 2\n')
    elif fault=='extra':(root/'unapproved.py').write_text('pass\n')
    elif fault=='missing':(root/OBJECTIVE).unlink()
    elif fault=='pin':c['pins'][str(root/OBJECTIVE)]='0'*64
    elif fault=='objective':c['objective_plugin']['path']=str(ROOT/OBJECTIVE)
    elif fault=='manifest':c['objective_extension']['forward_manifest_sha256']='0'*64
    elif fault=='symlink':(root/'shadow').symlink_to(source,target_is_directory=True)
    with pytest.raises(ValueError):verify_files(c)


def test_real_modules_install_in_fresh_process_without_path_overlay(tmp_path):
    c=setup(tmp_path);source=Path(c['source_root'])
    excluded={p for _,p in MODULES}
    for subtree in ('src','tools'):
        for path in (ROOT/subtree).rglob('*.py'):
            name=str(path.relative_to(ROOT))
            if name in excluded:continue
            target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=Path(c['dual_runtime_manifest'])
    manifest.write_text(json.dumps({str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')}))
    c['dual_runtime_identity']['source_manifest_sha256']=sha(manifest)
    c['objective_extension']['forward_manifest_sha256']=sha(manifest)
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(c))
    script='''import importlib.util,json,sys
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text())
entry=Path(c['objective_extension']['root'])/'bootstrap.py'
spec=importlib.util.spec_from_file_location('candidate_bootstrap',entry)
bootstrap=importlib.util.module_from_spec(spec);spec.loader.exec_module(bootstrap)
before=list(sys.path)
trainer=bootstrap.install(c)
assert sys.path==before
for formal in (False,True):
 compile(trainer.generate(c['source_root'],formal=formal),'<candidate>','exec')
actor=trainer.load_actor(c['source_root'],c['objective_plugin'])
assert actor.objective_plugin_identity['sha256']==c['objective_plugin']['sha256']
assert bootstrap.verify_files(c)
try: bootstrap.install(c)
except ValueError: pass
else: raise AssertionError('duplicate installation accepted')
print('isolated_install_passed')
'''
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join(str(source/p) for p in ('src','','tools/jev')))
    result=subprocess.run([sys.executable,'-B','-c',script,str(cfg)],cwd=tmp_path,
        env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'isolated_install_passed' in result.stdout


def test_install_from_script_inside_extension_root_ignores_its_own_tools_tree(tmp_path):
    """The real launch runs bootstrap.py by path, so sys.path[0] is the extension root.

    The sealed forward source has no tools/__init__.py, so `tools` is a namespace
    package and the extension root's own tools/reports payload joins it. Install
    must still resolve every parent package inside the sealed source.
    """
    c=setup(tmp_path);source=Path(c['source_root'])
    excluded={p for _,p in MODULES}
    for subtree in ('src','tools'):
        for path in (ROOT/subtree).rglob('*.py'):
            name=str(path.relative_to(ROOT))
            if name in excluded:continue
            target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    assert not (source/'tools/__init__.py').exists() and not (source/'tools/reports/__init__.py').exists()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=Path(c['dual_runtime_manifest'])
    manifest.write_text(json.dumps({str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')}))
    c['dual_runtime_identity']['source_manifest_sha256']=sha(manifest)
    c['objective_extension']['forward_manifest_sha256']=sha(manifest)
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(c))
    root=Path(c['objective_extension']['root'])
    # cwd is the extension root, so sys.path[0] ('') resolves there: this is the
    # import path torchrun gives bootstrap.py when it runs it by absolute path.
    # No file is added to the root: verify_files requires the exact payload.
    script='''import importlib.util,json,sys
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text())
entry=Path(c['objective_extension']['root'])/'bootstrap.py'
spec=importlib.util.spec_from_file_location('candidate_bootstrap',entry)
bootstrap=importlib.util.module_from_spec(spec);spec.loader.exec_module(bootstrap)
root=Path(c['objective_extension']['root']).resolve()
assert Path(sys.path[0] or '.').resolve()==root
bootstrap.install(c)
assert Path(sys.path[0] or '.').resolve()!=root
print('script_launch_install_passed')
'''
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join(str(source/p) for p in ('src','','tools/jev')))
    result=subprocess.run([sys.executable,'-B','-c',script,str(cfg)],cwd=root,
        env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'script_launch_install_passed' in result.stdout
