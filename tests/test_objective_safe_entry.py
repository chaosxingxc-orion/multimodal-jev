"""Regression: direct script execution must not add extension namespace roots."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('historical',[True,False])
@pytest.mark.parametrize('launcher',['python','torchrun'])
def test_actual_bootstrap_with_namespace_forward_tree_and_direct_entry(tmp_path,historical,launcher):
    from tools.reports.context_objective_bootstrap import MODULES,OBJECTIVE
    # Match packaged namespace layout: no tools/__init__.py or reports/__init__.
    source=tmp_path/'source';extension=tmp_path/'objective-extension';excluded={p for _,p in MODULES}
    for folder in ('src/multimodal_jev','tools/reports','tools/jev'):
        for p in (ROOT/folder).glob('*.py'):
            relative=str(p.relative_to(ROOT))
            if relative in excluded or relative in ('tools/__init__.py','tools/reports/__init__.py'):continue
            dest=source/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    for relative in excluded|{OBJECTIVE}:
        dest=extension/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,dest)
    bootstrap=(ROOT/'tools/reports/context_objective_bootstrap.py').read_text()
    # Reproduce the historical unpatched bootstrap without mutating live code.
    begin=bootstrap.find('    extension_entry=Path(root).resolve()')
    if historical and begin>=0:
        end=bootstrap.index('    parents={}',begin)
        bootstrap=bootstrap[:begin]+bootstrap[end:]
    (extension/'bootstrap.py').write_text(bootstrap)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=tmp_path/'RUNTIME-SOURCES.json';manifest.write_text(json.dumps({str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')}))
    files={str(p.relative_to(extension)):sha(p) for p in extension.rglob('*') if p.is_file()}
    c=dict(source_root=str(source),dual_runtime_manifest=str(manifest),dual_runtime_identity=dict(source_manifest_sha256=sha(manifest)),
        objective_extension=dict(root=str(extension),files=files,forward_manifest_sha256=sha(manifest)),
        objective_plugin=dict(path=str(extension/OBJECTIVE),sha256=files[OBJECTIVE],options={}),pins={str(extension/k):v for k,v in files.items()})
    config=tmp_path/'config.json';config.write_text(json.dumps(c))
    # Run from the extension directory, as torchrun does for bootstrap.py.
    probe=extension/'probe.py';probe.write_text('''import json,sys
from pathlib import Path
import bootstrap
c=json.loads(Path(sys.argv[1]).read_text())
Path(__file__).unlink()
trainer=bootstrap.install(c)
for formal in (False,True):compile(trainer.generate(c['source_root'],formal=formal),'generated','exec')
print('PASS')
'''.replace('import bootstrap',"import importlib.util\ns=importlib.util.spec_from_file_location('bootstrap',Path(sys.argv[1]).parent/'objective-extension/bootstrap.py')\nbootstrap=importlib.util.module_from_spec(s);s.loader.exec_module(bootstrap)"))
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(source/p) for p in ('','src','tools/jev')),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
    env.pop('PYTHONSAFEPATH',None)
    raw=probe.read_text()
    argv=[sys.executable]
    if launcher=='torchrun':
        # Match the production static rendezvous. --standalone advertises the
        # container hostname, which cannot resolve under network=none.
        argv+=['-m','torch.distributed.run','--nnodes=1','--node-rank=0',
               '--master-addr=127.0.0.1','--master-port=29617','--nproc-per-node=1']
    argv += [str(probe),str(config)]
    bad=subprocess.run(argv,cwd=tmp_path,env=env,capture_output=True,text=True,timeout=60)
    if historical:
        assert bad.returncode!=0 and 'parent package imported outside sealed source: tools.reports' in bad.stderr, bad.stdout+bad.stderr
    else:
        assert bad.returncode==0,bad.stdout+bad.stderr
    probe.write_text(raw);env['PYTHONSAFEPATH']='1'
    good=subprocess.run(argv,cwd=tmp_path,env=env,capture_output=True,text=True,timeout=60)
    assert good.returncode==0,good.stdout+good.stderr
    assert 'PASS' in good.stdout
    # Removing only the extension root must not disable sealed-source checks.
    foreign=tmp_path/'foreign';(foreign/'tools/reports').mkdir(parents=True)
    probe.write_text(raw);env['PYTHONPATH']=str(foreign)+os.pathsep+env['PYTHONPATH']
    contaminated=subprocess.run(argv,cwd=tmp_path,env=env,capture_output=True,text=True,timeout=60)
    assert contaminated.returncode!=0 and 'outside sealed source' in contaminated.stderr
