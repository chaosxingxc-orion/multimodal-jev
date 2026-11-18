"""Load an explicitly pinned training extension beside an unchanged runtime.

No sys.path overlay: only the three declared new module names are installed.
Inference services continue to import exclusively from the original source.
This bootstrap does not grant package, reference, or NPU admission.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys


MODULES = (
    ('multimodal_jev.objective_trace','src/multimodal_jev/objective_trace.py'),
    ('multimodal_jev.objective_plugin','src/multimodal_jev/objective_plugin.py'),
    ('tools.reports.context_plugin_trainer','tools/reports/context_plugin_trainer.py'),
)
OBJECTIVE = 'extensions/stage2_objective.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_files(config):
    extension=config['objective_extension']
    if set(extension)!={'root','files','forward_manifest_sha256'}:
        raise ValueError('explicit extension root, files and forward identity required')
    root=Path(extension['root']);source=Path(config['source_root'])
    if (not root.is_absolute() or not source.is_absolute() or root.is_symlink() or source.is_symlink()
            or root.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(root.resolve())):
        raise ValueError('disjoint regular forward and objective source directories required')
    expected={p for _,p in MODULES}|{OBJECTIVE,'bootstrap.py'}
    if set(extension['files'])!=expected:
        raise ValueError('only explicit new objective modules allowed')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual!=expected or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('complete regular extension payload required')
    for name,h in extension['files'].items():
        if sha(root/name)!=h or config['pins'].get(str(root/name))!=h:
            raise ValueError('extension file or runtime pin changed: '+name)
    manifest=Path(config['dual_runtime_manifest'])
    if (sha(manifest)!=extension['forward_manifest_sha256']
            or extension['forward_manifest_sha256']!=config['dual_runtime_identity']['source_manifest_sha256']):
        raise ValueError('unchanged forward runtime identity required')
    files=json.loads(manifest.read_text())
    actual={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py') if p.is_file()}
    if any(p.is_symlink() for p in source.rglob('*')) or actual!=files:
        raise ValueError('complete inherited forward source changed')
    if any((source/p).exists() for _,p in MODULES):
        raise ValueError('extension cannot override an inherited module')
    plugin=config['objective_plugin']
    if plugin['path']!=str(root/OBJECTIVE) or plugin['sha256']!=extension['files'][OBJECTIVE]:
        raise ValueError('objective file not bound to extension payload')
    return root,source


def install(config):
    root,source=verify_files(config)
    # Verify imported package locations before attaching modules. The launcher
    # must set PYTHONPATH to the sealed original source, never the worktree.
    #
    # Running this file as a script puts its own directory (the extension root)
    # at sys.path[0]. Because the sealed original source has no tools/__init__.py,
    # `tools` is a namespace package and the extension root's own tools/reports
    # payload would silently join it, so the sealed-source check below would see
    # a path outside the source. Remove exactly that entry: the extension payload
    # is installed by explicit file location later and needs no import path.
    extension_entry=Path(root).resolve()
    sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=extension_entry]
    if any(Path(p or '.').resolve()==extension_entry for p in sys.path):
        raise ValueError('extension root still on import path')
    parents={}
    for name,_ in MODULES:
        parent_name=name.rsplit('.',1)[0]
        parent=importlib.import_module(parent_name)
        paths=list(parent.__path__)
        if not paths or any(not Path(p).resolve().is_relative_to(source.resolve()) for p in paths):
            raise ValueError('parent package imported outside sealed source: '+parent_name)
        parents[parent_name]=parent
        if name in sys.modules or hasattr(parent,name.rsplit('.',1)[1]):
            raise ValueError('objective module already installed: '+name)
    for name,relative in MODULES:
        spec=importlib.util.spec_from_file_location(name,root/relative)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
        setattr(parents[name.rsplit('.',1)[0]],name.rsplit('.',1)[1],module)
    return sys.modules[MODULES[-1][0]]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    # Actual executed bootstrap bytes must match the sealed entry, not merely
    # a separately present copy bearing the same name.
    root=Path(config['objective_extension']['root'])
    if Path(__file__).resolve()!= (root/'bootstrap.py').resolve():
        raise ValueError('execute the packaged bootstrap entry')
    trainer=install(config)
    args.output.mkdir(parents=True,exist_ok=True)
    trainer.run(config,args.output)


if __name__=='__main__':main()
