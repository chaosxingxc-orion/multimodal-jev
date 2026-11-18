"""Package evaluation-only modules without changing sealed model sources.

The import extension contains a fixed allowlist. It cannot shadow a module in
the model source manifest, and all bytes are pinned in the generated entrypoint.
"""
import json
from pathlib import Path
import shutil

from multimodal_jev.baseline import file_hash


MODULES = (
    'context_transfer_portable_inputs',
    'audit_context_transfer_input_coverage',
    'audit_context_transfer_macro',
    'context_replica_transfer_client',
)


def bootstrap(hashes):
    return '''"""Pinned evaluation-only imports; model runtime stays unchanged."""
import argparse
import importlib
import json
from pathlib import Path
import sys
from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
import tools.reports

root = Path(__file__).resolve().parent / 'transfer-auxiliary'
expected = ''' + repr(hashes) + '''
if (root.is_symlink() or any(p.is_symlink() for p in root.rglob('*'))
        or {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} != set(expected)
        or any(not (root/name).is_file() or file_hash(root/name) != h for name,h in expected.items())):
    raise ValueError('pinned transfer auxiliary inventory changed')
for name in expected:
    module = 'tools.reports.' + name[:-3]
    if module in sys.modules or any((Path(p)/name).exists() for p in tools.reports.__path__):
        raise ValueError('auxiliary module cannot shadow sealed/runtime imports: ' + module)
tools.reports.__path__.append(str(root))
client = importlib.import_module('tools.reports.context_replica_transfer_client')
run = client.run
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=client.__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    p.add_argument('--phase', choices=['hf', 'macro'], required=True)
    a = p.parse_args()
    c, provenance = load_config(a.config, a.local)
    print(json.dumps(run(c, provenance, a.phase), indent=2))
'''


def prepare(package, source_manifest):
    """Add only new evaluation files to a fresh, unpublished package."""
    package = Path(package)
    manifest = json.loads(Path(source_manifest).read_text())
    filenames = [name + '.py' for name in MODULES]
    for name in filenames:
        if 'tools/reports/' + name in manifest:
            raise ValueError('auxiliary module already belongs to sealed runtime: ' + name)
    out = package/'transfer-auxiliary'
    wrapper = package/'transfer-client.py'
    if out.exists() or wrapper.exists():
        raise ValueError('fresh auxiliary destination required')
    out.mkdir()
    for name in filenames:
        shutil.copy2(Path(__file__).parent/name, out/name)
    hashes = {name: file_hash(out/name) for name in filenames}
    wrapper.write_text(bootstrap(hashes))
    return dict(files=hashes, entrypoint_sha256=file_hash(wrapper),
        runtime_source_manifest_sha256=file_hash(source_manifest),
        model_sources_modified=False, npu_executed=False)
