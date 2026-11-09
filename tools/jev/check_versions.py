import importlib.metadata as md
import json
from pathlib import Path
manifest = json.loads(Path('/opt/multimodal-jev/IMAGE-MANIFEST.json').read_text())
for name, expected in manifest['required_packages'].items():
    actual = md.version(name)
    if actual != expected:
        raise SystemExit(f'{name}: expected {expected}, got {actual}')
print('All pinned runtime versions match')
