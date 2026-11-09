"""Hash model and processor assets consistently for rollout and actor gates."""
import hashlib
import json
from pathlib import Path
from .baseline import file_hash


def checkpoint_identity(checkpoint):
    root=Path(checkpoint)
    files={str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*'))
           if p.is_file() and p.suffix in {'.json','.jinja','.safetensors','.model','.txt'}}
    if not any(name.endswith('.safetensors') for name in files):
        raise ValueError('model weights missing')
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),files
