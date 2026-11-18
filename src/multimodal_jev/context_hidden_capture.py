"""Explicit one-request diagnostic capture; never enabled by serving defaults."""
from pathlib import Path
import torch

from .local_checkpoint import commit_file
from .local_runtime_storage import require_training_storage


def capture_next_readout(readout, destination):
    path = Path(destination)
    require_training_storage(path.parent)
    if path.exists() or getattr(readout, '_jev_capture_pending', False):
        raise ValueError('new destination and no pending diagnostic capture required')
    readout._jev_capture_pending = True

    def capture(module, args, output):
        hidden, lengths, prompts, extras = args
        value = dict(hidden=hidden.detach().cpu(), lengths=lengths,
                     prompts=[p.detach().cpu() for p in prompts], extras=extras,
                     probabilities=[p.detach().cpu() for p in output],
                     head={k:v.detach().cpu() for k,v in module.head.state_dict().items()})
        commit_file(path, lambda stream: torch.save(value, stream))
        handle.remove()
        module._jev_capture_pending = False

    handle = readout.register_forward_hook(capture)
