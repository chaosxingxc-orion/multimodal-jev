"""Run the frozen scoring service with a separately bound teardown deadline."""
import argparse
from pathlib import Path
import time

from multimodal_jev.local_checkpoint import commit_json
from tools.reports.context_replica_runtime import service_namespace
import json

def read(path):
    return json.loads(Path(path).read_text())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--engine-timeout', type=float, required=True)
    a = p.parse_args()
    cfg = read(a.config)
    ns = service_namespace(cfg)
    factory = ns['owner_class']

    def owner(root):
        class TeardownOwner(factory(root)):
            def close(self):
                if self.engine is not None:
                    self.engine.llm_engine.engine_core.shutdown(timeout=a.engine_timeout)
                    self.engine = None
                self.phase = 'closed'
        return TeardownOwner

    ns['owner_class'] = owner
    a.output.mkdir(exist_ok=False)
    try:
        ns['run'](cfg, a.output)
    except BaseException as exc:
        commit_json(a.output/'terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time()))
        raise
    commit_json(a.output/'terminal.json', dict(exit_code=0, finished_unix=time.time()))
