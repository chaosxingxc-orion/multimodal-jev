"""Macro-configured one-shot endpoint analysis; no training or queue mutation."""
import argparse
import json
from pathlib import Path
import time
import traceback

import torch

from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_probability_endpoint_uncertainty import run
from tools.reports.committed_signal_report import sha, write
from tools.reports.context_metric_python312 import runtime_metric_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--local', required=True)
    args = parser.parse_args(); c, provenance = load_config(args.config, args.local)
    output = Path(c['output']).resolve()
    check_paths({'endpoint_output': output})
    output.mkdir(parents=True, exist_ok=False)
    write(output/'attempt.json', dict(config=c, config_provenance=provenance,
        started_unix=time.time(), implementation_sha256=sha(__file__)))
    torch.set_num_threads(1)
    try:
        # Match the original fixed-image Python 3.12 metric accumulation.
        # This is the same scoped compatibility used by the successor queue.
        with runtime_metric_sum():
            result = run(c['directory'], c['algorithm'], c['release'])
        write(output/'report.json', result)
        write(output/'COMPLETE.json', dict(passed=True, finished_unix=time.time(),
            report_sha256=sha(output/'report.json'), algorithm=c['algorithm'],
            complex_training_admitted=False))
        print(json.dumps(result['probability'], indent=2), flush=True)
    except BaseException as exc:
        write(output/'FAILED.json', dict(error=repr(exc), traceback=traceback.format_exc(),
            finished_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
