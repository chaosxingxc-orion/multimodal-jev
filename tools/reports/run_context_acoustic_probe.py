"""Run fresh baseline/acoustic service32 processes and retain numerical failures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import require_training_storage
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.measure_context_acoustic_candidate import compare


def run(package,output):
    import torch
    package,output=Path(package),Path(output)
    require_training_storage(output);output.mkdir(exist_ok=False)
    try:
        commit_json(output/'source.json',dict(files_sha256=verify(package)))
        reports={}
        for mode in ('baseline','acoustic'):
            argv=[sys.executable,'-B',str(package/'tools/reports/measure_context_acoustic_candidate.py'),
                '--config',str(package/'config.json'),'--output',str(output/mode),'--mode',mode]
            commit_json(output/(mode+'-command.json'),dict(argv=argv))
            with (output/(mode+'.log')).open('x') as log:
                result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
            commit_json(output/(mode+'-exit.json'),dict(exit_code=result.returncode))
            if result.returncode:raise RuntimeError(mode+' failed; preserve log')
            reports[mode]=json.loads((output/mode/'measurement.json').read_text())
        rows=torch.load(package/'inputs.pt',map_location='cpu',weights_only=True)
        result=compare(reports,rows)
        commit_json(output/'comparison.json',result)
        commit_json(output/'terminal.json',dict(exit_code=0,training_dispatch_allowed=False))
        return result
    except BaseException as exc:
        commit_json(output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.package,a.output)
