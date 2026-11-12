#!/usr/bin/env python3
"""Reauthenticate the complete matrix before optional supplementary reporting."""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.fsc_transfer_report import summarize
from score_fsc_span_transfer import run,read


def report(protocol_path,digest):
    full=run(protocol_path,digest) # Actual Docker/gate/matrix checks before target labels.
    protocol=read(protocol_path);result=summarize(read(protocol['task']),full['policies'])
    result['files']=dict(full['files'])
    for p in (Path(__file__),Path(__file__).resolve().parents[2]/'src/multimodal_jev/fsc_transfer_report.py'):
        result['files'][str(p)]=file_hash(p)
    result['protocol_sha256']=digest
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable supplementary report exists')
    atomic_json(a.output,report(a.protocol,a.protocol_sha256))
