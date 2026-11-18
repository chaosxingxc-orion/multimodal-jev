"""Explicit one-shot successor chain for a recovered, previously stopped run.

Owns original successor locks, waits an exact pidfd, retains old failures, and
uses original immutable specs. No periodic model calls or implicit retries.
"""
import argparse
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import select
import subprocess
import sys
import time

from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.jev.collect_context_rl_training import sha
from tools.reports.committed_signal_report import read, write


def verify(pins):
    if not pins: raise ValueError('Explicit dependency pins required')
    for name, digest in pins.items():
        if sha(name) != digest: raise ValueError('Dependency changed: '+name)


def process_identity(pid):
    root = Path('/proc')/str(pid)
    stat = (root/'stat').read_text().rsplit(')', 1)[1].split()
    return dict(start_ticks=stat[19], cmdline_sha256=sha(root/'cmdline'))


def wait_exact(process):
    pid = process['pid']
    fd = os.pidfd_open(pid)
    try:
        if process_identity(pid) != {k:process[k] for k in ('start_ticks','cmdline_sha256')}:
            raise ValueError('Predecessor process identity differs')
        poller = select.poll();poller.register(fd, select.POLLIN)
        poller.poll()
    finally:
        os.close(fd)


def command(stage):
    module = stage['module']
    from tools.reports.context_compatible_pipeline import DIRECT, DRIVERS
    permitted = set(DIRECT) | set(DRIVERS) | {
        'tools.reports.compare_fsc_context_reference',
        'tools.reports.run_slurp_context_test_experiment',
        'tools.reports.run_minds_context_experiment',
    }
    if module not in permitted: raise ValueError('Unregistered successor command')
    argv = [sys.executable, '-m']
    if module in DIRECT or module in DRIVERS:
        argv += ['tools.reports.context_compatible_pipeline','--module',module,'--']
    else: argv += [module]
    return argv + stage['arguments']


def run(template, local):
    c, provenance = load_config(template,local)
    required={'schema','output_directory','predecessor_process','predecessor_root','pins',
              'stopped_states','owner_locks','absent_paths','aliases','stages'}
    if set(c)!=required or c['schema']!='recovered-research-successors-v1':
        raise ValueError('Explicit recovered chain configuration required')
    verify(c['pins'])
    output=Path(c['output_directory'])
    check_paths({'evidence':output})
    if output.exists():raise ValueError('Existing successor attempt; preserve exact handles')
    with ExitStack() as stack:
        for name in c['owner_locks']:
            owner=stack.enter_context(Path(name).open('r'))
            fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,digest in c['stopped_states'].items():
            if sha(name)!=digest or not read(name)['phase'].startswith('stopped_preserve'):
                raise ValueError('Known predecessor failure changed')
        for name in c['absent_paths']:
            if os.path.lexists(name):raise ValueError('Existing downstream work: '+name)
        for stage in c['stages']:command(stage)
        output.mkdir(parents=True)
        write(output/'input.json',c);write(output/'provenance.json',provenance)
        def status(phase,**kw):
            write(output/'status.json',dict(phase=phase,pid=os.getpid(),unix=time.time(),**kw))
        try:
            status('waiting_exact_recovery_process',predecessor=c['predecessor_process'])
            wait_exact(c['predecessor_process'])
            root=Path(c['predecessor_root'])
            with (root/'owner.lock').open('r') as predecessor:
                fcntl.flock(predecessor,fcntl.LOCK_SH|fcntl.LOCK_NB)
                if read(root/'status.json')['phase']!='complete_training_and_selection_curve':
                    raise ValueError('Recovery did not complete original curve')
            verify(c['pins'])
            for name in c['absent_paths']:
                if os.path.lexists(name):raise ValueError('Downstream work appeared while waiting: '+name)
            # Only never-started output roots may be redirected to persistent
            # local storage. Original specs and downstream aliases stay intact.
            for alias,target in c['aliases'].items():
                alias,target=Path(alias),Path(target)
                check_paths({'alias_parent':alias.parent,'destination':target})
                if os.path.lexists(alias) or target.exists():raise ValueError('Alias/output already exists')
                target.parent.mkdir(parents=True,exist_ok=True)
                alias.symlink_to(target,target_is_directory=True)
            write(output/'aliases.json',c['aliases'])
            completed=[]
            for stage in c['stages']:
                verify(c['pins'])
                check_paths({name:Path(name) for name in stage['storage_paths']})
                status('executing',stage=stage['name'],completed=completed)
                with (output/(stage['name']+'.log')).open('x') as log:
                    subprocess.run(command(stage),stdin=subprocess.DEVNULL,stdout=log,
                                   stderr=subprocess.STDOUT,check=True)
                verify(c['pins'])
                for result in stage['results']:
                    actual=read(result['path'])
                    if any(actual.get(k)!=v for k,v in result['equals'].items()):
                        raise ValueError('Stage did not produce accepted terminal result')
                completed.append(stage['name'])
                write(output/(stage['name']+'-complete.json'),dict(
                    outputs={r['path']:sha(r['path']) for r in stage['results']}))
            status('complete_predeclared_chain',completed=completed,complex_training_admitted=False)
        except BaseException as exc:
            status('stopped_preserve_handles',error=repr(exc),restart=False)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    a=parser.parse_args();run(a.config,a.local)
