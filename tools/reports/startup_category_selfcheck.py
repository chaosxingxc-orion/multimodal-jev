"""Configured CPU selfcheck by root-cause class; never dispatches training.

Tests run from a private read-only source snapshot with external logs/temp.
Findings are fail-closed, level-specific evidence, not NPU admission.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tempfile
import xml.etree.ElementTree as ET

from multimodal_jev.context_execution import map_io, widths
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.source_snapshot import freeze, verify


def rewrite_sites(root, paths):
    """Conservative inventory; replace sites are review findings, not proof of exploit."""
    findings=[]
    for relative in paths:
        tree=ast.parse((root/relative).read_text())
        for node in ast.walk(tree):
            if not isinstance(node,ast.Call):continue
            function=node.func
            name=function.id if isinstance(function,ast.Name) else function.attr if isinstance(function,ast.Attribute) else ''
            if name in ('getsource','exec','replace'):
                findings.append(dict(path=relative,line=node.lineno,call=name))
    return findings


def run(config,provenance):
    execution=widths(config['execution'])
    root=Path(config['output']);check_paths({'selfcheck':root})
    root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',config);commit_json(root/'provenance.json',provenance)
    try:
        snapshot=freeze(config['project'],root/'source',patterns=config['source_patterns'],
                        io_workers=execution['io_workers'])
        commit_json(root/'source-binding.json',snapshot)
        source=Path(snapshot['root'])
        temp_parent=Path(config['scratch_root'])
        check_paths({'selfcheck_scratch':temp_parent})
        temp_parent.mkdir(parents=True,exist_ok=True)
        def test(spec):
            name=spec['id']
            if not name.isidentifier():raise ValueError('identifier check ID required')
            out=root/name;out.mkdir()
            # AF_UNIX sockets have a 108-byte pathname limit. Evidence paths
            # are verbose; use a configured short local scratch path instead.
            tmp=Path(tempfile.mkdtemp(prefix='jsc-',dir=temp_parent))
            if len(os.fsencode(str(tmp)))>40:
                raise ValueError('short scratch root required for socket integration tests')
            env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
                     PYTHONSAFEPATH='1',TMPDIR=str(tmp),
                     PYTHONPATH=os.pathsep.join([str(source/p) for p in ('src','','tools/jev','tests')]+config['support_pythonpaths']))
            argv=[config['python'],'-B','-m','pytest','-q','-p','no:cacheprovider',
                  '--basetemp='+str(tmp/'p'),'--junitxml='+str(out/'junit.xml'),*spec['tests']]
            start=time.monotonic()
            try:
                with (out/'pytest.log').open('x') as log:
                    child=subprocess.Popen(argv,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,
                                           start_new_session=True)
                    try:
                        code=child.wait(timeout=config['test_timeout_seconds'])
                    finally:
                        # Kill only our test process group, including a worker
                        # left behind after pytest timed out or exited.
                        try:os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        child.wait()
                suites=ET.parse(out/'junit.xml').getroot().iter('testsuite')
                counts={k:0 for k in ('tests','errors','failures','skipped')}
                for suite in suites:
                    for k in counts:counts[k]+=int(suite.attrib.get(k,0))
                passed=code==0 and counts['tests']>0 and not any(counts[k] for k in ('errors','failures','skipped'))
                record=dict(id=name,category=spec['category'],status='verified_cpu' if passed else 'failed',
                            exit_code=code,counts=counts)
            except (subprocess.TimeoutExpired,FileNotFoundError,ET.ParseError) as error:
                record=dict(id=name,category=spec['category'],status='failed',error=repr(error))
            record.update(seconds=time.monotonic()-start,argv=argv,log=str(out/'pytest.log'),scratch=str(tmp))
            commit_json(out/'result.json',record)
            return record
        results,stats=map_io(test,config['test_groups'],workers=execution['cpu_workers'])
        # Each subprocess is a single-thread CPU worker. Thread orchestration
        # only waits for independent, bounded process groups.
        for spec in config['static_checks']:
            sites=rewrite_sites(source,spec['paths'])
            results.append(dict(id=spec['id'],category=spec['category'],
                                status='failed' if sites else 'verified_cpu',findings=sites))
        for spec in config['pending_checks']:
            results.append(dict(spec,status='not_exercised'))
        verify(source,manifest_sha256=snapshot['manifest_sha256'],io_workers=execution['io_workers'])
        report=dict(schema='jev-startup-category-selfcheck-v1',results=results,execution=stats,
                    source=snapshot,model_calls=0,npu_executed=False,training_dispatch_allowed=False,
                    release_ready=False,status='classified_selfcheck_complete_not_release_admission')
        commit_json(root/'report.json',report)
        code=1 if any(r['status']=='failed' for r in results) else 0
        commit_json(root/'terminal.json',dict(exit_code=code,status=report['status'],release_ready=False))
        return report
    except BaseException as error:
        if not (root/'terminal.json').exists():
            commit_json(root/'terminal.json',dict(exit_code=1,error=repr(error),restart=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    report=run(config,provenance)
    print(json.dumps(dict(output=config['output'],checks=[{k:r[k] for k in ('id','category','status')} for r in report['results']],release_ready=False),indent=2))
    sys.exit(1 if any(r['status']=='failed' for r in report['results']) else 0)
