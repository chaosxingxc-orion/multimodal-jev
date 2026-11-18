"""Prepare/reconstruct a two-update plugin package without changing forward sources.

No launch entry or admission flags. Existing reference/parent auditing is
reused, and all additional training code has a separate explicit identity.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports import prepare_context_cached_training as parent_api
from tools.reports.context_objective_bootstrap import MODULES,OBJECTIVE
from tools.reports.prepare_context_replica_training_candidate import rebase


def sha(raw):return hashlib.sha256(raw).hexdigest()
def encode(value):return (json.dumps(value,indent=2)+'\n').encode()


def read_bound(binding):
    p=Path(binding['path'])
    if set(binding)!={'path','sha256'} or p.is_symlink() or sha(p.read_bytes())!=binding['sha256']:
        raise ValueError('bound evidence changed')
    return p.read_bytes()


def expected(c,provenance):
    proof=parent_api.audit(c['parent'],provenance)
    if proof['status']!='cached_training_independently_audited_not_npu_admitted':
        raise ValueError('independently reconstructed cached parent required')
    parent=Path(c['parent']['output']);old=json.loads((parent/'PREPARED.json').read_text())
    config=json.loads((parent/'config.json').read_text());remote=c['remote_root']
    parent_api.path(remote)
    output=Path(c['output'])
    if (output.resolve().is_relative_to(parent.resolve()) or parent.resolve().is_relative_to(output.resolve())
            or Path(remote).is_relative_to(old['remote_root']) or Path(old['remote_root']).is_relative_to(remote)
            or Path(remote).is_relative_to(config['reference_dataset']['root'])
            or Path(config['reference_dataset']['root']).is_relative_to(remote)):
        raise ValueError('disjoint new training package and reference required')
    if (config['experiment_kind']!='engineering_prefix' or config['start_update']!=0
            or config['stop_update']!=2 or config.get('resume_root')):
        raise ValueError('initial two-update prefix parent required')
    report=json.loads(read_bound(c['extension_cpu_report']))
    snapshot=Path(c['extension_source'])
    manifest=json.loads((snapshot/'FILES.json').read_text())
    if (sha((snapshot/'FILES.json').read_bytes())!=report['source_manifest_sha256']
            or report['counts']!=dict(tests=47,failures=0,errors=0,skipped=0)
            or report['image']!=old['image'] or report['state']['ExitCode']!=0
            or report['state']['OOMKilled'] or report['terminal']['exit_code']!=0):
        raise ValueError('exact successful trace CPU evidence required')
    # The actual bound forward tree, not the current project, must be the one
    # tested with this extension. Missing inherited entrypoints are not ignored.
    for name,h in json.loads((parent/'RUNTIME-SOURCES.json').read_text()).items():
        if manifest.get(name)!=h:raise ValueError('CPU evidence used another forward dependency: '+name)
    additions={}
    for name in [p for _,p in MODULES]+[OBJECTIVE]:
        p=snapshot/name
        if p.is_symlink() or sha(p.read_bytes())!=manifest[name]:
            raise ValueError('tested extension changed: '+name)
        if (parent/'source'/name).exists():raise ValueError('extension overrides forward source')
        additions[name]=p.read_bytes()
    additions['bootstrap.py']=read_bound(c['bootstrap'])
    if not isinstance(c['options'],dict):raise ValueError('explicit objective options required')
    json.dumps(c['options'],allow_nan=False)
    files={n:(parent/n).read_bytes() for n in old['files']}
    for n,raw in additions.items():files['objective-extension/'+n]=raw
    runner=files['run.py'].decode()
    before="str(r/'source/tools/reports/context_cached_reference_trainer.py')"
    if runner.count(before)!=1:raise ValueError('cached runner anchor changed')
    files['run.py']=runner.replace(before,"str(r/'objective-extension/bootstrap.py')").encode()
    config=rebase(config,old['remote_root'],remote)
    config.update(objective_trace=True,objective_plugin=dict(path=remote+'/objective-extension/'+OBJECTIVE,
        sha256=sha(additions[OBJECTIVE]),options=copy.deepcopy(c['options'])),
        objective_extension=dict(root=remote+'/objective-extension',
            files={n:sha(raw) for n,raw in additions.items()},
            forward_manifest_sha256=config['dual_runtime_identity']['source_manifest_sha256']))
    if config['reference_dataset']!=json.loads((parent/'config.json').read_text())['reference_dataset']:
        raise ValueError('fixed reference binding changed')
    files['OBJECTIVE-DEPENDENCIES.json']=encode(dict(parent_manifest_sha256=c['parent']['manifest_sha256'],
        extension_cpu_report_sha256=c['extension_cpu_report']['sha256'],bootstrap_sha256=c['bootstrap']['sha256'],
        forward_identity=config['dual_runtime_identity'],objective_extension=config['objective_extension'],
        parent_audit=parent_api.without_provenance(proof)))
    names={'config.json',*old['layout']['services'].values()}
    external={n:h for n,h in config['pins'].items() if not Path(n).is_relative_to(remote)}
    config['pins']=dict(external,**{remote+'/'+n:sha(raw) for n,raw in files.items() if n not in names})
    files['config.json']=encode(config)
    for name in old['layout']['services'].values():
        value=rebase(json.loads((parent/name).read_text()),old['remote_root'],remote)
        value['pins']=config['pins'];files[name]=encode(value)
    m=dict(old,status='objective_training_candidate_prepared_only',remote_root=remote,
        parent_sha256=c['parent']['manifest_sha256'],files={n:sha(raw) for n,raw in files.items()},
        config_provenance=provenance,training_dispatch_allowed=False,finite_npu_dispatch_allowed=False,
        numerical_admission=False,objective_npu_admitted=False,
        pending=['bootstrap fixed-image validation','NPU forward parity and two updates',
                 'integrated independent objective and optimizer audit','fresh-process recovery'])
    # Include the dependency declaration in collection; extension bytes already
    # belong to pins and must receive effective readonly mounts at launch.
    m['layout']=copy.deepcopy(old['layout'])
    m['layout']['configs'].append('OBJECTIVE-DEPENDENCIES.json')
    return files,m


def prepare(c,provenance):
    root=Path(c['output'])
    if root.exists():raise ValueError('fresh training candidate required')
    files,manifest=expected(c,provenance)
    check_paths({'objective_training_candidate':root});root.mkdir(parents=True,exist_ok=False)
    for name,raw in files.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    commit_json(root/'PREPARED.json',manifest)
    return dict(status=manifest['status'],manifest_sha256=sha((root/'PREPARED.json').read_bytes()),
                training_dispatch_allowed=False,npu_executed=False)


def audit(c,provenance):
    root=Path(c['output']);p=root/'PREPARED.json'
    if root.is_symlink() or sha(p.read_bytes())!=c['manifest_sha256']:
        raise ValueError('candidate manifest changed')
    declared=json.loads(p.read_text());files,m=expected(c,declared['config_provenance'])
    actual={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file() and p!=root/'PREPARED.json'}
    if any(p.is_symlink() for p in root.rglob('*')) or actual!=files or declared!=m:
        raise ValueError('candidate differs from independent reconstruction')
    return dict(status='objective_training_candidate_reconstructed_not_admitted',
        manifest_sha256=c['manifest_sha256'],parent_sha256=c['parent']['manifest_sha256'],
        forward_source_unchanged=True,reference_binding_unchanged=True,
        training_dispatch_allowed=False,npu_admitted=False,provenance=provenance)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--local',required=True);p.add_argument('--action',choices=['prepare','audit'],default='prepare')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    result=prepare(c,provenance) if a.action=='prepare' else audit(c,provenance)
    if a.action=='audit':commit_json(c['audit_output'],result)
    print(json.dumps(result,indent=2))
