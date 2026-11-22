"""Replay the frozen single-card finite window, changing only its execution microbatch."""
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports.experiment_ledger import plan
from tools.reports.native_seal_operations import run


def read(p):
    return json.loads(p.read_bytes())


def capacity(experiment, local):
    from tools.reports.experiment_package import REGISTRY, committed
    selected = plan(experiment, states=('in_progress', 'planned'))
    spec = json.loads(committed(selected['commit'], REGISTRY / (experiment+'.json')))
    protocol = Path(spec['capacity_window'])
    if protocol.read_bytes() != committed(selected['commit'], protocol):
        raise ValueError('capacity protocol differs from ledger')
    c, _ = load_config(str(protocol), str(local))
    base = Path(c['package'])
    parent = read(base/'operations-local.json')['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    ops = Path(parent['operations'])
    prep = read(ops/'prepare-collected.json')
    old = read(ops/'finite-collected.json')
    if (old['engineering_prefix'] is not True or old['updates'] != 2
            or prep['single_card']['questions_per_sequence'] != 1
            or file_hash(parent['archive']) != c['archive_sha256']
            or parent['entry_archive_sha256'] != c['archive_sha256']):
        raise ValueError('frozen parent or admission differs')
    def checked(*args):
        return subprocess.check_output(['ssh','-o','BatchMode=yes',parent['host'],shlex.join(args)])
    root = parent['remote_root']
    if (json.loads(checked('cat',root+'/prepare/receipt.json')) != prep
            or checked('sha256sum', root+'/prepare/single.json').decode().split()[0] != prep['single_sha256']):
        raise ValueError('prepared prompt identity differs')
    queue = Path(c['output']); queue.mkdir(parents=True,exist_ok=True)
    finite = next(j for g in parent['chains'][0] for j in g if j['name']=='finite')
    leaf = read(base/'entry/finite.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    result = {str(leaf['execution_microbatch']):dict(status='passed',ops=str(ops))}
    for width in c['widths']:
        if str(width) in result:
            continue
        if type(width)!=int or width<1:
            raise ValueError('invalid capacity width')
        tag = parent['tag']+'-capacity-m'+str(width)
        dst=queue/tag
        dst.mkdir()
        archive=dst/'entry.tar'
        with tarfile.open(parent['archive'],'r') as src, tarfile.open(archive,'x') as out:
            for member in src:
                if member.name=='entry/finite.json':
                    spec=read(base/'entry/finite.json')
                    spec['macros']['NATIVE_SEAL_CONTAINER_CONFIG']['execution_microbatch']=width
                    raw=(json.dumps(spec,indent=2)+'\n').encode()
                    member.size=len(raw)
                    out.addfile(member,io.BytesIO(raw))
                else:
                    out.addfile(member,src.extractfile(member) if member.isfile() else None)
        from hashlib import sha256
        cfg=dict(parent, tag=tag, remote_root=str(Path(root).with_name(tag)), operations=str(dst/'ops'),
                 archive=str(archive),entry_archive_sha256=file_hash(archive),
                 entry_pins={**parent['entry_pins'],'finite.json':sha256(raw).hexdigest()},
                 read_only_mounts=parent['read_only_mounts']+[root+'/prepare:/jev-work/prepare'],
                 chains=[[[finite]]])
        commit_json(dst/'operations.json',cfg)
        try:
            if run(cfg):raise RuntimeError('finite command failed')
        except Exception:
            p=dst/'ops/finite.log'
            if not p.exists() or not re.search(r'(out of memory|alloc.*memory.*failed)',p.read_text(),re.I):
                raise
            status='out_of_memory'
        else:
            proof=read(dst/'ops/finite-collected.json')
            if proof['updates']!=old['updates'] or proof['engineering_prefix'] is not True:
                raise ValueError('finite window incomplete')
            status='passed'
        result[str(width)]=dict(status=status,ops=cfg['operations'])
        commit_json(dst/'result.json',result[str(width)])
    commit_json(queue/'results.json',result)
    return result
