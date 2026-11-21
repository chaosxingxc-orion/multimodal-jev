"""Materialize a pinned HF git checkout from SHA256 LFS pointers into a shared cold snapshot."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.request import urlopen
from multimodal_jev.local_config import load_config


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(c, name):
    repo = c['sources'][name]
    checkout = Path(c['clones']) / repo['checkout']
    dest = Path(c['datasets']) / repo['directory']
    scratch = Path(c['clones'])
    def git(*args):
        return subprocess.check_output(['git','-C',str(checkout),*args])
    if git('rev-parse','HEAD').decode().strip()!=repo['revision']:
        raise ValueError('unexpected checkout')
    paths=git('ls-tree','-r','--name-only','-z','HEAD').decode().rstrip('\0').split('\0')
    selected = {item['path']: item['sha256'] for item in repo.get('eval', {}).get('files', {}).values()}
    if selected:
        if not set(selected) <= set(paths):
            raise ValueError('evaluation source missing from the pinned revision')
        paths = sorted(selected)
    def one(rel):
        src=checkout/rel
        # The clone has pointer files or small ordinary files. Verify its tree first.
        blob=src.read_bytes()
        gitsha=hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
        if tree[rel]!=gitsha:raise ValueError('git blob changed: '+rel)
        lfs=blob.startswith(b'version https://git-lfs.github.com/spec/v1\n')
        if lfs:
            _,oid,length=blob.decode().splitlines()
            expected=oid.removeprefix('oid sha256:');size=int(length.removeprefix('size '))
        else:
            size=len(blob);expected=hashlib.sha256(blob).hexdigest()
        if selected and selected[rel] != expected:
            raise ValueError('evaluation source digest differs from the pinned revision')
        target=dest/rel
        if target.is_file():
            if target.stat().st_size!=size or sha(target)!=expected:raise ValueError('existing source mismatch: '+rel)
            return size
        target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            local=Path(tmp)/'object'
            if lfs:
                url=f"https://huggingface.co/datasets/{repo['id']}/resolve/{repo['revision']}/{rel}"
                for attempt in range(5):
                    try:
                        with urlopen(url+f'?snapshot=full{attempt}',timeout=100) as stream,local.open('wb') as out:
                            shutil.copyfileobj(stream,out,2**20)
                        if local.stat().st_size!=size or sha(local)!=expected:raise ValueError('LFS verification failed')
                        break
                    except Exception:
                        if attempt==4:raise
                        time.sleep(2**attempt)
            else:local.write_bytes(blob)
            tmpname=target.with_name('.'+target.name+'.publishing')
            shutil.copyfile(local,tmpname);os.replace(tmpname,target)
        return size
    tree={entry.split('\t',1)[1]:entry.split('\t',1)[0].split()[2]
          for entry in git('ls-tree','-rz','HEAD').decode().rstrip('\0').split('\0')}
    with concurrent.futures.ThreadPoolExecutor(c['io_workers']) as pool:
        total=sum(pool.map(one,paths))
    print(json.dumps(dict(repo=repo['id'],revision=repo['revision'],files=len(paths),bytes=total)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('local');p.add_argument('source')
    a=p.parse_args();run(load_config(a.config,a.local)[0],a.source)
