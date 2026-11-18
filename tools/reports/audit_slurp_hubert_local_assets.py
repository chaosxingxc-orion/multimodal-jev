"""Read-only exact-size/content search for pinned public SLURP baseline assets.

No imports of model loaders and no network. Does not traverse symlink directories.
Git blob SHA1 is used only where the publisher provides a regular Git object ID;
LFS weights use SHA256 of actual bytes. A pointer file is not model content.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def content_digest(path,algorithm):
    digest=hashlib.new('sha1' if algorithm=='git_blob_sha1' else 'sha256')
    if algorithm=='git_blob_sha1':digest.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def assets(audit):
    result=[]
    for kind,entries in [('task',audit['task_assets']),('base',audit['base_weight_candidates'])]:
        for entry in entries:
            lfs=entry.get('lfs')
            result.append(dict(role=kind,repository=audit[kind+'_repository'],revision=audit[kind+'_revision'],
                file=entry['rfilename'],bytes=entry['size'],algorithm='sha256' if lfs else 'git_blob_sha1',
                digest=lfs['sha256'] if lfs else entry['blobId']))
    return result


def scan(expected,roots):
    found={i:[] for i in range(len(expected))};errors=[];missing_roots=[];count=0;hashes=0;seen=set()
    by_size={}
    for i,entry in enumerate(expected):by_size.setdefault(entry['bytes'],[]).append(i)
    def onerror(e):errors.append(str(e))
    for root in roots:
        root=Path(root)
        if not root.is_dir():missing_roots.append(str(root));continue
        for parent,dirs,files in os.walk(root,followlinks=False,onerror=onerror):
            for name in files:
                path=Path(parent)/name
                try:
                    stat=path.stat();key=(stat.st_dev,stat.st_ino)
                    if key in seen:continue
                    seen.add(key);count+=1
                    for index in by_size.get(stat.st_size,[]):
                        entry=expected[index];actual=content_digest(path,entry['algorithm']);hashes+=1
                        if actual==entry['digest']:found[index].append(str(path))
                except OSError as e:errors.append(str(e))
    results=[dict(entry,matches=found[i],status='verified_content' if found[i] else 'not_found_in_scanned_files') for i,entry in enumerate(expected)]
    return dict(assets=results,roots=list(map(str,roots)),missing_roots=missing_roots,errors=errors,
        unique_files_statted=count,matching_size_hashes=hashes,symlink_directories_followed=False,
        all_core_assets_found=all(found.values()),missing_bytes=sum(r['bytes'] for r in results if not r['matches']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-audit',required=True);p.add_argument('--root',action='append',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.source_audit);r=scan(assets(json.loads(source.read_text())),a.root)
    r.update(schema='slurp-hubert-offline-asset-inventory-v1',source_audit=a.source_audit,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Core weights/tokenizer content only. Existing config/code receipts separate. No model load, training reproduction, download or remote-host filesystem search.')
    with Path(a.output).open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r,indent=2))
