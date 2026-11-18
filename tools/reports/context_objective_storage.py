"""Protect the complete objective extension using the shared storage plan."""
from pathlib import Path

from tools.reports.context_cached_storage_operations import command as cached_command
from tools.reports.context_cached_storage_operations import audit_mounts,mounts


def command(c,manifest,config):
    argv=cached_command(c,manifest,config)
    root=config['objective_extension']['root']
    if root!=manifest['remote_root']+'/objective-extension':
        raise ValueError('packaged objective directory required')
    # Replace individual pinned-file binds with a single readonly directory.
    # No writable child bind can then introduce extra imported files.
    out=[];index=0
    while index<len(argv):
        if argv[index]=='-v':
            destination=argv[index+1].split(':')[1]
            if Path(destination).is_relative_to(root):
                index+=2;continue
        out.append(argv[index]);index+=1
    index=out.index('--entrypoint');out[index:index]=['-v',root+':'+root+':ro']
    audit_mounts(out,config)
    if (root,root,True) not in mounts(out):
        raise ValueError('complete objective source must be readonly')
    return out
