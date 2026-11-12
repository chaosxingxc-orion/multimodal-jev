"""Narrow identity checks for audited non-NPU host evidence readers.

Never exempts a declared NPU mapping or the mandatory host device-FD audit.
Unknown privileged containers remain reservations, including changed programs.
"""
import hashlib
from pathlib import PurePosixPath

IMAGE = 'sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026'
SCRIPT_SHA256 = '54718b1252f0485ac394550e458e6209bc9f91d843dea0d884cca08f9f860193'
PREFIX = ['/host', '/usr/bin/nsenter', '-t', '1', '-m', '/usr/bin/python3', '-c']
ARCHIVE_SCRIPT_SHA256 = '354df8c32045d00ba62b1e679550fee78a8ce0d69443eca5e73729cb30108892'
ARCHIVE_ROOT = '/var/tmp/s1-bl-candidate-ops-media-v1/runtime'


def is_audited_completion_observer(row):
    host = row['HostConfig']; args = row.get('Args', [])
    if (row.get('Image') != IMAGE or row.get('Path') != 'chroot'
            or row['State']['Status'] != 'running' or host.get('Privileged') is not True
            or host.get('Devices') or host.get('PidMode') != 'host'
            or host.get('AutoRemove') is not True or host.get('Binds') != ['/:/host']
            or len(args) not in (9, 10) or args[:7] != PREFIX):
        return False
    digest = hashlib.sha256(args[7].encode()).hexdigest()
    if len(args) == 10:
        # Exact reviewed tar-stream collector: two fixed host runtime paths,
        # no model imports, child execution or accelerator opens. A new script
        # or runtime root requires a separate audit, never a name exemption.
        return (digest == ARCHIVE_SCRIPT_SHA256
                and args[8:] == [ARCHIVE_ROOT, ARCHIVE_ROOT + '/queue'])
    if digest != SCRIPT_SHA256:
        return False
    path = PurePosixPath(args[8])
    return (str(path) == args[8] and path.parent == PurePosixPath('/var/tmp/omni-local-state')
            and path.name not in ('', '.', '..'))
