"""Narrow identity check for an audited non-NPU host completion listener.

Never exempts a declared NPU mapping or the mandatory host device-FD audit.
Unknown privileged containers remain reservations, including changed programs.
"""
import hashlib
from pathlib import PurePosixPath

IMAGE = 'sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026'
SCRIPT_SHA256 = '54718b1252f0485ac394550e458e6209bc9f91d843dea0d884cca08f9f860193'
PREFIX = ['/host', '/usr/bin/nsenter', '-t', '1', '-m', '/usr/bin/python3', '-c']


def is_audited_completion_observer(row):
    host = row['HostConfig']; args = row.get('Args', [])
    if (row.get('Image') != IMAGE or row.get('Path') != 'chroot'
            or row['State']['Status'] != 'running' or host.get('Privileged') is not True
            or host.get('Devices') or host.get('PidMode') != 'host'
            or host.get('AutoRemove') is not True or host.get('Binds') != ['/:/host']
            or len(args) != 9 or args[:7] != PREFIX
            or hashlib.sha256(args[7].encode()).hexdigest() != SCRIPT_SHA256):
        return False
    path = PurePosixPath(args[8])
    return (str(path) == args[8] and path.parent == PurePosixPath('/var/tmp/omni-local-state')
            and path.name not in ('', '.', '..'))
