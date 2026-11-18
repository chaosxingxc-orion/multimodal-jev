import hashlib
import io
import tarfile

import pytest

from tools.jev.collect_context_rl_selection import check_handle, unpack


def test_failed_terminal_is_collectable():
    launch = dict(container_id='a', image='image')
    check_handle(launch, dict(Id='a', Image='image', State=dict(Status='exited', ExitCode=1)))


@pytest.mark.parametrize('field,value', [('Id', 'wrong'), ('Image', 'wrong'), ('State', {'Status': 'running'})])
def test_handle_mismatch_or_live_rejected(field, value):
    observed = dict(Id='a', Image='image', State=dict(Status='exited'))
    observed[field] = value
    with pytest.raises(ValueError):check_handle(dict(container_id='a', image='image'), observed)


@pytest.mark.parametrize('fault', [None, 'escape', 'link', 'duplicate', 'missing', 'hash'])
def test_exact_archive_binding(tmp_path, fault):
    archive = tmp_path / 'data.tar';content = b'{"fixture":true}'
    files = {'config.json': hashlib.sha256(content).hexdigest()}
    with tarfile.open(archive, 'w') as tar:
        name = '../config.json' if fault == 'escape' else 'config.json'
        info = tarfile.TarInfo(name);info.size = len(content)
        if fault == 'link':info.type = tarfile.SYMTYPE;info.linkname = '/outside'
        tar.addfile(info, io.BytesIO(content))
        if fault == 'duplicate':tar.addfile(info, io.BytesIO(content))
    if fault == 'missing':files['other.json'] = files['config.json']
    if fault == 'hash':files['config.json'] = 'wrong'
    output = tmp_path / 'output';output.mkdir()
    if fault:
        with pytest.raises(ValueError):unpack(archive, output, files)
    else:
        unpack(archive, output, files)
        assert (output / 'config.json').read_bytes() == content
