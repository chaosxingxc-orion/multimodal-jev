import os,socket,uuid
import pytest
from multimodal_jev.local_ipc import check_ipc_directory


def test_linux_byte_limit_not_character_count():
    assert check_ipc_directory('/'+'a'*69)['maximum_socket_path_bytes']==107
    for path in ('/'+'a'*70,'/'+'中'*24,'relative/path'):
        with pytest.raises(ValueError):check_ipc_directory(path)


def test_actual_unix_socket_in_short_local_temp(tmp_path):
    # pytest temp roots may be long: use a short per-test directory explicitly.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='ji-',dir='/tmp') as directory:
        check_ipc_directory(directory)
        path=directory+'/'+str(uuid.uuid4())
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
            server.bind(path)
            assert os.path.exists(path)


def test_original_failure_path_rejected_before_backend_start():
    with pytest.raises(ValueError,match='107'):
        check_ipc_directory('/data/xingchao/multimodal-jev/simple-final-v1/sampled-npu-grpo_v1-parity-v1/cache')
