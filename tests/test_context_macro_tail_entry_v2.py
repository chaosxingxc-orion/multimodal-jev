from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from tools.reports.context_macro_tail_entry_v2 import main
from tools.reports import context_macro_tail_operations_v2 as operations
from test_prepare_context_macro_tail import fixture
from test_context_macro_training_operations import fixture as command_fixture


def test_two_ranks_share_output_parent_but_reject_reused_rank_directories(tmp_path):
    config = tmp_path/'config.json'; config.write_text('{"sentinel":1}')
    output = tmp_path/'run'; start = threading.Barrier(2)
    def rank_run(rank):
        start.wait(timeout=5)
        def runner(c, root):
            assert c == {'sentinel': 1}
            (root/f'rank-{rank}').mkdir(exist_ok=False)
            return rank
        return main(['--config', str(config), '--output', str(output)], runner=runner)
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(rank_run, i) for i in range(2)]
        assert [f.result(timeout=10) for f in futures] == [0, 1]
    with pytest.raises(FileExistsError):
        main(['--config', str(config), '--output', str(output)],
             runner=lambda c, root: (root/'rank-0').mkdir(exist_ok=False))


@pytest.mark.parametrize('role', ['preflight', 'client', 'service-0', 'service-1'])
def test_only_client_cli_changes(tmp_path, role):
    c, _ = command_fixture(tmp_path)
    before = operations.original.command(c, role)
    after = operations.command(c, role)
    if role == 'client':
        changed = [(a, b) for a, b in zip(before, after) if a != b]
        assert len(before) == len(after) and len(changed) == 1
        assert changed[0][1].endswith('/context_macro_tail_entry_v2.py')
    else: assert before == after


def test_new_package_contains_cli_and_preserves_parent_and_no_dispatch(tmp_path, monkeypatch):
    c, source = fixture(tmp_path, monkeypatch)
    # Bind after fixture changes so only validation/storage are test substitutes.
    prepare = operations.bind(operations.preparation.prepare, validate=lambda c: None,
        ADDITIONS=(*operations.preparation.ADDITIONS, 'tools/reports/context_macro_tail_entry_v2.py'))
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    result = prepare(c, {})
    from pathlib import Path
    package = Path(c['output'])/'package'
    assert (package/'tools/reports/context_macro_tail_entry_v2.py').is_file()
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    assert not result['launched']
    assert not json.loads((package.parent/'operations.json').read_text())['tail_diagnostic_dispatch_allowed']
