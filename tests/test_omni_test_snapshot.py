import subprocess
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from run_omni_test_queue import export_source


def test_snapshot_export_uses_explicit_repo_and_exact_commit_from_non_git_cwd(tmp_path,monkeypatch):
    repo=tmp_path/'repo';repo.mkdir()
    def git(*args):
        return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
    git('init','-q');git('config','user.name','Fixture');git('config','user.email','fixture@example.invalid')
    (repo/'input.txt').write_text('pinned source')
    git('add','input.txt');git('commit','-qm','fixture');commit=git('rev-parse','HEAD')
    (repo/'input.txt').write_text('uncommitted change must not leak')
    cwd=tmp_path/'snapshot';cwd.mkdir();monkeypatch.chdir(cwd)
    destination=tmp_path/'export'
    export_source(repo,commit,destination)
    assert (destination/'input.txt').read_text()=='pinned source'
    with pytest.raises(ValueError):export_source(repo,commit[:8],tmp_path/'short')
    with pytest.raises(subprocess.CalledProcessError):export_source(cwd,commit,tmp_path/'bad')
    assert not (tmp_path/'bad').exists() and not (tmp_path/'short').exists()
