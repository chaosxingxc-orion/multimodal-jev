"""Drive a complete CPU fixture through the published CLI and a temporary Git remote."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXP = "JEV-TEST"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def package(tmp_path):
    repo, state = tmp_path / "repo", tmp_path / "state"
    repo.mkdir()
    for name in (
        "run_experiment.py", "tools/reports/experiment_ledger.py",
        "tools/reports/experiment_package.py", "src/multimodal_jev/local_config.py",
        "src/multimodal_jev/local_checkpoint.py", "src/multimodal_jev/local_runtime_storage.py",
    ):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, repo / name)
    # Use a real local disk in production; pytest's staging may be tmpfs.
    (repo / "src/multimodal_jev/__init__.py").touch()
    (repo / "src/multimodal_jev/local_runtime_storage.py").write_text(
        "def check_paths(*args, **kwargs): pass\n")
    (repo / "tools/reports/fixture.py").write_text("""\
import json, sys
from pathlib import Path
root, phase = Path(sys.argv[1]), sys.argv[2]
with (root/'calls').open('a') as out:
    out.write(phase+'\\n')
if phase == 'train':
    assert (root/'data.json').is_file()
    if (root/'skip-audit').exists():
        sys.exit(0)
if phase == 'evaluate':
    assert (root/'train.json').is_file()
proof = dict(audited=True)
if phase == 'evaluate':
    proof['SLURP'] = {'candidate': {'queries': 2, 'policy_id': 'new', 'f1': 0.75},
                      'control': {'queries': 2, 'policy_id': 'old', 'f1': 0.5}}
(root/(phase+'.json')).write_text(json.dumps(proof))
""")
    (repo / "experiments/pipelines").mkdir(parents=True)
    (repo / "experiments/protocol.md").write_text("# Registered protocol\n")
    (repo / "wiki/reports").mkdir(parents=True)
    (repo / "wiki/reports/JEV-TEST.md").write_text("# Report\n\nHistorical notes.\n")
    (repo / "wiki/experiment-ledger.md").write_text("""\
# Ledger
## 实验队列

| 实验 ID | 优先级 | 状态 | 协议 | 下一步 |
| --- | ---: | --- | --- | --- |
| JEV-LATER | 20 | planned | [Protocol](../experiments/protocol.md) | Later |
| JEV-TEST | 10 | planned | [Protocol](../experiments/protocol.md) | Ready |

## History

Preserve this.
""")
    step = lambda phase: dict(id=phase, checks=[dict(path=str(state / f"{phase}.json"),
                                                     equals=[[["audited"], True]])],
                              run=[["tools.reports.fixture", str(state), phase]])
    stages = {phase: [step(phase)] for phase in ("data", "train", "evaluate")}
    stages["evaluate"][0]["panels"] = [
        dict(id="SLURP", scope="源任务", population=2, primary="Top-1 macro-F1",
             arms=[dict(label=label, proof=0, path=["SLURP", key], policy=policy,
                        metrics={"Top-1 macro-F1": "f1"})
                   for label, key, policy in (("New", "candidate", "new"), ("Old", "control", "old"))])
    ]
    spec = dict(schema="jev-experiment-package-v1", protocol="experiments/protocol.md",
                state_root=str(state), stages=stages)
    (repo / "experiments/pipelines/JEV-TEST.json").write_text(json.dumps(spec))
    local = tmp_path / "local.json"
    local.write_text(json.dumps(dict(schema="jev-local-macros-v1", macros={})))
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.name", "Fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Register fixture")

    def cli(*args, code=0):
        result = subprocess.run([sys.executable, "-B", str(repo / "run_experiment.py"),
                                 "--local", str(local), *args], cwd=tmp_path, capture_output=True,
                                text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        assert result.returncode == code, (result.stdout, result.stderr)
        return result

    return repo, state, cli


def test_claim_run_report_and_publish(package, tmp_path):
    repo, state, cli = package
    plan = json.loads(cli("plan").stdout)
    assert plan["experiment_id"] == EXP
    assert not state.exists()
    assert json.loads(cli("plan", "--experiment", "JEV-LATER", code=1).stdout)["status"] == "unbound"
    run = json.loads(cli("run").stdout)
    assert run["status"] == "complete"
    assert (state / "calls").read_text().splitlines() == ["data", "train", "evaluate"]
    report = (repo / "wiki/reports/JEV-TEST.md").read_text()
    assert "Historical notes." in report and "+25.00 个百分点" in report
    assert "| JEV-TEST | 10 | complete |" in (repo / "wiki/experiment-ledger.md").read_text()

    remote = tmp_path / "remote"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-q", "-u", "origin", "main")
    cli("publish", "--experiment", EXP)
    assert git(remote, "show", "main:wiki/reports/JEV-TEST.md").endswith("<!-- experiment-package:end -->")
    assert json.loads(cli("plan", code=1).stdout)["experiment_id"] == "JEV-LATER"


def test_exit_without_audit_is_not_resubmitted_and_resume_uses_receipts(package):
    repo, state, cli = package
    state.mkdir()
    (state / "skip-audit").touch()
    partial = json.loads(cli("run", code=1).stdout)
    assert partial["status"] == "in_progress"
    assert [s["status"] for s in partial["steps"]] == ["audited", "failed", "waiting_for_dependency"]
    cli("resume", "--experiment", EXP, code=1)
    assert (state / "calls").read_text().splitlines() == ["data", "train"]
    (state / "train.json").write_text('{"audited":true}')
    resumed = json.loads(cli("resume", "--experiment", EXP).stdout)
    assert resumed["status"] == "complete"
    assert (state / "calls").read_text().splitlines() == ["data", "train", "evaluate"]
    (state / "data.json").write_text('{"audited":true,"different":true}')
    invalid = json.loads(cli("status", "--experiment", EXP, code=1).stdout)
    assert invalid["steps"][0]["status"] == "failed"
