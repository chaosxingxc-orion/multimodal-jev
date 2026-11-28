"""Drive a complete CPU fixture through the published CLI and a temporary Git remote."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

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
        "src/multimodal_jev/tpo_anchor_layout.py",
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


def test_registered_tpo_device_plan_controls_dispatch(package, tmp_path):
    repo, state, cli = package
    config = "configs/portable/operations/tpo-single-stage-full.json"
    (repo / config).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / config, repo / config)
    local = tmp_path / "local.json"
    macros = dict(PROJECT_ROOT=str(repo), FULL_TPO_PACKAGE=str(state / "package"),
                  FULL_TPO_RUNTIME=str(state / "parent.json"), FULL_TPO_DEPLOYMENT={},
                  FULL_TPO_INITIAL=str(state / "initial"), FULL_TPO_INITIAL_CONFIG_SHA="fixture",
                  FULL_TPO_INPUT_IDENTITIES=str(state / "input"), FULL_TPO_CUSTODY=str(state / "custody"),
                  FULL_TPO_PREDECESSOR=str(state / "predecessor"),
                  FULL_TPO_TRAINING_DEVICES=[0, 1, 2, 3, 4, 5], FULL_TPO_INFERENCE_DEVICES=[6, 7],
                  FULL_TPO_OFFLINE_REFERENCE_DEVICES=list(range(8)))

    def bind():
        local.write_text(json.dumps(dict(schema="jev-local-macros-v1", macros=macros)))

    pipeline = repo / "experiments/pipelines/JEV-TEST.json"
    spec = json.loads(pipeline.read_bytes())
    data = spec["stages"]["data"][0]
    data.update(config=config, run=[["tools.reports.prepare_tpo_windows", config, "${EXPERIMENT_LOCAL_CONFIG}"],
                                    ["tools.reports.run_full_joint_tpo", config, "${EXPERIMENT_LOCAL_CONFIG}"]])
    pipeline.write_text(json.dumps(spec))
    stub = """\
from pathlib import Path
import sys
from multimodal_jev.local_config import load_config
c, _ = load_config(*sys.argv[1:])
assert c['offline_reference_devices'] == list(range(8))
assert Path.cwd() != Path(c['repository'])
"""
    (repo / "tools/reports/prepare_tpo_windows.py").write_text(stub + "Path(c['package']).mkdir()\n")
    (repo / "tools/reports/run_full_joint_tpo.py").write_text(stub +
        "assert Path(c['package']).is_dir()\n"
        "Path(c['package']).parent.joinpath('data.json').write_text('{\"audited\": true}')\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Register TPO test")

    macros["FULL_TPO_OFFLINE_REFERENCE_DEVICES"] = [6, 7]
    bind()
    rejected = json.loads(cli("plan", code=1).stdout)["steps"][0]
    assert rejected["status"] == "failed" and "Offline fixed reference" in rejected["detail"]
    assert not state.exists()

    macros["FULL_TPO_OFFLINE_REFERENCE_DEVICES"] = list(range(8))
    bind()
    expected = dict(fixed_reference=list(range(8)), actor=list(range(6)), online_reference=[6, 7])
    for action in ("plan", "status", "run"):
        result = json.loads(cli(action).stdout)
        assert result["steps"][0]["device_plan"] == expected
    assert result["status"] == "complete"
    assert json.loads((state / EXP / "data.log").read_text())["device_plan"] == expected


@pytest.mark.parametrize("ambiguous", [False, True])
def test_resume_at_intent_boundary(package, ambiguous):
    repo,state,cli = package
    state.mkdir()
    (state/'skip-audit').touch()
    cli('run',code=1)
    step = state/EXP
    # Reconstruct the two crash cuts on either side of backend invocation.
    for ext in ['exit.json','command-0.returned'] + ([] if ambiguous else ['command-0.started']):
        (step/f'train.{ext}').unlink()
    (state/'skip-audit').unlink()
    result=json.loads(cli('resume',code=int(ambiguous)).stdout)
    if ambiguous:
        assert result['steps'][1]['status']=='failed'
        assert 'original backend handle' in result['steps'][1]['detail']
        assert (state/'calls').read_text().count('train')==1
    else:
        assert result['status']=='complete'
        assert (state/'calls').read_text().count('train')==2


def test_killing_driver_during_backend_preserves_lock_and_single_submission(package):
    repo,state,cli=package
    fixture=repo/'tools/reports/fixture.py'
    source=fixture.read_text().replace("if phase == 'train':","""if phase == 'train':
    (root/'inside').touch()
    with (root/'gate').open() as gate: gate.read()
""")
    fixture.write_text(source)
    git(repo,'commit','-qam','Interruptible test backend')
    state.mkdir(); os.mkfifo(state/'gate')
    command=[sys.executable,str(repo/'run_experiment.py'),'--local',str(repo.parent/'local.json'),'run']
    child=subprocess.Popen(command,cwd=repo,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        for _ in range(500):
            if (state/'inside').exists(): break
            assert child.poll() is None
            time.sleep(.01)
        assert (state/'inside').exists()
        child.kill();child.wait(timeout=5)
        assert 'temporarily unavailable' in cli('resume',code=2).stderr
        original = git(repo, 'rev-parse', 'HEAD')
        fixture.write_text('raise AssertionError("new source must not run")\n')
        git(repo, 'commit', '-qam', 'Advance source while the backend owns the lease')
        with (state/'gate').open('w') as gate: gate.write('continue')
        result=json.loads(cli('resume','--wait-for-ledger').stdout)
        assert result['status']=='complete'
        assert result['execution_commit'] == original != result['commit']
        assert (state/'calls').read_text().splitlines()==['data','train','evaluate']
    finally:
        if child.poll() is None:
            child.kill();child.wait()


def test_stage_handoff_survives_wiki_commit_in_running_checkout(package):
    repo, state, _ = package
    fixture = repo / 'tools/reports/fixture.py'
    fixture.write_text(fixture.read_text().replace("if phase == 'train':", """if phase == 'train':
    (root/'inside').touch()
    with (root/'gate').open() as gate: gate.read()
"""))
    git(repo, 'commit', '-qam', 'Register training gate')
    original = git(repo, 'rev-parse', 'HEAD')
    state.mkdir()
    os.mkfifo(state/'gate')
    command = [sys.executable, str(repo/'run_experiment.py'), '--local', str(repo.parent/'local.json'), 'run']
    child = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(500):
            if (state/'inside').exists():
                break
            assert child.poll() is None
            time.sleep(.01)
        assert (state/'inside').exists()
        ledger = repo/'wiki/experiment-ledger.md'
        ledger.write_text(ledger.read_text().replace('| Ready |', '| Training milestone |'))
        report = repo/'wiki/reports/JEV-TEST.md'
        report.write_text(report.read_text() + '\nTraining milestone.\n')
        git(repo, 'commit', '-qam', 'Publish training milestone')
        with (state/'gate').open('w') as gate:
            gate.write('continue')
        stdout, stderr = child.communicate(timeout=15)
        assert child.returncode == 0, (stdout, stderr)
        assert json.loads(stdout)['status'] == 'complete'
        assert (state/'calls').read_text().splitlines() == ['data', 'train', 'evaluate']
        assert json.loads((state/EXP/'evaluate.intent.json').read_bytes())['commit'] == original
        assert 'Training milestone.' in report.read_text()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_resume_rejects_in_place_pipeline_change(package):
    repo, state, cli = package
    cli('run', '--through', 'data')
    pipeline = repo/'experiments/pipelines/JEV-TEST.json'
    spec = json.loads(pipeline.read_bytes())
    spec['stages']['evaluate'][0]['run'][0].append('changed')
    pipeline.write_text(json.dumps(spec))
    git(repo, 'commit', '-qam', 'Change the registered pipeline in place')
    assert 'pipeline_sha256' in cli('resume', code=2).stderr
    assert (state/'calls').read_text().splitlines() == ['data']


def test_wiki_publish_preserves_remote_updates_and_refuses_same_row_conflict(package, tmp_path):
    repo, state, cli = package
    remote, writer = tmp_path/'remote', tmp_path/'writer'
    git(tmp_path, 'init', '-q', '--bare', '-b', 'main', str(remote))
    git(repo, 'remote', 'add', 'origin', str(remote))
    git(repo, 'push', '-q', '-u', 'origin', 'main')
    git(tmp_path, 'clone', '-q', str(remote), str(writer))
    git(writer, 'config', 'user.name', 'Colleague')
    git(writer, 'config', 'user.email', 'writer@example.invalid')
    ledger = writer/'wiki/experiment-ledger.md'
    ledger.write_text(ledger.read_text().replace('| Later |', '| Other experiment milestone |'))
    git(writer, 'commit', '-qam', 'Update another experiment')
    git(writer, 'push', '-q')
    own_head = git(repo, 'rev-parse', 'HEAD')
    partial = json.loads(cli('run', '--through', 'train', '--publish').stdout)
    assert partial['published'] and partial['status'] == 'in_progress'
    result = json.loads(cli('resume', '--publish').stdout)
    assert result['published'] and result['status'] == 'complete'
    assert git(repo, 'rev-parse', 'HEAD') == own_head
    published = git(remote, 'show', 'main:wiki/experiment-ledger.md')
    assert '| JEV-TEST | 10 | complete |' in published
    assert '| Other experiment milestone |' in published
    cli('publish', '--experiment', EXP)
    git(writer, 'pull', '-q', '--ff-only')
    ledger.write_text(ledger.read_text().replace('| JEV-TEST | 10 | complete |', '| JEV-TEST | 10 | stopped |'))
    git(writer, 'commit', '-qam', 'Change the status of the same experiment')
    git(writer, 'push', '-q')
    assert 'Remote Wiki row changed' in cli('publish', '--experiment', EXP, code=2).stderr
    assert (state/'calls').read_text().splitlines() == ['data', 'train', 'evaluate']


def test_prerequisite_manifest_is_pinned_in_actual_backend_config(package, tmp_path):
    from hashlib import sha256
    repo, state, cli = package
    pipe = repo/'experiments/pipelines/JEV-TEST.json'
    spec = json.loads(pipe.read_bytes())
    op = 'experiments/downstream.json'
    (repo/op).write_text(json.dumps({'pool_sha256': '${INPUT_SHA}', 'root': '${OUTPUT_FAMILY}'}))
    spec['operation_bindings'] = {'OUTPUT_FAMILY': '${RUN_ROOT}/pool'}
    s = spec['stages']['train'][0]
    s.update(config=op, inputs=[{'path':str(state/'data.json')}], input_sha256_macros={'INPUT_SHA':0})
    s['run'][0].append('${EXPERIMENT_LOCAL_CONFIG}')
    fixture = repo/'tools/reports/fixture.py'
    fixture.write_text(fixture.read_text().replace("    assert (root/'data.json').is_file()", """    from multimodal_jev.local_config import load_config
    from hashlib import sha256
    cfg,_=load_config('experiments/downstream.json',sys.argv[3])
    assert cfg==dict(pool_sha256=sha256((root/'data.json').read_bytes()).hexdigest(),root=str(root/'pool'))
"""))
    pipe.write_text(json.dumps(spec))
    git(repo,'add','.')
    git(repo,'commit','-qm','Bind the downstream stage to its audited input')
    local=tmp_path/'local.json'
    def bind(**kwargs):
        local.write_text(json.dumps(dict(schema='jev-local-macros-v1',macros=dict(RUN_ROOT=str(state),**kwargs))))
    bind(OUTPUT_FAMILY='/old')
    assert 'override' in cli('plan',code=2).stderr
    bind(INPUT_SHA='old')
    cli('run','--through','data')
    assert 'overridden' in json.loads(cli('status',code=1).stdout)['steps'][1]['detail']
    bind()
    assert json.loads(cli('resume').stdout)['status']=='complete'
    assert json.loads((state/'JEV-TEST/train.macros.json').read_bytes())['macros']['INPUT_SHA']==sha256((state/'data.json').read_bytes()).hexdigest()
