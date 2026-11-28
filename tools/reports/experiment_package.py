"""Run one registered experiment through existing data, training and evaluation tools."""
import argparse
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from tempfile import NamedTemporaryFile, TemporaryDirectory

from multimodal_jev.local_checkpoint import commit_json, sync_directory
from multimodal_jev.local_config import resolve
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.tpo_anchor_layout import window_device_plan
from tools.reports.experiment_ledger import LEDGER, git_output, plan, queue_rows

PHASES = ("data", "train", "evaluate")
ACTIVE = ("in_progress", "planned")
REGISTRY = Path("experiments/pipelines")
LOCAL = Path(".local/config/macros.json")
CLAIM_SCHEMA = "jev-experiment-claim-v2"


def digest(raw):
    return sha256(raw).hexdigest()


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def committed(commit, path):
    return git_output("show", f"{commit}:{path}")


def execution_selection(selection, claim):
    for key in ("experiment_id", "plan", "plan_sha256", "pipeline_sha256"):
        if claim[key] != selection[key]:
            raise ValueError(f"Claim identity changed: {key}; register a new pipeline revision")
    if claim["schema"] == CLAIM_SCHEMA:
        return dict(selection, commit=claim["commit"], execution_pinned=True)
    if claim["schema"] != "jev-experiment-selection-v1":
        raise ValueError("Unknown experiment claim schema")
    # Legacy claims did not freeze future commands. Keep any existing
    # command's own source pin without rewriting history or journals.
    return selection


def check_experiment_content(commit, selection):
    for path, key in ((selection["plan"], "plan_sha256"),
                      (REGISTRY / f"{selection['experiment_id']}.json", "pipeline_sha256")):
        if digest(committed(commit, path)) != selection[key]:
            raise ValueError(f"Committed experiment changed: {path}")


def lookup(value, fields):
    for field in fields:
        value = value[field]
    return value


def resolve_step(template, macros):
    try:
        return resolve(template, macros), None
    except ValueError as exc:
        if not str(exc).startswith("Missing local macro:"):
            raise
        return None, str(exc)


def attest(checks):
    """Read the result of the backend's independent audit and its identity pins."""
    proofs, missing = [], []
    for check in checks:
        path = Path(check["path"])
        if not path.is_absolute():
            raise ValueError("Evidence requires an absolute path")
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            missing.append(path.name)
            continue
        identity = digest(raw)
        if "sha256" in check and identity != check["sha256"]:
            raise ValueError(f"Evidence digest differs: {path.name}")
        value = json.loads(raw)
        for fields, expected in check.get("equals", []):
            try:
                actual = lookup(value, fields)
            except (KeyError, TypeError):
                raise ValueError(f"Evidence field missing: {path.name} {fields}") from None
            if fingerprint(actual) != fingerprint(expected):
                raise ValueError(f"Evidence field differs: {path.name} {fields}")
        proofs.append(dict(file=path.name, sha256=identity, value=value))
    return proofs, missing


def validate(spec, selection):
    if (spec["schema"] != "jev-experiment-package-v1"
            or spec["protocol"] != selection["plan"]):
        raise ValueError("Pipeline schema or ledger protocol differs")
    if "revision" in spec and not re.fullmatch(r"[a-z0-9][a-z0-9-]*", spec["revision"]):
        raise ValueError("Invalid revision for the experiment state directory")
    if set(spec["stages"]) != set(PHASES):
        raise ValueError("Register the data, train and evaluate stages")
    ids = set()
    for phase in PHASES:
        if not spec["stages"][phase]:
            raise ValueError(f"Empty stage: {phase}")
        for step in spec["stages"][phase]:
            name = step["id"]
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) or name in ids:
                raise ValueError("Step identifiers must be unique")
            ids.add(name)
            if "pending" in step:
                if set(step) != {"id", "pending"}:
                    raise ValueError("A pending step only has a reason")
                continue
            if not step.get("checks") or any(
                not (check.get("sha256") or check.get("equals")) for check in step["checks"]
            ):
                raise ValueError(f"Register independent evidence checks for {name}")
            if phase != "evaluate" and step.get("panels"):
                raise ValueError("Only evaluation steps can declare panels")
            if "run" in step and not step["run"]:
                raise ValueError(f"Empty command list for {name}")
            for panel in step.get("panels", []):
                if (type(panel["population"]) is not int or panel["population"] <= 0
                        or any(not 0 <= arm["proof"] < len(step["checks"]) for arm in panel["arms"])):
                    raise ValueError(f"Invalid evaluation panel: {panel['id']}")
    if not any(step.get("panels") or "pending" in step for step in spec["stages"]["evaluate"]):
        raise ValueError("Register evaluation panel coverage")


def check_commands(commands, checkout):
    for module, *args in commands:
        if not re.fullmatch(r"tools(?:\.[a-zA-Z_][a-zA-Z_0-9]*){2,}", module):
            raise ValueError(f"Expected a repository tool module: {module}")
        if any(not isinstance(arg, str) for arg in args):
            raise ValueError("Arguments must be strings")
        # The selected tool and its dependencies are read from the committed checkout.
        if not (checkout / (module.replace(".", "/") + ".py")).is_file():
            raise ValueError(f"Tool is absent from committed source: {module}")
        if any(a in {"--detach", "detach"} for a in args):
            raise ValueError("Backends must run in the foreground inside the package driver")


def run_commands(step, state, lock, selection, binding, resources):
    commands = step["run"]
    intent = state / f"{step['id']}.intent.json"
    python = Path(step.get("python", sys.executable))
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("A local backend Python interpreter is required")
    # Pack from the selected commit; another developer can keep editing the main checkout.
    with TemporaryDirectory(prefix=".source-", dir=state) as directory:
        source = Path(directory)
        with TemporaryDirectory(prefix=".archive-", dir=state) as staging:
            archive = Path(staging) / "source.tar"
            subprocess.run(["git", "archive", "--format=tar", "-o", str(archive), selection["commit"]],
                           check=True)
            with tarfile.open(archive) as tar:
                tar.extractall(source, filter="data")
        check_commands(commands, source)
        journal = dict(schema="experiment-command-intent-v2", commit=selection["commit"],
                       binding_sha256=binding, commands=commands, python=str(python))
        if intent.exists():
            if json.loads(intent.read_bytes()) != journal:
                raise ValueError("Cannot resume: source, command, interpreter or bindings changed")
        else:
            commit_json(intent, journal)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (source / "src", source))))
        with (state / f"{step['id']}.log").open("a") as log:
            if resources:
                print(json.dumps(resources), file=log, flush=True)
            code = 0
            for index, (module, *args) in enumerate(commands):
                prefix = state / f"{step['id']}.command-{index}"
                started, returned = Path(str(prefix)+".started"), Path(str(prefix)+".returned")
                if returned.exists():
                    code = json.loads(returned.read_bytes())["exit_code"]
                    if code:
                        break
                    continue
                if started.exists():
                    raise ValueError(f"Command {index} may have run; collect the original backend handle before resuming")
                result = subprocess.run([str(python), "-B", "-m", "tools.reports.experiment_package",
                                         "_backend", str(started), str(returned), binding, str(index),
                                         str(lock.fileno()), module, *args],
                                        cwd=source, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                        env=env, pass_fds=(lock.fileno(),))
                code = result.returncode
                if not returned.exists() or json.loads(returned.read_bytes())["exit_code"] != code:
                    raise ValueError(f"Backend wrapper interrupted at command {index}; use resume")
                if code:
                    break
        commit_json(state / f"{step['id']}.exit.json", dict(exit_code=code))
        return code == 0


def evaluate_panel(panel, proofs):
    rows = []
    for arm in panel["arms"]:
        value = lookup(proofs[arm["proof"]]["value"], arm["path"])
        if type(value["queries"]) is not int or value["queries"] != panel["population"]:
            raise ValueError(f"{panel['id']}: evaluation population differs")
        policy = arm["policy"]
        if isinstance(policy, dict):
            policy = lookup(proofs[policy["proof"]]["value"], policy["path"])
        observed_policy = (lookup(proofs[arm["proof"]]["value"], arm["policy_field"])
                           if "policy_field" in arm else value["policy_id"])
        if not isinstance(policy, str) or not policy or observed_policy != policy:
            raise ValueError(f"{panel['id']}: policy identity differs")
        metrics = {}
        for label, key in arm["metrics"].items():
            number = value[key]
            if type(number) not in (float, int) or not math.isfinite(number) or not 0 <= number <= 1:
                raise ValueError(f"{panel['id']}: invalid metric {key}")
            metrics[label] = number
        rows.append(dict(label=arm["label"], policy_id=policy, metrics=metrics))
    if len(rows) < 2 or len({row["label"] for row in rows}) != len(rows):
        raise ValueError(f"{panel['id']}: target and distinct comparisons required")
    primary = panel["primary"]
    target, baseline, *_ = rows
    delta = target["metrics"][primary] - baseline["metrics"][primary]
    return dict(id=panel["id"], scope=panel["scope"], population=panel["population"],
                primary=primary, delta=delta, rows=rows)


def step_result(template, macros, state, selection, lock, upstream, resume=False):
    name = template["id"]
    resolved, issue = resolve_step(template, macros)
    if issue:
        return dict(id=name, status="unbound", detail=issue)
    step, _ = resolved
    intent = state / f"{name}.intent.json"
    journal = json.loads(intent.read_bytes()) if intent.is_file() else None
    if journal and journal.get("schema") == "experiment-command-intent-v2":
        if selection.get("execution_pinned") and journal["commit"] != selection["commit"]:
            raise ValueError(f"Dispatched source does not match the frozen experiment: {name}")
        selection = dict(selection, commit=journal["commit"])

    done = state / f"{name}.audit.json"
    prerequisites, unready = attest(step.get("inputs", []))
    if unready:
        return dict(id=name, status="awaiting_evidence", detail="、".join(unready))
    # Some immutable input identities only exist after an earlier step (e.g. this
    # endpoint’s mined TRAIN view). Inject the hash actually read by attest()
    # into the saved backend configuration; never accept a stale local override.
    for key, index in step.get("input_sha256_macros", {}).items():
        if (not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or key in macros or not step.get("config")
                or type(index) is not int or not 0 <= index < len(prerequisites)):
            raise ValueError("Invalid or overridden prerequisite digest: " + key)
        macros = dict(macros, **{key: prerequisites[index]["sha256"]})
    config, config_hash = None, None
    resource_plan = {}
    if "config" in step:
        template_config = json.loads(committed(selection["commit"], step["config"]))
        resolved, issue = resolve_step(template_config, macros)
        if issue:
            return dict(id=name, status="unbound", detail=issue)
        config, used = resolved
        config_hash = fingerprint(config)
    for module, *args in step.get("run", []):
        if module in {"tools.reports.prepare_tpo_windows", "tools.reports.run_full_joint_tpo"}:
            if config is None or args != [step["config"], macros["EXPERIMENT_LOCAL_CONFIG"]]:
                raise ValueError("Register the actual TPO package template as the step config")
            if config.get("joint") or config.get("full_joint"):
                resource_plan["device_plan"] = window_device_plan(config)
    binding = fingerprint([step, config_hash, upstream, [p["sha256"] for p in prerequisites]])
    pinned = json.loads(done.read_bytes()) if done.is_file() else None
    proofs, missing = attest(step["checks"])
    if pinned is not None and missing:
        raise ValueError(f"Previously audited evidence missing for {name}")
    terminal = state / f"{name}.exit.json"
    if journal is not None and journal["binding_sha256"] != binding:
        raise ValueError(f"Previously dispatched step binding differs: {name}")
    guards = [Path(path) for path in step.get("dispatch_guards", [])]
    if any(not path.is_absolute() for path in guards):
        raise ValueError("Dispatch guards require absolute paths")
    occupied = [path.name for path in guards if path.exists()]
    owned = step.get("owned_checks", range(len(step["checks"])))
    if (step.get("requires_current_dispatch") and not intent.exists()
            and (occupied or any(Path(step["checks"][i]["path"]).exists() for i in owned))):
        raise ValueError(f"{name}: unclaimed output in this revision; bind a fresh output directory")
    if missing and occupied and not intent.exists():
        return dict(id=name, status="awaiting_evidence", detail="Existing output: " + "、".join(occupied),
                    **resource_plan)
    recoverable = (resume and journal is not None and not terminal.exists()
                   and journal.get("schema") == "experiment-command-intent-v2")
    if missing and recoverable and occupied and not (state / f"{name}.command-0.started").exists():
        raise ValueError("Output appeared before this command started: " + ", ".join(occupied))
    if missing and step.get("run") and lock is not None and (not intent.exists() or recoverable):
        if config is not None:
            config_path = state / f"{name}.macros.json"
            resolved = {"schema": "jev-local-macros-v1", "macros": {key: macros[key] for key in used}}
            if config_path.exists():
                if json.loads(config_path.read_bytes()) != resolved:
                    raise ValueError(f"Saved local bindings changed for {name}")
            else:
                # Persist only the macros referenced by the registered template.
                fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as out:
                    json.dump(resolved, out)
                    out.flush()
                    os.fsync(out.fileno())
                sync_directory(state)
            step["run"] = [[str(config_path) if arg == macros["EXPERIMENT_LOCAL_CONFIG"] else arg for arg in command]
                           for command in step["run"]]
        if run_commands(step, state, lock, selection, binding, resource_plan):
            proofs, missing = attest(step["checks"])
        else:
            return dict(id=name, status="failed", detail=f"See {name}.log", **resource_plan)
    if not missing and journal is not None and not terminal.exists():
        if journal.get("schema") == "experiment-command-intent-v2":
            count = len(journal["commands"])
            if not all((state / f"{name}.command-{i}.started").exists() for i in range(count)):
                raise ValueError(f"{name}: outputs appeared before the registered commands started")
            for i in range(count):
                returned = state / f"{name}.command-{i}.returned"
                if returned.exists() and json.loads(returned.read_bytes())["exit_code"] != 0:
                    raise ValueError(f"{name}: command {i} failed; inspect the original backend")
    if terminal.exists() and json.loads(terminal.read_bytes())["exit_code"] != 0:
        return dict(id=name, status="failed", detail=f"See {name}.log", **resource_plan)
    if missing:
        if resume and intent.exists() and not terminal.exists() and not recoverable:
            return dict(id=name, status="failed", detail="Legacy intent lacks a command journal; collect the original backend handle", **resource_plan)
        if intent.exists() and terminal.exists():
            status = "failed"
        elif intent.exists() or not step.get("run"):
            status = "awaiting_evidence"
        else:
            status = "ready"
        return dict(id=name, status=status, detail="、".join(missing), **resource_plan)

    for output, input_ in step.get("lineage", []):
        if lookup(proofs, output) != lookup(prerequisites, input_):
            raise ValueError(f"{name}: evidence does not match prerequisite {input_}: {output}")
    for i, check in enumerate(step["checks"]):
        if check.get("verify_files"):
            from multimodal_jev.baseline import file_hash
            base = Path(check["path"]).parent
            files = proofs[i]["value"]["files"]
            from multimodal_jev.context_execution import map_io
            def verify(item):
                key, expected = item
                return Path(key).name == key and not (base/key).is_symlink() and file_hash(base/key) == expected
            valid, _ = map_io(verify, files.items(), workers=lookup(config,check["hash_workers"]))
            if set(files) != set(check["verify_files"]) or not all(valid):
                raise ValueError(f"{name}: published file hash differs")
    evidence = [{k: r[k] for k in ("file", "sha256")} for r in proofs]
    for check, proof, identity in zip(step["checks"], proofs, evidence, strict=True):
        if "cite" in check:
            identity["file"] = check["cite"]
        if "report" in check:
            identity["fields"] = {label: lookup(proof["value"], path)
                                  for label, path in check["report"].items()}
    result = dict(id=name, status="audited", evidence=evidence,
                  mode="executed" if intent.exists() else "reused",
                  panels=[evaluate_panel(p, proofs) for p in step.get("panels", [])])
    receipt = dict(binding_sha256=binding, result=result)
    if pinned is not None and pinned != receipt:
        raise ValueError(f"Previously audited stage changed: {name}")
    if lock is not None and pinned is None:
        commit_json(done, receipt)
    return dict(result, **resource_plan)


def workflow(spec, macros, state, selection, lock=None, resume=False, through=None):
    results, upstream, downstream = [], [], True
    stopped = False
    for phase in PHASES:
        phase_results = []
        for step in spec["stages"][phase]:
            if stopped:
                result = dict(id=step["id"], status="outside_requested_scope")
            elif "pending" in step:
                result = dict(id=step["id"], status="pending", detail=step["pending"])
            elif not downstream:
                result = dict(id=step["id"], status="waiting_for_dependency")
            else:
                try:
                    result = step_result(step, macros, state, selection, lock, upstream, resume)
                except (KeyError, IndexError, TypeError, ValueError, OSError,
                        subprocess.CalledProcessError) as exc:
                    result = dict(id=step["id"], status="failed", detail=str(exc))
            phase_results.append(dict(stage=phase, **result))
            if step["id"] == through:
                stopped = True
            if result["status"] == "failed" or (spec.get("ordered_steps") and result["status"] != "audited"):
                downstream = False
        downstream = downstream and all(row["status"] == "audited" for row in phase_results)
        if downstream:
            upstream.extend([[row["id"], [p["sha256"] for p in row["evidence"]]] for row in phase_results])
        results.extend(phase_results)
    return results


def table_cell(text):
    return str(text).replace("|", "&#124;").replace("\n", " ")


def report_block(claim, selection, results, status):
    lines = [
        "## 实验入口执行记录", "",
        f"队列状态 **{status}**。领取提交 `{claim['commit']}`，"
        f"台账行 SHA-256 `{claim['row_sha256']}`，"
        f"协议 SHA-256 `{claim['plan_sha256']}`。",
        f"执行提交 `{selection.get('execution_commit', selection['commit'])}`，"
        f"配置 SHA-256 `{selection['pipeline_sha256']}`。", "",
        "| 阶段 | 步骤 | 状态 | 执行方式 | 证据或待办 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in results:
        items = []
        for evidence in row.get("evidence", []):
            fields = "、".join(f"{k}={v}" for k, v in evidence.get("fields", {}).items())
            items.append(f"`{evidence['file']}` SHA-256 `{evidence['sha256']}` {fields}".strip())
        detail = table_cell("；".join(items) or row.get("detail", ""))
        lines.append(f"| {row['stage']} | {row['id']} | {row['status']} | {row.get('mode', '—')} | {detail} |")
    for row in results:
        for panel in row.get("panels", []):
            lines += ["", f"### {panel['id']}", "",
                      f"{panel['scope']}。配对人口 {panel['population']:,}。",
                      "", "| 策略 | 指标 |", "| --- | --- |"]
            for arm in panel["rows"]:
                metrics = "、".join(f"{key} {value:.2%}" for key, value in arm["metrics"].items())
                lines.append(f"| {arm['label']} | {metrics} |")
            target, baseline, *others = panel["rows"]
            for comparison in (baseline, *others):
                deltas = "、".join(f"{metric} {(v - comparison['metrics'][metric]) * 100:+.2f} 个百分点"
                                   for metric, v in target["metrics"].items()
                                   if metric in comparison["metrics"])
                lines += ["", f"相对 {comparison['label']}，{deltas}。"]
            direction = "上升" if panel["delta"] > 0 else "下降" if panel["delta"] < 0 else "持平"
            lines += ["", f"相对主对照，{panel['scope']}的 {panel['primary']} {direction}。"]
    if status != "complete":
        lines += ["", "尚有预登记项目待审核，研究结论待完整证据。"]
    return "\n".join(lines) + "\n"


def write_wiki(path, before, after):
    if after == before:
        return
    check_paths({"wiki": path}, min_free_bytes=0, min_free_fraction=0)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as tmp:
        temp = Path(tmp.name)
        try:
            tmp.write(after.encode())
            tmp.flush()
            os.fchmod(tmp.fileno(), 0o644)
            os.fsync(tmp.fileno())
            current = path.read_text() if path.exists() else ""
            if current != before:
                raise ValueError(f"Concurrent Wiki edit: {path}")
            os.replace(temp, path)
            sync_directory(path.parent)
        finally:
            temp.unlink(missing_ok=True)


def update_wiki(claim, selection, results, status):
    check_experiment_content(git_output("rev-parse", "HEAD").decode().strip(), selection)
    ledger = Path(LEDGER)
    original = ledger.read_text()
    experiment = selection["experiment_id"]
    report = Path("wiki/reports") / f"{experiment}.md"
    row = queue_rows(original)[experiment]
    if row["plan"] != selection["plan"] or row["queue_status"] not in (*ACTIVE, "complete"):
        raise ValueError("Working Wiki row no longer matches the experiment")
    if any(row[key] != selection["working_row"][key] for key in ("priority", "queue_status", "plan")):
        raise ValueError("Working Wiki row changed during execution")
    notes = "全部登记步骤已审核" if status == "complete" else next(
        f"{r['id']} {r['status']}" for r in results if r["status"] != "audited")
    notes += f"。[执行报告](reports/{report.name})"
    lines = original.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if digest(line.rstrip("\r\n").encode()) == row["row_sha256"]:
            columns = line.split("|")
            columns[3], columns[5] = f" {status} ", f" {notes} "
            lines[index] = "|".join(columns)
            break

    before = report.read_text() if report.exists() else ""
    begin = "<!-- experiment-package:begin -->"
    end = "<!-- experiment-package:end -->"
    block = f"{begin}\n{report_block(claim, selection, results, status)}{end}"
    if before.count(begin) != before.count(end) or before.count(begin) > 1:
        raise ValueError("Invalid generated report markers")
    if begin in before:
        start = before.index(begin)
        stop = before.index(end, start) + len(end)
        if start > before.index(end):
            raise ValueError("Invalid generated report marker order")
        after = before[:start] + block + before[stop:]
    else:
        prefix = before.rstrip() if before else f"# {experiment}\n\n[实验台账](../experiment-ledger.md)"
        after = prefix + "\n\n" + block + "\n"
    write_wiki(report, before, after)
    write_wiki(ledger, original, "".join(lines))


def execute(action, experiment, local, *, publish=False, wait_for_ledger=False, through=None):
    if action == "publish":
        if not experiment or not re.fullmatch(r"JEV-[A-Z0-9]+(?:-[A-Z0-9]+)*", experiment):
            raise ValueError("Specify --experiment for Wiki publication")
        selection = plan(experiment, states=(*ACTIVE, "complete", "blocked", "failed", "stopped"))
        raw = committed(selection["commit"], REGISTRY / f"{experiment}.json")
        selection = dict(selection, pipeline_sha256=digest(raw))
        spec = json.loads(raw)
        binding = json.loads(local.read_bytes())
        workspace, _ = resolve(spec["state_root"], binding["macros"])
        with (Path(workspace) / "ledger.lock").open("r") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            publish_wiki(experiment, selection=selection)
        return dict(experiment_id=experiment, status="published")
    selection = plan(experiment, states=(*ACTIVE, "complete", "blocked", "failed", "stopped")
                     if action == "status" and experiment else ACTIVE)
    if selection["status"] != "selected":
        return dict(selection, status="no_eligible_experiment")
    path = REGISTRY / f"{selection['experiment_id']}.json"
    if not git_output("ls-tree", selection["commit"], "--", str(path)):
        return dict(selection, status="unbound", detail=f"Register {path}")
    raw = committed(selection["commit"], path)
    selection = dict(selection, pipeline_sha256=digest(raw))
    spec = json.loads(raw)
    validate(spec, selection)
    if through is not None and (action not in {"run", "resume"} or
            through not in {s["id"] for group in spec["stages"].values() for s in group}):
        raise ValueError("Unknown or inapplicable --through step")
    bindings = json.loads(local.read_bytes())
    if bindings["schema"] != "jev-local-macros-v1":
        raise ValueError("Expected jev-local-macros-v1")
    macros = bindings["macros"]
    # Route reusable operation templates to this experiment’s output families.
    # A historical local alias must never redirect a new run.
    aliases = spec.get("operation_bindings", {})
    if not isinstance(aliases, dict) or set(aliases) & (set(macros) | {"EXPERIMENT_LOCAL_CONFIG"}):
        raise ValueError("Local macros must not override experiment operation bindings")
    macros = dict(macros, **aliases, EXPERIMENT_LOCAL_CONFIG=str(local.resolve()))
    workspace, _ = resolve(spec["state_root"], macros)
    root = Path(workspace)
    if not root.is_absolute():
        raise ValueError("The local state root must be absolute")
    state_name = selection["experiment_id"]
    if "revision" in spec:
        state_name += "." + spec["revision"]
    state = root / state_name
    claim_path = state / "claim.json"
    if action in {"status", "plan"}:
        execution = execution_selection(selection, json.loads(claim_path.read_bytes())) if claim_path.is_file() else selection
        results = workflow(spec, macros, state, execution)
        return dict(selection, execution_commit=execution["commit"], status="read_only", steps=results if action == "status" else [
            {k: v for k, v in step.items() if k not in ("evidence", "panels")} for step in results])

    check_paths({"experiment state": root})
    root.mkdir(parents=True, exist_ok=True)
    # The inherited lock stays held if the launcher dies before its backend exits.
    with (root / "ledger.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if wait_for_ledger else fcntl.LOCK_NB))
        check_experiment_content(git_output("rev-parse", "HEAD").decode().strip(), selection)
        work_row = queue_rows(Path(LEDGER).read_text())[selection["experiment_id"]]
        if work_row["plan"] != selection["plan"] or work_row["queue_status"] not in (*ACTIVE, "complete"):
            raise ValueError("Working Wiki row is no longer eligible")
        state.mkdir(exist_ok=True)
        if work_row["queue_status"] == "complete" and not claim_path.is_file():
            raise ValueError("Working Wiki row is complete; sync its report before continuing")
        if action == "resume" and work_row["queue_status"] == "planned" and not claim_path.is_file():
            raise ValueError("Use run to claim a planned experiment")
        if not claim_path.exists():
            if work_row["row_sha256"] != selection["row_sha256"]:
                raise ValueError("Experiment queue row changed before claim; select it again")
            commit_json(claim_path, dict(selection, schema=CLAIM_SCHEMA))
        claim = json.loads(claim_path.read_bytes())
        execution = execution_selection(selection, claim)
        report_selection = dict(selection, execution_commit=execution["commit"])
        if work_row["queue_status"] == "planned":
            starting = [dict(id="claim", stage="ledger", status="claimed")]
            update_wiki(claim, dict(report_selection, working_row=work_row),
                        starting, "in_progress")
            work_row = queue_rows(Path(LEDGER).read_text())[selection["experiment_id"]]
        results = workflow(spec, macros, state, execution, lock, resume=action == "resume", through=through)
        statuses = {r["status"] for r in results}
        status = "complete" if statuses == {"audited"} else "in_progress"
        result = dict(selection, execution_commit=execution["commit"], status=status, steps=results, published=False)
        try:
            update_wiki(claim, dict(report_selection, working_row=work_row), results, status)
            if publish:
                publish_wiki(selection["experiment_id"], selection=selection)
                result["published"] = True
        except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
            result["report_error"] = str(exc)
        return result


def wiki_row(markdown, row):
    return next(line for line in markdown.splitlines() if digest(line.encode()) == row["row_sha256"])


def publish_wiki(experiment, *, selection=None):
    """Publish only this experiment's generated report and the queue update."""
    if git_output("branch", "--show-current").strip() != b"main":
        raise ValueError("Publish the Wiki from the main branch")
    subprocess.run(["git", "fetch", "origin", "main"], check=True, stdout=sys.stderr)
    remote = git_output("rev-parse", "origin/main").decode().strip()
    try:
        return publish_wiki_revision(experiment, remote, selection)
    except subprocess.CalledProcessError as exc:
        if exc.cmd[:3] != ["git", "push", "origin"]:
            raise
        subprocess.run(["git", "fetch", "origin", "main"], check=True, stdout=sys.stderr)
        latest = git_output("rev-parse", "origin/main").decode().strip()
        if latest == remote:
            raise
        return publish_wiki_revision(experiment, latest, selection)


def publish_wiki_revision(experiment, remote, selection):
    head = git_output("rev-parse", "HEAD").decode().strip()
    if selection:
        check_experiment_content(remote, selection)
    report = f"wiki/reports/{experiment}.md"
    files = [LEDGER, report]
    work = Path(LEDGER).read_text()
    base = committed(head, LEDGER).decode()
    old, new = queue_rows(base), queue_rows(work)
    if (experiment not in old or set(old) != set(new)
            or any(old[k] != new[k] for k in old if k != experiment)
            or not Path(report).is_file()
            or "<!-- experiment-package:begin -->" not in Path(report).read_text()):
        raise ValueError("Publish only the selected generated report and queue row")
    def without_row(markdown, row):
        return [line for line in markdown.splitlines() if digest(line.encode()) != row["row_sha256"]]
    if without_row(base, old[experiment]) != without_row(work, new[experiment]):
        raise ValueError("Other uncommitted edits exist in the ledger")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", remote, head], check=False)
    if ancestor.returncode == 1:
        return publish_on_remote(experiment, head, remote, report, work, new[experiment])
    ancestor.check_returncode()
    if not set(git_output("diff", "--name-only", remote, head).decode().splitlines()) <= set(files):
        raise ValueError("Sync main before publishing the Wiki")
    if git_output("status", "--porcelain", "--", *files):
        subprocess.run(["git", "add", "--", report], check=True)
        subprocess.run(["git", "commit", "--only", "-m", f"Update {experiment} experiment report", "--", *files],
                       check=True, stdout=sys.stderr)
    subprocess.run(["git", "push", "origin", "HEAD:refs/heads/main"], check=True, stdout=sys.stderr)


def publish_on_remote(experiment, head, remote, report, work, desired_row):
    """Apply only this experiment's changes when remote main has advanced."""
    common = git_output("merge-base", head, remote).decode().strip()
    # Keep the receipt outside the branch: a later publication from this same
    # checkout can advance its own report without treating it as a remote edit.
    publication_ref = f"refs/jev/wiki-publications/{experiment}"
    saved = subprocess.run(["git", "rev-parse", "--verify", "--quiet", publication_ref],
                           capture_output=True, text=True)
    base = common
    if saved.returncode == 0 and subprocess.run(
            ["git", "merge-base", "--is-ancestor", saved.stdout.strip(), remote]).returncode == 0:
        base = saved.stdout.strip()
    latest = committed(remote, LEDGER).decode()
    remote_row = wiki_row(latest, queue_rows(latest)[experiment])
    wanted_row = wiki_row(work, desired_row)

    def report_at(commit):
        return committed(commit, report) if git_output("ls-tree", commit, "--", report) else None

    wanted_report = Path(report).read_bytes()
    remote_report = report_at(remote)
    previous = committed(base, LEDGER).decode()
    old_row = wiki_row(previous, queue_rows(previous)[experiment])
    if remote_row not in (old_row, wanted_row):
        raise ValueError("Remote Wiki row changed; reconcile this experiment before publishing")
    if remote_report not in (report_at(base), wanted_report):
        raise ValueError("Remote Wiki report changed; reconcile this experiment before publishing")
    updates = {}
    if remote_row != wanted_row:
        updates[LEDGER] = re.sub(r"^" + re.escape(remote_row) + r"$", lambda _: wanted_row,
                                 latest, count=1, flags=re.MULTILINE).encode()
    if remote_report != wanted_report:
        updates[report] = wanted_report
    if not updates:
        subprocess.run(["git", "update-ref", publication_ref, remote], check=True)
        return
    with TemporaryDirectory(prefix="jev-wiki-index-") as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        subprocess.run(["git", "read-tree", remote], env=env, check=True)
        for path, payload in updates.items():
            blob = subprocess.check_output(["git", "hash-object", "-w", "--stdin"], input=payload).decode().strip()
            subprocess.run(["git", "update-index", "--add", "--cacheinfo", "100644", blob, path],
                           env=env, check=True)
        tree = subprocess.check_output(["git", "write-tree"], env=env, text=True).strip()
        revision = git_output("commit-tree", tree, "-p", remote,
                              "-m", f"Update {experiment} experiment report").decode().strip()
        subprocess.run(["git", "push", "origin", f"{revision}:refs/heads/main"],
                       check=True, stdout=sys.stderr)
        subprocess.run(["git", "update-ref", publication_ref, revision], check=True)


def main():
    if sys.argv[1:2] == ["_backend"]:
        # The child owns both journal writes and inherits the lease. Killing the coordinator
        # cannot leave a normally returning backend without its exit receipt.
        started, returned, binding, index, fd, module, *args = sys.argv[2:]
        commit_json(Path(started), dict(index=int(index), binding_sha256=binding))
        code = subprocess.run([sys.executable, "-B", "-m", module, *args], pass_fds=(int(fd),)).returncode
        code = code if code >= 0 else 128-code
        commit_json(Path(returned), dict(exit_code=code))
        return code
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "status", "run", "resume", "publish", "capacity", "score-shard", "score-merge"), nargs="?", default="plan")
    parser.add_argument("--experiment", help="Experiment ID in the committed queue")
    parser.add_argument("--scoring", action="store_true", help="Use the registered scoring capacity window")
    parser.add_argument("--capacity-variant", help="Use a named training capacity variant from the experiment package")
    parser.add_argument("--local", type=Path, default=LOCAL)
    parser.add_argument("--publish", action="store_true", help="Sync the generated Wiki changes to origin/main")
    parser.add_argument("--through", help="Run through this registered step inclusive, then stop before later steps")
    parser.add_argument("--wait-for-ledger", action="store_true", help="Wait for the current experiment to release the ledger")
    args = parser.parse_args()
    if args.through and args.action not in {"run", "resume"}:
        parser.error("--through requires run or resume")
    if args.publish and args.action not in {"run", "resume"}:
        parser.error("--publish is available with run or resume")
    if args.scoring and args.action != "capacity":
        parser.error("--scoring is available with capacity")
    if args.capacity_variant and (args.action != "capacity" or args.scoring):
        parser.error("--capacity-variant requires training capacity")
    try:
        if args.action in {"score-shard", "score-merge"}:
            from tools.reports.card_scoring_shards import execute as score_shard, merge as score_merge
            selected = plan(args.experiment, states=ACTIVE)
            pipeline = json.loads(committed(selected["commit"], REGISTRY/(selected["experiment_id"]+".json")))
            template = Path(pipeline["scoring_" + args.action.split("-", 1)[1]])
            if template.read_bytes() != committed(selected["commit"], template):
                raise ValueError("Scoring template differs from the committed ledger")
            handler = score_shard if args.action == "score-shard" else score_merge
            print(json.dumps(handler(template, args.local), indent=2))
            return 0
        if args.action == "capacity":
            if args.scoring:
                from tools.reports.prepare_card_scoring_benchmark import capacity
            else:
                from tools.reports.card_capacity_sweep import capacity
            params = {"variant": args.capacity_variant} if args.capacity_variant else {}
            print(json.dumps(capacity(args.experiment, args.local, **params), indent=2))
            return 0
        result = execute(args.action, args.experiment, args.local, publish=args.publish,
                         wait_for_ledger=args.wait_for_ledger, through=args.through)
    except (KeyError, OSError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int("report_error" in result or result["status"] == "unbound" or any(
        step["status"] in {"failed", "unbound"} for step in result.get("steps", [])))


if __name__ == '__main__':
    sys.exit(main())
