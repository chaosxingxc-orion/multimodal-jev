"""Read a planned experiment from a committed Wiki queue."""
import argparse
from hashlib import sha256
import json
import re
import subprocess


LEDGER = "wiki/experiment-ledger.md"
QUEUE_TITLE = "## 实验队列"
HEADER = "| 实验 ID | 优先级 | 状态 | 协议 | 下一步 |"
STATES = {"planned", "in_progress", "blocked", "complete", "failed", "stopped"}


def git_output(*args):
    return subprocess.check_output(["git", *args])


def queue_rows(markdown):
    sections = re.split(r"(?=^## )", markdown, flags=re.MULTILINE)
    queues = [part for part in sections if part.split("\n", 1)[0] == QUEUE_TITLE]
    if len(queues) != 1:
        raise ValueError("Expected exactly one experiment queue")
    lines = queues[0].splitlines()
    if lines.count(HEADER) != 1:
        raise ValueError("Missing or duplicate experiment queue table")
    table = lines[lines.index(HEADER) + 1:]
    if not table or not re.fullmatch(r"\|(?: *:?-{3,}:? *\|){5}", table[0]):
        raise ValueError("Invalid experiment queue separator")
    rows = {}
    for line in table[1:]:
        if not line.startswith("|"):
            break
        columns = [cell.strip() for cell in line.split("|")]
        if len(columns) != 7 or columns[0] or columns[-1]:
            raise ValueError(f"Invalid experiment queue row: {line}")
        _, experiment, priority, status, plan, next_step, _ = columns
        link = re.fullmatch(r"\[[^\]]+\]\(\.\./(experiments/[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*\.md)(?:#[^)\s]+)?\)", plan)
        if (not re.fullmatch(r"JEV-[A-Z0-9]+(?:-[A-Z0-9]+)*", experiment)
                or experiment in rows or not re.fullmatch(r"[0-9]+", priority)
                or status not in STATES or link is None or not next_step):
            raise ValueError(f"Invalid or duplicate experiment queue row: {line}")
        path = link[1]
        rows[experiment] = {
            "experiment_id": experiment,
            "priority": int(priority),
            "queue_status": status,
            "plan": path,
            "next_step": next_step,
            "row_sha256": sha256(line.encode("utf-8")).hexdigest(),
        }
    return rows


def plan(experiment=None):
    commit = git_output("rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    rows = queue_rows(git_output("show", f"{commit}:{LEDGER}").decode("utf-8"))
    receipt = {"schema": "jev-experiment-selection-v1", "ledger": LEDGER, "commit": commit}
    if experiment is not None:
        if experiment not in rows:
            raise ValueError(f"Experiment is absent from the queue: {experiment}")
        selected = rows[experiment]
        if selected["queue_status"] != "planned":
            raise ValueError(f"{experiment} has queue status {selected['queue_status']}, expected planned")
    else:
        candidates = (row for row in rows.values() if row["queue_status"] == "planned")
        selected = min(candidates, key=lambda row: (row["priority"], row["experiment_id"]), default=None)
        if selected is None:
            return dict(receipt, status="no_planned_experiment")

    plan_content = git_output("cat-file", "blob", f"{commit}:{selected['plan']}")
    if not plan_content.strip():
        raise ValueError("Experiment plan must be a nonempty committed file")
    return dict(receipt, status="selected", plan_sha256=sha256(plan_content).hexdigest(), **selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan"])
    parser.add_argument("--experiment", help="Select a specific planned experiment ID")
    args = parser.parse_args()
    try:
        receipt = plan(args.experiment)
    except (ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
