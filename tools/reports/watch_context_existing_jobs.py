"""Schedule the existing read-only observer; never dispatch or restart jobs."""
import fcntl
import json
import os
from pathlib import Path
import time

from tools.reports.observe_context_existing_jobs import BASE, INTERVAL, observe


def next_due(result, now):
    if result.get('status') == 'observed_original_handle':
        if not result['state']['Running']:
            return None
    return max(now + 1, result.get('next_ordinary_observation_unix', now + INTERVAL))


def main():
    root = BASE / 'ordinary-observations-v1'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'watcher.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = dict.fromkeys(('grpo', 'inputs'), 0.0)
        status = root / 'watcher-status.json'
        while pending:
            for target, due in list(pending.items()):
                if time.time() < due:
                    continue
                try:
                    result = observe(target)
                except Exception as exc:
                    # Preserve handles and defer errors, including lock contention.
                    result = dict(target=target, status='watcher_observation_error',
                                  error=repr(exc))
                print(json.dumps(result), flush=True)
                due = next_due(result, time.time())
                if due is None:
                    del pending[target]
                else:
                    pending[target] = due
            temporary = status.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(pid=os.getpid(),
                updated_unix=time.time(), pending=pending,
                scope='read_only_original_grpo_and_inputs; no dispatch or restart'), indent=2)+'\n')
            temporary.replace(status)
            if pending:
                time.sleep(max(1, min(60, min(pending.values()) - time.time())))


if __name__ == '__main__':
    main()
