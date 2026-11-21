"""Read atomic macro commit markers inside an existing scoring container."""

import json
from pathlib import Path
import sys


def committed_batches(journal, *, policy, total, batch_size):
    """Read the durable prefix; a final in-flight batch does not count."""
    completed, batches = 0, 0
    paths = sorted(journal.glob('batch-*'))
    for idx, batch in enumerate(paths):
        name = f'batch-{idx:08d}'
        if batch.name != name or batch.is_symlink() or not batch.is_dir():
            raise ValueError('expected contiguous reference batch journal')
        marker = batch/'COMPLETE.json'
        if not marker.is_file():
            if idx != len(paths)-1:
                raise ValueError('uncommitted batch before reference journal tail')
            break
        value = json.loads(marker.read_text())
        indices = value['completed_indices']
        queries = len(indices)
        if (marker.is_symlink() or queries <= 0 or value['batch_id'] != name
                or value['policy_id'] != policy or value['status'] != 'complete_macro_inference'
                or value['actor_update_performed'] is not False
                or indices != list(range(queries)) or queries > min(batch_size, total-completed)
                or (queries < batch_size and idx != len(paths)-1)):
            raise ValueError('invalid reference batch commit')
        completed += queries
        batches += 1
    return dict(committed_distinct_inputs=completed, configured_source_recordings=total, committed_batches=batches,
                policy_id=policy, first_batch_committed=batches > 0, independent_audit_complete=False)


if __name__ == '__main__':
    journal, policy, total, batch_size = sys.argv[1:]
    print(json.dumps(committed_batches(Path(journal), policy=policy, total=int(total), batch_size=int(batch_size))))
