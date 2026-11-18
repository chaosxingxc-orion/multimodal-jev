#!/usr/bin/env python3
"""Stage the complete pinned learning population on local storage before NPU use."""
import argparse
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json
from multimodal_jev.local_audio_mirror import LocalAudioMirror
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.slurp_official_training import load


def run(release, release_sha256, root):
    root = Path(root)
    mirror = LocalAudioMirror(root / 'audio')
    rows, _, _ = load(release, release_sha256, 'learning', purpose='optimize')
    started = time.monotonic()
    records = []
    for index, row in enumerate(rows):
        path = mirror.publish(row['audio'], row['audio_sha256'])
        records.append(dict(id=row['id'], source=row['audio'], sha256=row['audio_sha256'],
                            local=str(path), bytes=path.stat().st_size))
        if (index + 1) % 256 == 0:
            atomic_json(root / 'progress.json', dict(completed=index + 1, total=len(rows),
                seconds=time.monotonic() - started))
    # The immutable publication is created only after the complete population passes.
    report = dict(schema='context-learning-audio-mirror-v1', release_sha256=release_sha256,
                  role='learning', recordings=len(rows), records=records,
                  seconds=time.monotonic() - started,
                  scope='Pinned source bytes mirrored; no processor or training validation implied.')
    commit_json(root / 'MANIFEST.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.release, args.release_sha256, args.output)
    print(result['recordings'], 'learning recordings staged')
