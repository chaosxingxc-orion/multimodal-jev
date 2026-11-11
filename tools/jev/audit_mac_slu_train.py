#!/usr/bin/env python3
"""Audit local MAC-SLU TRAIN without opening dev/test labels."""
import argparse
import json
from pathlib import Path
import tarfile

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.mac_slu_audit import audit_audio_members, audit_rows


def run(root, output):
    labels = root / 'label/train_set.jsonl'
    audio = root / 'audio_train.tar.gz'
    rows = [json.loads(line) for line in labels.open()]
    report = audit_rows(rows)
    with tarfile.open(audio, 'r|gz') as archive:
        report['audio'] = audit_audio_members(archive, [r['id'] for r in rows])
    report['sources'] = {str(p): file_hash(p) for p in (labels, audio, root / 'README.md')}
    report['code'] = {str(p): file_hash(p) for p in (Path(__file__),
        Path(__file__).resolve().parents[2] / 'src/multimodal_jev/mac_slu_audit.py')}
    report['split'] = 'train'
    report['test_labels_opened'] = False
    atomic_json(output, report)
    print(json.dumps({k: report[k] for k in ('records', 'domain_intent_types',
        'bound_slot_role_types', 'flag_record_counts', 'records_without_empty_containers', 'audio')}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError('new audit output required')
    run(a.root, a.output)
