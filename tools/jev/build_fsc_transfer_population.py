#!/usr/bin/env python3
"""Freeze full official FSC evaluation audio inputs without exporting labels."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.fsc_transfer_population import build_population


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--split', choices=('valid', 'test'), required=True)
    p.add_argument('--source-receipt', type=Path, required=True)
    p.add_argument('--source-receipt-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError('immutable population already exists')
    if file_hash(a.source_receipt) != a.source_receipt_sha256:
        raise ValueError('registered source receipt differs')
    receipt = json.loads(a.source_receipt.read_text())
    population = build_population(a.dataset, a.split, receipt['source_csv_sha256'])
    expected = receipt['splits'][a.split]
    if (population['recordings'] != expected['recordings'] or population['speakers'] != expected['speakers']):
        raise ValueError('official population counts differ')
    a.output.mkdir(parents=True, exist_ok=False)
    atomic_json(a.output/'population.json', population)
    atomic_json(a.output/'MANIFEST.json', dict(population_sha256=file_hash(a.output/'population.json'),
        source_receipt_sha256=a.source_receipt_sha256,
        files={str(Path(__file__).resolve()): file_hash(__file__),
               str(Path(__file__).resolve().parents[2]/'src/multimodal_jev/fsc_transfer_population.py'):
               file_hash(Path(__file__).resolve().parents[2]/'src/multimodal_jev/fsc_transfer_population.py')},
        scope=population['scope']))
    print(json.dumps(dict(output=str(a.output),recordings=population['recordings'],speakers=population['speakers'])))
