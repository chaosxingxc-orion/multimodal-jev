#!/usr/bin/env python3
"""Full-denominator MMAR scoring using the publisher's unchanged match function."""
import argparse
import ast
import json
from pathlib import Path
import re
from multimodal_jev.baseline import file_hash
from multimodal_jev.mmar import score


def publisher_match(path):
    # Load only the unchanged pure function; the publisher CLI imports optional
    # tqdm/pickle and skips missing predictions. Neither behavior is needed here.
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'string_match']
    if len(nodes) != 1:
        raise ValueError('publisher match function missing or ambiguous')
    namespace = {'re': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['string_match']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--publisher-scorer', type=Path, required=True)
    p.add_argument('--prediction-format', choices=('probabilities', 'raw-text'), default='probabilities')
    args = p.parse_args()
    manifest = json.loads((args.release / 'MANIFEST.json').read_text())
    references = args.release / 'references.json'
    if file_hash(references) != manifest['source_sha256']:
        raise ValueError('references changed')
    if file_hash(args.publisher_scorer) != manifest['publisher_scorer_sha256']:
        raise ValueError('publisher scorer changed')
    scorer = score
    if args.prediction_format == 'raw-text':
        from multimodal_jev.mmar_omni import score_text
        scorer = score_text
    result = scorer(json.loads(references.read_text()),
                   [json.loads(s) for s in args.predictions.read_text().splitlines()],
                   publisher_match(args.publisher_scorer))
    print(json.dumps({**result, 'release_id': manifest['release_id'], 'prediction_format': args.prediction_format,
                      'predictions_sha256': file_hash(args.predictions),
                      'scorer_sha256': file_hash(args.publisher_scorer)}, indent=2))
