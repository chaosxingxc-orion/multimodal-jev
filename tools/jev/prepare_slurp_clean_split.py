#!/usr/bin/env python3
"""Publish authenticated clean TRAIN source roles; never open TEST targets."""
import argparse
from collections import Counter
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.slurp_clean_split import partition


def prepare(clean, native, output):
    if output.exists():
        raise FileExistsError('preserve prior split')
    receipt = json.loads((clean/'ALIGNMENT-RECEIPT.json').read_text())
    metadata = clean/'metadata/train.jsonl'
    if file_hash(metadata) != receipt['counts']['train']['metadata_sha256']:
        raise ValueError('clean metadata differs')
    manifest = json.loads((native/'MANIFEST.json').read_text())
    requests = native/'train.requests.jsonl'
    if (manifest['identity']['source_sha256']['train'] != receipt['source_hashes']['train']
            or file_hash(requests) != manifest['splits']['train']['requests_sha256']):
        raise ValueError('native source or requests differ')
    result = partition([json.loads(x) for x in metadata.open()],
                       [json.loads(x) for x in requests.open()])
    result['identity'] = dict(clean_metadata_sha256=file_hash(metadata),
        alignment_sha256=file_hash(clean/'ALIGNMENT-RECEIPT.json'),
        native_manifest_sha256=file_hash(native/'MANIFEST.json'),
        requests_sha256=file_hash(requests),
        implementation_sha256=file_hash(Path(__file__).parents[2]/'src/multimodal_jev/slurp_clean_split.py'))
    counts = Counter((r['role'],r['supervision_status']) for r in result['records'])
    result['counts'] = {role: {status:counts[role,status] for status in ('available','conflict')}
                        for role in ('learning','selection')}
    output.mkdir(parents=True)
    (output/'source-split.json').write_text(json.dumps(result,indent=2)+'\n')
    summary = {k:v for k,v in result.items() if k not in ('groups','records','quarantine_ids')}
    summary.update(output=str(output),source_groups=len(result['groups']),
                   recordings=len(result['records']),quarantined=len(result['quarantine_ids']),
                   split_sha256=file_hash(output/'source-split.json'),
                   test_opened=False,scope='TRAIN source closure and conflict quarantine; rule-family split and training release remain pending')
    (output/'receipt.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('clean','native','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.clean,args.native,args.output),indent=2))
