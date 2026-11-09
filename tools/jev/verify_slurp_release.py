#!/usr/bin/env python3
"""Independent streaming audit of the written SLURP transfer release."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from multimodal_jev.slurp import sha,normalize
from multimodal_jev.task_schema import JevExample


def verify(root):
    receipt=json.loads((root/'MANIFEST.json').read_text());heldout=set(receipt['identity']['config']['heldout_scenarios'])
    seen=set();partition_counts={};train_text=set();evaluation_text=set();group_partitions={}
    for filename,expected in receipt['manifests'].items():
        assert sha(root/filename)==expected, filename
        count=0;groups=Counter()
        partition=filename.removesuffix('.jsonl');split=partition.removeprefix('heldout-')
        with (root/filename).open() as handle:
            for line in handle:
                ex=JevExample.from_record(json.loads(line))
                assert ex.example_id not in seen;seen.add(ex.example_id)
                assert ex.split==split and ex.metadata['partition']==partition
                assert (ex.metadata['scenario'] in heldout)==partition.startswith('heldout-')
                group=ex.metadata['source_group'];groups[group]+=1
                previous=group_partitions.setdefault(group,partition);assert previous==partition
                words=normalize(ex.metadata['transcript_reference'])
                (train_text if split=='train' else evaluation_text).add(words)
                assert set(ex.model_input())=={'audio','schema'}
                if ex.schema.decision_type=='open_slot':
                    assert (ex.target is None)==(not ex.metadata['native_token_span'])
                count+=1
        assert set(groups.values())=={7}
        assert count==receipt['counts'][partition+'_decisions']
        assert len(groups)==receipt['counts'][partition+'_recordings']
        partition_counts[partition]={'decisions':count,'recordings':len(groups)}
    assert not train_text&evaluation_text
    return {'status':'passed','release_id':receipt['release_id'],'partitions':partition_counts,
            'unique_decision_ids':len(seen),'source_groups':len(group_partitions),
            'train_evaluation_transcript_overlap':0,'schema_targets_validated':True,
            'scope':'all written decisions, immutable file hashes and cross-partition invariants; no inference'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=verify(args.release)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2))
