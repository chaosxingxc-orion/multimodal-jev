#!/usr/bin/env python3
"""Add speaker-cluster statistics only after complete accepted FSC TEST runs."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.fsc_cluster_comparison import compare
from compare_omni_test import load
from watch_qwen3_omni_probe import inspect


def report(root,protocol_hash):
    root=Path(root);path=root/'test-protocol.json'
    if file_hash(path)!=protocol_hash:raise ValueError('registered TEST protocol changed')
    protocol=json.loads(path.read_text())
    if json.loads((root/'evaluation-terminal.json').read_text())['exit_code']!=0:
        raise ValueError('complete TEST queue required before statistics')
    runs={}
    for key in protocol['prompts']:
        launch=json.loads((root/key/'launch.json').read_text());item=inspect(launch['container_id'])
        if (item['Id']!=launch['container_id'] or item['Image']!=protocol['image']
                or item['State']['Status']!='exited' or item['State']['ExitCode']!=0):
            raise ValueError('exact successful TEST Docker required')
        runs[key]=load(root/key/'run',path,key)
    parent,rows,base,_=runs['parent']
    if any(v[1]!=rows for v in runs.values()):raise ValueError('same complete TEST population required')
    target=Path(parent['config']['requests']).with_name('test.targets.jsonl')
    if file_hash(target)!=protocol['targets_sha256']:raise ValueError('registered targets changed')
    targets=[json.loads(line) for line in target.open()]
    results={key:compare(rows,targets,list(base.values()),list(v[2].values()),draws=10000,seed=17)
             for key,v in runs.items() if key!='parent'}
    return dict(schema='fsc-official-test-cluster-report-v1',root=str(root.resolve()),
        protocol_sha256=protocol_hash,targets_sha256=file_hash(target),
        results=results,prediction_hashes={k:file_hash(root/k/'run/predictions.jsonl') for k in runs},
        source_files={n:file_hash(Path(__file__).parents[2]/n) for n in
            ('tools/jev/compare_fsc_test_clusters.py','tools/jev/compare_omni_test.py',
             'src/multimodal_jev/fsc_cluster_comparison.py','src/multimodal_jev/compare_policies.py',
             'src/multimodal_jev/generative_teacher_feedback.py','src/multimodal_jev/six_view_score.py')},
        test_feedback_used_for_optimization=False,
        scope='Supplementary descriptive uncertainty report after complete frozen TEST acceptance. '
              'Does not change registered prompts, decoding, candidate selection or official scores.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--protocol-hash',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior statistics')
    result=report(a.root,a.protocol_hash)
    with a.output.open('xb') as handle:handle.write(canonical(result))
    print(a.output)
