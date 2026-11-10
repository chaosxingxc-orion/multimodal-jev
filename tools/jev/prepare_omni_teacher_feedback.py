#!/usr/bin/env python3
"""Build local bounded teacher feedback from completed controls; no API call."""
import argparse,json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.generative_teacher_feedback import TEMPLATE,build_feedback
from compare_omni_waveform import compare,load


def prepare(root,output):
    if json.loads((root/'evaluation-terminal.json').read_text())['exit_code']!=0:
        raise ValueError('complete paired controls required')
    for name in ['actual','zeros']:
        if json.loads((root/name/'evaluation-terminal.json').read_text())['exit_code']!=0:
            raise ValueError('successful child evaluation required')
        launch=json.loads((root/name/'launch.json').read_text())
        terminal=json.loads((root/name/'docker-terminal.json').read_text())
        if (terminal['container_id']!=launch['container_id'] or terminal['state']['Status']!='exited'
                or terminal['state']['ExitCode']!=0):raise ValueError('successful exact container required')
    comparison=compare(root/'actual/run',root/'zeros/run')
    identity,rows,actual=load(root/'actual/run','actual');_,_,zeros=load(root/'zeros/run','zeros')
    cfg=identity['config'];targets=Path(cfg['requests']).with_name('train.targets.jsonl')
    if file_hash(cfg['asset_audit']) != identity['asset_audit_sha256']:
        raise ValueError('model asset audit changed')
    audit=json.loads(Path(cfg['asset_audit']).read_text())
    context=build_feedback(rows,[json.loads(line) for line in targets.open()],actual,zeros,
        parent_instruction=Path(cfg['instruction']).read_text(),parent_prompt_sha256=identity['instruction_sha256'],
        actor_revision=audit['revision'],metrics=comparison['actual_metrics'],sensitivity=comparison['waveform_sensitivity'])
    raw=canonical(context)
    if len(raw)>150000:raise ValueError('bounded feedback exceeded')
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    messages=output/'messages.private.json';messages.write_bytes(canonical([dict(role='system',content=TEMPLATE),dict(role='user',content=raw.decode())]));messages.chmod(0o600)
    receipt=dict(API_sent=False,max_candidates=2,max_instruction_characters=1500,
        messages_sha256=file_hash(messages),learning_groups=64,summary_records=len(context['all_group_summary']),
        representatives=len(context['representatives']),context_bytes=len(raw),
        actual_result_sha256=comparison['actual_result_sha256'],zeros_result_sha256=comparison['zeros_result_sha256'],
        targets_sha256=file_hash(targets),partition_sha256=cfg['partition_sha256'],selection_or_test_feedback_used=False,
        scope='local feedback artifact only; teacher budget/config/dispatch not scheduled')
    atomic_json(output/'prepared.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.root,a.output),indent=2))
