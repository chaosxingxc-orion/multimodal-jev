#!/usr/bin/env python3
"""Derive a bounded TRAIN-only teacher revision without replaying prior spend."""
import argparse
import json
import os
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.consumption import canonical


def derive(context):
    if (context['schema']!='jev-direct-audio-ace-feedback-v1' or context['split']!='train'
            or context['selection_or_test_feedback'] is not False or context['weight_updates']!=0
            or set(context['actor_models'])!={'audio_model'}):
        raise ValueError('direct audio TRAIN-only source required')
    result=json.loads(json.dumps(context));kept=[];views=set()
    for row in result['representatives']:
        view=row['evidence_id'].rsplit(':',1)[-1]
        if view not in views:kept.append(row);views.add(view)
    if views!={'native','permutation','rebind','missing_candidate','noul.True','noul.False'}:
        raise ValueError('all six representative views required')
    result['representatives']=kept
    for cell in result['paired_diagnostic']['by_view'].values():
        cell.pop('actual_values',None);cell.pop('zeros_values',None)
    return result


def prepare(source,output,executor,review_path=None):
    prior=json.loads((source/'response/result.json').read_text())
    if review_path is None and (prior['state']!='unknown' or
            json.loads((source/'process-terminal.json').read_text())['exit_code']!=1):
        raise ValueError('registered unknown local-terminal source required')
    if review_path is not None and (prior['state']!='response_received' or prior['finish_reasons']!=['stop']):
        raise ValueError('complete prior response required for review feedback')
    cfg=json.loads((source/'executor-config.private.json').read_text())
    if file_hash(cfg['messages']['path'])!=cfg['messages']['sha256']:
        raise ValueError('source message pin changed')
    pinned='c9f9bbd333f038955945db82d946cb2757bafbc340b1a8ca476f52f3ab406bd4'
    if file_hash(executor)!=pinned:raise ValueError('Omni executor pin changed')
    messages=json.loads(Path(cfg['messages']['path']).read_text())
    context=derive(json.loads(messages[1]['content']))
    template=messages[0]['content'].replace('at most four distinct instructions, at most 4000 characters each.',
        'at most two distinct instructions, at most 1500 characters each.')
    template+='\nRevision v2: return at most 3 concise diagnoses and 2 candidates (p1/p2). Keep analysis brief; stop once evidence supports a bounded proposal. Do not enumerate or reanalyze every sample. Do not attempt to solve or transcribe the audio. Use representative evidence IDs. Final JSON is the deliverable; reserve space for it. No gains are established.\n'
    if review_path is not None:
        review=json.loads(review_path.read_text())
        proposal=source/'candidates/proposal.json'
        if (review['proposal_sha256']!=file_hash(proposal)
                or review['parent_prompt_sha256']!=context['parent_prompt_sha256']
                or review['evaluation_eligible']!=[]
                or review['selection_or_test_items_read_for_review'] is not False):
            raise ValueError('bound rejected proposals and TRAIN-only review required')
        context['previous_proposal']=json.loads(proposal.read_text())
        context['local_rejection_feedback']={'candidates':review['candidates'],
            'diagnosis_corrections':review['diagnosis_corrections']}
        template+='''\nRevision v3 addresses the supplied rejection feedback. Previous candidates were NOT evaluated on held-out data. Produce new instructions, not the rejected text. The runtime has alternatives with label, value, description. Actor MUST output exactly the single character in the chosen label field, never the value field. Boolean/null values are mapped by the readout, not emitted by the actor. Do NOT tell the actor to output true, false, Unknown, null, candidate IDs or JSON. Instead say choose the alternative satisfying the exact proposition and emit its label. No fixed action/object/location checklist, no task synonyms, no command-specific facts, no none=unspecified rule; semantics come ONLY from supplied question and descriptions. Unknown is considered only if explicitly listed; choose its label only if warranted by the supplied definition/evidence. Choose and map are internal deliberation, never a separate transcription stage. Different prompts may emphasize content matching or hypothesis verification; remain generic. Correct the reported counting error. Keep each instruction under1500 characters and diagnoses brief.\n'''
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    msg=output/'messages.private.json'
    msg.write_bytes(canonical([{'role':'system','content':template},{'role':'user','content':canonical(context).decode()}]));msg.chmod(0o600)
    cfg.update(messages={'path':str(msg.resolve()),'sha256':file_hash(msg)},output_root=str((output/'response').resolve()))
    config=output/'executor-config.private.json';config.write_bytes(canonical(cfg));config.chmod(0o600)
    receipt={'source':str(source),'source_messages_sha256':json.loads((source/'executor-config.private.json').read_text())['messages']['sha256'],
        'source_result_sha256':file_hash(source/'response/result.json'),'messages_sha256':file_hash(msg),
        'executor_sha256':pinned,'context_bytes':len(canonical(context)),'summary_records':len(context['all_group_summary']),
        'representatives':len(context['representatives']),'max_candidates':2,'max_instruction_characters':1500,
        'max_tokens':cfg['max_tokens'],'max_new_requests':1,'automatic_retry':False,'API_sent':False,
        'prior_usage':prior.get('usage'),'review_sha256':file_hash(review_path) if review_path else None,
        'credential_available':bool(os.environ.get(cfg['api_key_env'])),
        'selection_or_test_feedback_used':False,'reasoning_budget_provider_enforced':False}
    atomic_json(output/'prepared.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','output','executor'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--review',type=Path)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output,a.executor,a.review),indent=2))
