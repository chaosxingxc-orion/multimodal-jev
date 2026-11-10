"""SFC-Jev candidate structure checks and isolated generic-prefix replacement.

These checks do not establish semantic suitability or authenticate teacher wire
content. Staging is local data preparation, never approval or inference launch.
"""
import hashlib
import json
from pathlib import Path

from .baseline import file_hash
from .consumption import canonical
from .sfc_jev_runtime import INSTRUCTION, messages as parent_messages
from .teacher_candidates import strict_json


def check_instruction(text):
    if type(text) is not str or not text.strip() or len(text)>1500:
        raise ValueError('nonempty instruction of at most 1500 characters required')
    if any(marker in text for marker in ('<|','|>','<audio>','### Available Functions:',
                                         '### Environment:','### Current decision:')):
        raise ValueError('instruction cannot inject control tokens or task sections')
    if any(ord(c)<32 and c not in '\n\t\r' for c in text):
        raise ValueError('unexpected control character')
    return text


def messages(request, decision_id, audios, instruction):
    check_instruction(instruction)
    result=parent_messages(request,decision_id,audios)
    original=result[0]['content']
    if not original.startswith(INSTRUCTION):raise ValueError('parent instruction boundary changed')
    # Preserve the baseline byte-for-byte when selecting the parent control.
    prefix=instruction if instruction==INSTRUCTION else instruction.rstrip()+'\n\n'
    result[0]=dict(role='system',content=prefix+original[len(INSTRUCTION):])
    return result


def vllm_request(processor,request,decision_id,audios,instruction):
    prompt=processor.apply_chat_template(messages(request,decision_id,audios,instruction),
                                          tokenize=False,add_generation_prompt=True)
    return dict(prompt=prompt,multi_modal_data={'audio':[(wave,16000) for wave in audios]})


def validate(payload,context):
    if (context.get('schema')!='sfc-jev-offline-teacher-feedback-v1' or context.get('split')!='train_id'
            or context.get('selection_or_test_feedback') is not False or context.get('weight_updates')!=0
            or context.get('parent_instruction')!=INSTRUCTION
            or context.get('parent_prompt_sha256')!=hashlib.sha256(INSTRUCTION.encode()).hexdigest()):
        raise ValueError('original learning-only parent feedback required')
    if (type(payload) is not dict or set(payload)!={'schema','parent_prompt_sha256','diagnosis','candidates'}
            or payload['schema']!='sfc-jev-teacher-prompts-v1'
            or payload['parent_prompt_sha256']!=context['parent_prompt_sha256']):
        raise ValueError('exact SFC teacher response schema and parent hash required')
    summaries={r['evidence_id'] for r in context['all_decision_summary']}
    representatives={r['evidence_id'] for r in context['representatives']}
    if not representatives or not representatives<=summaries:raise ValueError('feedback evidence differs')
    def row_fields(row,fields,text_fields,allowed):
        if type(row) is not dict or set(row)!=set(fields):raise ValueError('exact proposal fields required')
        if any(type(row[k]) is not str or not row[k].strip() for k in text_fields):
            raise ValueError('nonempty proposal text required')
        refs=row['evidence_ids']
        if (type(refs) is not list or not refs or any(type(x) is not str for x in refs)
                or len(set(refs))!=len(refs) or not set(refs)<=allowed):
            raise ValueError('known unique evidence references required')
    if type(payload['diagnosis']) is not list or not 1<=len(payload['diagnosis'])<=3:
        raise ValueError('one to three diagnoses required')
    for row in payload['diagnosis']:
        row_fields(row,('mechanism','evidence_ids','uncertainty'),('mechanism','uncertainty'),summaries)
    candidates=payload['candidates']
    if type(candidates) is not list or not 1<=len(candidates)<=2:raise ValueError('one or two candidates required')
    ids=set();instructions={INSTRUCTION.strip()}
    for row in candidates:
        row_fields(row,('id','instruction','mechanism','evidence_ids','counterevidence','falsification'),
                   ('id','instruction','mechanism','counterevidence','falsification'),representatives)
        if row['id'] not in ('p1','p2') or row['id'] in ids:raise ValueError('unique bounded candidate IDs required')
        text=check_instruction(row['instruction'])
        if text.strip() in instructions:raise ValueError('candidate must differ from parent and other candidates')
        if any(uid.split('#',1)[0] in text for uid in summaries):raise ValueError('sample ID in instruction')
        ids.add(row['id']);instructions.add(text.strip())
    return candidates


def stage(proposal,feedback,output,proposal_hash,feedback_hash):
    if file_hash(proposal)!=proposal_hash or file_hash(feedback)!=feedback_hash:
        raise ValueError('proposal or feedback file changed')
    payload=strict_json(proposal.read_text());context=strict_json(feedback.read_text())
    candidates=validate(payload,context)
    output.mkdir(parents=True,exist_ok=False,mode=0o700)
    (output/'proposal.private.json').write_bytes(canonical(payload));(output/'proposal.private.json').chmod(0o600)
    entries=[]
    for row in candidates:
        path=output/(row['id']+'.txt');path.write_text(row['instruction']);path.chmod(0o600)
        entries.append(dict(id=row['id'],instruction_sha256=file_hash(path),characters=len(row['instruction'])))
    receipt=dict(schema='sfc-jev-local-candidate-staging-v1',proposal_sha256=proposal_hash,feedback_sha256=feedback_hash,
        parent_prompt_sha256=context['parent_prompt_sha256'],candidates=entries,semantic_review='pending',
        candidate_accepted=False,inference_dispatched=False,API_sent=False,wire_authenticated=False,
        scope='Local structure screening only. No teacher wire authentication, semantic approval, token-budget '
              'verification or performance claim. Candidate generic prefix is separate from fixed task sections.')
    (output/'staged.json').write_bytes(canonical(receipt))
    return receipt
