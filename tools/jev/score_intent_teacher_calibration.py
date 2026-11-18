#!/usr/bin/env python3
"""Score complete fixed-panel proposals against offline calibration references."""
import argparse
from collections import Counter
import json
from pathlib import Path

from build_blind_semantic_annotation import validate_response
from consolidate_intent_presence_references import read, require, rows, sha, unique
from parse_semantic_annotation_response import parse_response_content


def summarize(cases, outputs):
    details = []
    for c in cases:
        response = outputs.get(c['request_id'], {})
        predicted = response.get(c['semantic_id'])
        if predicted is not None:
            require(predicted in ('positive', 'negative', 'unavailable'), 'invalid predicted status')
        details.append(dict(**c, predicted_status=predicted,
                            agreement=predicted == c['expected_status']))
    def group(expected):
        subset = [d for d in details if d['expected_status'] in expected]
        complete = [d for d in subset if d['predicted_status'] is not None]
        agree = sum(d['agreement'] for d in subset)
        return dict(targets=len(subset), returned=len(complete), agreements=agree,
                    agreement_all_targets=agree/len(subset) if subset else None,
                    agreement_returned_only=agree/len(complete) if complete else None,
                    failures=len(subset)-len(complete),
                    predicted_counts=dict(Counter(d['predicted_status'] or 'request_invalid' for d in subset)))
    return dict(known=group({'positive', 'negative'}), positive=group({'positive'}),
                negative=group({'negative'}), ambiguous=group({'unavailable'})), details


def score(packet, run, output):
    require(not output.exists(), 'output exists')
    terminal = read(run/'terminal.json')  # Never treat a running request as a failed proposal.
    plan = read(run/'plan.json')
    require(plan['packet_report_sha256'] == sha(packet/'report.json'), 'packet changed')
    report = read(packet/'report.json')
    for name, digest in report['files'].items():
        require(sha(packet/name) == digest, 'changed packet file')
    requests = unique(rows(packet/'requests.jsonl'), 'request_id')
    terminals = unique(terminal['results'], 'request_id')
    require(set(terminals) == set(requests), 'incomplete terminal request set')
    outputs, receipts, proposals = {}, [], []
    for rid, request in requests.items():
        path = run/rid
        intent = read(path/'intent.json')
        require(intent['request_id']==rid and intent['template_sha256']==request['template_sha256'],
                'request identity mismatch')
        require(intent['executor_sha256']==plan['executor_sha256'], 'executor identity mismatch')
        require(sha(path/'messages.private.json')==intent['messages_sha256'], 'messages changed')
        require(sha(path/'config.private.json')==intent['config_sha256'], 'config changed')
        messages = read(path/'messages.private.json')
        require(len(messages)==2 and json.loads(messages[1]['content'])==request['wire'], 'wire changed')
        cfg = read(path/'config.private.json')
        require(cfg['max_tokens']==plan['max_tokens_per_request'] and cfg['thinking']=={'type':'enabled'},
                'profile changed')
        local_terminal = read(path/'terminal.json')
        require(local_terminal==terminals[rid], 'terminal mismatch')
        receipt = dict(request_id=rid, panel_id=request['panel_id'], valid=False,
                       state=local_terminal.get('state'), exit_code=local_terminal.get('exit_code'))
        result_path=path/'response/result.json'
        if result_path.exists():
            result=read(result_path)
            receipt.update(result_sha256=sha(result_path), usage=result.get('usage'),
                           http_status=result.get('http_status'), finish_reasons=result.get('finish_reasons'))
            if result.get('state')=='response_received':
                wire=read(path/'response/request.json')
                require(wire['messages']==messages and wire['max_tokens']==cfg['max_tokens'], 'sent request changed')
                response=read(path/'response/response.json')
                try:
                    require(result['http_status']==200, 'HTTP failure')
                    choice=response['choices'][0]
                    require(choice.get('finish_reason')=='stop', 'noncomplete finish reason')
                    parsed, representation=parse_response_content(choice['message'].get('content',''))
                    validate_response(parsed,request['wire'])
                    outputs[rid]={d['id']:d['status'] for d in parsed['decisions']}
                    proposals.append(dict(request_id=rid, decisions=parsed['decisions'], optimization_allowed=False))
                    receipt.update(valid=True, representation=representation)
                except (ValueError, TypeError, KeyError, IndexError) as error:
                    receipt['validation_error']=str(error)
        receipts.append(receipt)
    metrics, details=summarize(rows(packet/'cases.offline.jsonl'),outputs)
    usage=Counter()
    for receipt in receipts:
        for key in ('prompt_tokens','completion_tokens','total_tokens'):
            if isinstance(receipt.get('usage'),dict):
                usage[key]+=receipt['usage'].get(key,0)
    output.mkdir(parents=True)
    for name, data in [('comparisons.offline.jsonl',details),('proposals.offline.jsonl',proposals)]:
        (output/name).write_text(''.join(json.dumps(d,ensure_ascii=False,sort_keys=True)+'\n' for d in data))
    observed=dict(schema='slurp-intent-calibration-observed-v1',packet_report_sha256=sha(packet/'report.json'),
        plan_sha256=sha(run/'plan.json'),terminal_sha256=sha(run/'terminal.json'),
        implementation_sha256=sha(__file__),requests=len(requests),valid_requests=len(outputs),
        metrics=metrics,usage=dict(usage),receipts=receipts,
        files={n:sha(output/n) for n in ('comparisons.offline.jsonl','proposals.offline.jsonl')},
        semantic_admission=0,training_dispatch_allowed=False,
        scope='Agreement with selected unadmitted reference proposals, not independent accuracy or audio evaluation. All failed request targets remain in denominators; additional panel answers are not gold-scored.')
    (output/'report.json').write_text(json.dumps(observed,ensure_ascii=False,indent=2)+'\n')
    return observed


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('packet','run','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    r=score(a.packet,a.run,a.output)
    print(json.dumps({k:r[k] for k in ('requests','valid_requests','metrics','usage')},indent=2))
