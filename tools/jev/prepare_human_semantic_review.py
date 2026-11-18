#!/usr/bin/env python3
"""Export full blinded review material and import explicit human decisions."""
import argparse
import csv
import json
from pathlib import Path

from build_blind_semantic_annotation import validate_response
from consolidate_intent_presence_references import read, rows, require, sha
from prepare_semantic_quality_audit import load_quality_cohort
from probe_slurp_panel_context import expand_slot_rules

FIELDS = ['request_id', 'semantic_id', 'status', 'rationale']


def rule_markdown(pid,t,ontology):
    return '\n\n'.join(['# '+pid, t['instruction'], '## Questions',
        '\n'.join('- '+q['id']+': '+q.get('question','') for q in t['questions']),
        '## Complete task rules', '```json\n'+json.dumps(ontology,ensure_ascii=False,indent=2)+'\n```'])+chr(10)


def prepare(cohort, output):
    require(not output.exists(), 'output exists')
    report, sources, templates, schedule, _ = load_quality_cohort(cohort)
    output.mkdir(parents=True)
    def write(name, value):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+chr(10))
    # No bindings, source IDs, native labels, calibration targets or model opinions.
    write('sources.json', {k: v['reference_text'] for k,v in sources.items()})
    write('schedule.json', list(schedule.values()))
    write('templates.json', list(templates.values()))
    rule_files = []
    for pid,t in templates.items():
        ontology = expand_slot_rules(t['ontology']) if t['ontology'].get('representation') == 'shared_rules_v1' else t['ontology']
        rendered = dict(panel_id=pid,instruction=t['instruction'],ontology=ontology,
                        questions=t['questions'],response_contract=t['response_contract'])
        name = pid+'.rules.json';write(name,rendered);rule_files.append(name)
        name = pid+'.md'
        (output/name).write_text(rule_markdown(pid,t,ontology));rule_files.append(name)
    with (output/'answers.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader()
        for rid,s in schedule.items():
            for q in templates[s['panel_id']]['questions']:
                writer.writerow(dict(request_id=rid,semantic_id=q['id'],status='',rationale=''))
    instructions = '''# 独立语义盲审

本包包含完整固定审核队列。请仅依据参考文本和任务规则逐项判决，不查看模型答案、作者参考或原生标签。规则文件按面板提供；schedule.json将request_id关联到review_id和panel_id，sources.json提供参考文本。

在answers.csv中填写status及rationale，保留request_id和semantic_id。status仅允许positive、negative、unavailable。positive表示满足定义；negative表示审阅完整文本后可确认定义中的角色/意图缺席；unavailable表示真实歧义使该谓词无法判定。值未知不自动使类型未知。未审阅请留空，不能填negative或unavailable代替未完成。

每项填写简短的文本证据和相关规则；规则矛盾请在理由中明确指出，不自行改规则或强制给出正负。允许多个谓词同时成立。查看同一来源的多个面板时仍需逐项审核，不能按推测的原生意图过滤。

这是参考文本审核，不是音频听审。若需要原音确认，请记录需要听审，不声称已听过录音。提交时提供reviewer_id、审核方法、开始/结束时间、先前接触这些材料或答案的情况及是否未查看他人答案。导入工具只检查完整性与文件身份，不能认证审核者身份或判断独立性。

答案文件不作为训练数据直接发布。空项、分歧和规则冲突需要明确处理；不同审核者的一致也不自动等于正确。
'''
    (output/'README.md').write_text(instructions)
    identity=dict(schema='jev-human-reviewer-identity-v1',reviewer_id='',method='reference_text',
        started_at='',finished_at='',prior_exposure='',blind_review_attested=None)
    write('reviewer.json',identity)
    fixed=['sources.json','schedule.json','templates.json','README.md',*rule_files]
    manifest=dict(schema='jev-human-semantic-review-packet-v1',cohort_report_sha256=sha(cohort/'report.json'),
        cohort=str(cohort.resolve()),files={n:sha(output/n) for n in fixed},
        blank_answers_sha256=sha(output/'answers.csv'),blank_identity_sha256=sha(output/'reviewer.json'),
        sources=len(sources),panels=len(templates),requests=len(schedule),decisions=report['decisions_per_pass'],
        implementation_sha256=sha(__file__),semantic_admission=0,training_dispatch_allowed=False,
        scope='Blinded human reference-text review package. No actual reviewer or judgments; no automatic independence certification.')
    write('manifest.json',manifest)
    return manifest


def validate_submission(packet, answers, identity):
    manifest=read(packet/'manifest.json')
    require(manifest['schema']=='jev-human-semantic-review-packet-v1','invalid packet')
    for n,h in manifest['files'].items():require(sha(packet/n)==h,'review material changed')
    _,sources,templates,schedule,_=load_quality_cohort(Path(manifest['cohort']))
    require(sha(Path(manifest['cohort'])/'report.json')==manifest['cohort_report_sha256'],'cohort changed')
    require(read(packet/'sources.json')=={k:v['reference_text'] for k,v in sources.items()},'source material differs')
    require(read(packet/'templates.json')==list(templates.values()) and read(packet/'schedule.json')==list(schedule.values()),'schedule/template differs')
    for pid,t in templates.items():
        ontology=expand_slot_rules(t['ontology']) if t['ontology'].get('representation')=='shared_rules_v1' else t['ontology']
        expected_rule=dict(panel_id=pid,instruction=t['instruction'],ontology=ontology,
                           questions=t['questions'],response_contract=t['response_contract'])
        require(read(packet/(pid+'.rules.json'))==expected_rule,'expanded rules differ')
        require((packet/(pid+'.md')).read_text()==rule_markdown(pid,t,ontology),'human-readable rules differ')
    who=read(identity)
    require(who.get('schema')=='jev-human-reviewer-identity-v1' and who.get('method')=='reference_text','invalid reviewer identity')
    for field in ('reviewer_id','started_at','finished_at','prior_exposure'):
        require(isinstance(who.get(field),str) and bool(who[field].strip()),'explicit reviewer identity/exposure required')
    require(type(who.get('blind_review_attested')) is bool,'explicit blinding attestation required')
    expected={(rid,q['id']) for rid,s in schedule.items() for q in templates[s['panel_id']]['questions']}
    decisions={}
    with answers.open(newline='') as stream:
        reader=csv.DictReader(stream)
        require(reader.fieldnames==FIELDS,'answer columns changed')
        for row in reader:
            require(set(row)==set(FIELDS) and all(isinstance(v,str) for v in row.values()),'malformed answer row')
            key=row['request_id'],row['semantic_id']
            require(key in expected and key not in decisions,'duplicate or unknown coordinate')
            require(row['status'] in ('positive','negative','unavailable') and bool(row['rationale'].strip()),'all decisions require status and evidence; unfinished review')
            decisions[key]=row
    require(set(decisions)==expected,'incomplete review coordinates')
    receipts=[]
    for rid,s in schedule.items():
        response=dict(decisions=[dict(id=q['id'],status=decisions[rid,q['id']]['status'],rationale=decisions[rid,q['id']]['rationale'])
                                 for q in templates[s['panel_id']]['questions']])
        validate_response(response,templates[s['panel_id']])
        receipts.append(dict(request_id=rid,response=response,valid=True))
    return manifest,who,receipts


def collect(packet, answers, identity, output):
    require(not output.exists(),'output exists')
    manifest,who,receipts=validate_submission(packet,answers,identity)
    # Materialize only after complete validation; original CSV remains the evidence.
    output.mkdir(parents=True)
    result=dict(schema='jev-human-semantic-review-pass-v1',cohort_report_sha256=manifest['cohort_report_sha256'],
        reviewer_id=who['reviewer_id'],method='reference_text',complete=True,requests=receipts,
        packet_manifest=dict(path=str((packet/'manifest.json').resolve()),sha256=sha(packet/'manifest.json')),
        answers=dict(path=str(answers.resolve()),sha256=sha(answers)),identity=dict(path=str(identity.resolve()),sha256=sha(identity)),
        independence_certified=False,semantic_admission=0,training_dispatch_allowed=False,
        scope='Complete human-authored review import; distinct from model transport and not independent certification or admission.')
    (output/'pass.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+chr(10))
    return result


def load_human_pass(path, cohort_hash, schedule, role):
    data=read(path)
    require(data.get('schema')=='jev-human-semantic-review-pass-v1' and data.get('complete') is True,'complete human review required')
    require(role=='blind_reviewer','human import is the independent-review arm, not a model proposer')
    require(data.get('cohort_report_sha256')==cohort_hash,'human review cohort changed')
    for field in ('packet_manifest','answers','identity'):
        pin=data[field];require(sha(pin['path'])==pin['sha256'],'human evidence changed')
    packet=Path(data['packet_manifest']['path']).parent
    manifest,who,receipts=validate_submission(packet,Path(data['answers']['path']),Path(data['identity']['path']))
    require(manifest['cohort_report_sha256']==cohort_hash and data['requests']==receipts,'human review differs from original CSV')
    require(data.get('method')=='reference_text' and data['reviewer_id']==who['reviewer_id'],'human reviewer identity differs')
    require({r['request_id'] for r in receipts}==set(schedule),'human schedule mismatch')
    judgments={(r['request_id'],d['id']):d['status'] for r in receipts for d in r['response']['decisions']}
    panels={}
    for task in schedule.values():
        p=panels.setdefault(task['panel_id'],dict(complete=0,requests=0,fraction=1.0,threshold=.98,numerical_gate_passed=True))
        p['complete']+=1;p['requests']+=1
    normalized={**data,'role':role,'model_identity':'not_applicable_human_reference_review',
        'input_manifest_sha256':data['packet_manifest']['sha256']}
    return normalized,judgments,panels


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    for k in ('cohort','output'):prep.add_argument('--'+k,type=Path,required=True)
    imp=sub.add_parser('collect')
    for k in ('packet','answers','identity','output'):imp.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    r=prepare(a.cohort,a.output) if a.command=='prepare' else collect(a.packet,a.answers,a.identity,a.output)
    print(json.dumps(dict(schema=r['schema'],semantic_admission=0,training_dispatch_allowed=False)))
