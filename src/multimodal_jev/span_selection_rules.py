"""Reference-reviewed selection projection, separate from training admission."""
from .slurp_training import content_hash
from .span_contract import SpanQuestion,SpanRequest
from .task_schema import DecisionSchema
from .span_rule_dataset import build_rule_controls
from .span_rule_balance import add_complementary_rules


def project_selection(packet,review,tasks):
    entries={r['id']:r for r in review['records']}
    if (len(entries)!=len(review['records']) or len({r['request']['id'] for r in packet})!=len(packet)
            or set(entries)!={r['request']['id'] for r in packet}
            or review['schema']!='jev-selection-reference-review-v1'):
        raise ValueError('complete distinct selection review required')
    keys=('date','time','timeofday')
    if set(tasks)!=set(keys):raise ValueError('fixed temporal question bank required')
    questions=tuple(SpanQuestion(k,DecisionSchema.from_dict(tasks[k]['schema'])) for k in keys)
    requirements=tuple(tasks['date']['thinking']['requirements'])
    bundles=[];masks=[]
    for row in packet:
        request,target=row['request'],row['target'];entry=entries[request['id']]
        if request['role']!='selection' or request['official_split']!='train':
            raise ValueError('selection role required')
        if (entry['request_sha256']!=content_hash(request) or entry['native_target_sha256']!=content_hash(target)
                or entry['task_sha256']!={k:content_hash(tasks[k]) for k in keys}
                or not entry['reviewer'] or not entry['note'] or not entry['evidence']
                or set(entry['status'])!=set(keys) or set(entry['reviewed_values'])!=set(keys)):
            raise ValueError('stale or incomplete semantic review')
        types={e['type'] for e in target['native_target']['entities']}
        active=[];values=[]
        for k in keys:
            status=entry['status'][k];value=entry['reviewed_values'][k]
            if status not in ('reviewed','pending','conflict','unavailable'):
                raise ValueError('unknown review status')
            if status=='reviewed':
                if type(value) is not bool or value is not (k in types):
                    raise ValueError('reviewed value contradicts native label; separate correction required')
            elif value is not None:raise ValueError('masked decision cannot supply gold')
            active.append(status=='reviewed');values.append(value)
            if status!='reviewed':masks.append(dict(id=request['id'],task=k,status=status,reason=entry['note']))
        bundles.append(dict(id=request['id'],request=SpanRequest(request['input']['audio'],questions,'timeofday',requirements),
            targets=values,supervised=active,source_group=request['source_group'],audio_sha256=request['audio_sha256'],
            decision_ids=[request['id']+'::'+k for k,ok in zip(keys,active) if ok]))
    # Reuse pure question/Boolean rendering to keep exact matched input syntax.
    # These helpers use learning as their historical output default. No output
    # is published or admitted until the explicit selection role is installed.
    rendered,report=build_rule_controls(bundles)
    rows=add_complementary_rules(rendered)
    for row in rows:
        row['split']='selection'
        row['optimization_allowed']=False
        row['provenance']['review_scope']=review['scope']
        row['provenance']['source_role']='selection'
    report.update(schema='jev-selection-rule-projection-v1',derived_rows=len(rows),masks=masks,
        training_admitted=False,evaluation_admitted=False,
        scope='All three question arms and complementary rules on reference-reviewed reserved selection sources; no resampling or test-score claim')
    return rows,report
