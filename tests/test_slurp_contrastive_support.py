import importlib.util
from pathlib import Path
from dataclasses import asdict
import pytest
from multimodal_jev.task_schema import Candidate, DecisionSchema

spec=importlib.util.spec_from_file_location('contrast_support',Path(__file__).parents[1]/'tools/jev/analyze_slurp_contrastive_support.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture():
    pairs={'alarm_set':['alarm','set'],'alarm_query':['alarm','query'],'calendar_set':['calendar','set']}
    task=dict(kind='intent',candidate_pairs=pairs,thinking=dict(requirements=['Audio only.']),
        schema=asdict(DecisionSchema('intent','choice','Which request?',
            tuple(Candidate(k,k) for k in pairs),False)))
    rows=[dict(id=str(i),source_group=str(i),audio_sha256=str(i),role='learning',official_split='train',
        known_pending_source=False,intent=k,scenario=pairs[k][0],action=pairs[k][1])
        for i,k in enumerate(['alarm_set','alarm_set','alarm_query','calendar_set'])]
    return rows,task


def test_partition_counts_cover_all_pairs_and_views_preserve_definitions():
    rows,task=fixture();report=module.analyze(rows,task)
    assert report['ordered_class_pairs']==6 and report['recording_candidate_comparisons']==8
    assert report['comparisons_by_partition']==dict(same_scenario=3,same_action=3,different_scenario_and_action=2)
    assert report['negative_pools']['alarm_set']['same_action']==['calendar_set']
    assert all(r['unchanged_candidate_objects'] for r in report['candidate_view_checks'])
    assert report['learning_population_sha256']==module.analyze(list(reversed(rows)),task)['learning_population_sha256']


@pytest.mark.parametrize('fault',['selection','pending','pair','duplicate','conflicting_group'])
def test_population_must_be_unique_admitted_and_consistent(fault):
    rows,task=fixture()
    if fault=='selection':rows[0]['role']='selection'
    elif fault=='pending':rows[0]['known_pending_source']=True
    elif fault=='pair':rows[0]['action']='query'
    elif fault=='duplicate':rows[1]['id']=rows[0]['id']
    else:rows[2]['source_group']=rows[0]['source_group']
    with pytest.raises(ValueError):module.analyze(rows,task)
