from pathlib import Path
import runpy

mod=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/audit_mac_slu_binding_information.py'))


def frame(intent,role,domain='d'):
    return dict(domain=domain,intent=intent,slots=[dict(name=role,value='v')])


def test_same_presence_different_binding_and_no_negative_invention():
    rows=[dict(id='a',frames=[frame('i','r'),frame('j','s')],unresolved=[]),
          dict(id='b',frames=[frame('i','s'),frame('j','r')],unresolved=[])]
    result=mod['audit'](rows)
    assert result['counts']['IS_signatures_with_multiple_bindings']==1
    assert result['counts']['unannotated_product_edges']==4
    assert all(r['absent_edge_status']=='unavailable_not_negative' for r in result['records'])


def test_domain_namespaces_and_repeated_instances_preserved_in_audit():
    r=dict(id='a',frames=[frame('i','r'),frame('i','s'),frame('j','t','other')],unresolved=[])
    p=mod['projection'](r)
    assert p['repeated_intent_instances']==1
    assert not p['unannotated_product_edges']
    assert len(p['edges'])==3


def test_empty_containers_excluded_from_collision_claim_not_population():
    rows=[dict(id='a',frames=[frame('i','r'),frame('j','s')],unresolved=[]),
          dict(id='b',frames=[frame('i','s'),frame('j','r')],unresolved=[{'kind':'empty_instance'}])]
    result=mod['audit'](rows)
    assert result['counts']['records']==2
    assert result['counts']['IS_signatures_with_multiple_bindings']==0
