import pytest

from tools.reports import evaluate_context_selection_macro as entry


def test_full_selection_compilation_drops_labels_and_keeps_order(monkeypatch):
    rows=[dict(id=str(i),audio=f'/audio/{i}.wav',audio_sha256='digest',
               intent='must_not_leak',transcript='must_not_leak',source_group='private') for i in range(9226)]
    task=dict(schema=dict(candidates=[dict(id=str(i),description=f'candidate {i}') for i in range(60)]),
              thinking=dict(requirements=['listen']))
    def load(root,digest,role,*,purpose):
        assert role=='selection' and purpose=='evaluate'
        return rows,task,dict(counts=dict(selection=dict(recordings=9226)))
    monkeypatch.setattr(entry,'load',load)
    population,requests,candidates=entry.inputs(dict(release='fixture',release_sha256='digest'))
    assert len(population)==len(requests)==9226 and len(candidates)==60
    assert [p['id'] for p in population]==[r['id'] for r in rows]
    assert all(set(p)=={'id','audio_sha256','request_sha256'} for p in population)
    assert all('must_not_leak' not in str(r.model_input()) for r in requests.values())
    rows.pop()
    with pytest.raises(ValueError):entry.inputs(dict(release='fixture',release_sha256='digest'))
