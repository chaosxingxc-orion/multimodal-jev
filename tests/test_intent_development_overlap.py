from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from audit_intent_development_overlap import compare


def row(uid,group,audio):return dict(id=uid,file=uid+'.flac',source_group=group,audio_sha256=audio)


def test_source_disjoint_does_not_imply_speaker_disjoint():
    a=[row('a','g1','h1')];b=[row('b','g2','h2'),row('c','g3','h3')]
    out=compare(a,b,{'a.flac':{'s1'},'b.flac':{'s1'},'c.flac':{'s2'}})
    assert out['overlap_counts']==dict(source_group=0,file=0,audio_sha256=0,speaker=1)
    assert out['development_seen_speaker_recordings']==1 and out['development_unseen_speaker_recordings']==1


def test_audio_duplicates_reported_without_reselecting():
    a=[row('a','g1','same')];b=[row('b','g2','same')]
    out=compare(a,b,{'a.flac':{'s1'},'b.flac':{'s2'}})
    assert out['overlap_counts']['audio_sha256']==1
    assert out['counts']['development']['recordings']==1


@pytest.mark.parametrize('bad',['missing','ambiguous','duplicate'])
def test_incomplete_or_ambiguous_metadata_rejected(bad):
    a=[row('a','g1','h1')];b=[row('b','g2','h2')];speakers={'a.flac':{'s1'},'b.flac':{'s2'}}
    if bad=='missing':speakers.pop('b.flac')
    if bad=='ambiguous':speakers['b.flac'].add('s3')
    if bad=='duplicate':b+=b
    with pytest.raises(ValueError):compare(a,b,speakers)
