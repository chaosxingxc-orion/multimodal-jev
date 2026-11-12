"""Read-only reference coverage; matching audio does not match task knowledge."""
from collections import Counter,defaultdict


def compare(roster,jobs,ontology):
    if not roster or len({r['id'] for r in roster})!=len(roster):raise ValueError('distinct nonempty current roster required')
    if not jobs or len({j['sample_id'] for j in jobs})!=len(jobs):raise ValueError('distinct reference requests required')
    support=set(ontology);index=defaultdict(list);sizes=Counter();all_labels=set()
    for job in jobs:
        ranking=job['ranking'];labels=[r['label'] for r in ranking];indices=[r['index'] for r in ranking]
        if (not labels or len(set(labels))!=len(labels) or indices!=list(range(1,len(labels)+1))
                or not set(labels)<=support):raise ValueError('distinct indexed candidates inside shared ontology required')
        if job['audio_sha256']!=job['audio']['audio_sha256']:raise ValueError('reference audio identities disagree')
        sizes[len(labels)]+=1;all_labels.update(labels);index[job['audio_sha256']].append(job)
    matched=[];missing=[]
    for row in roster:
        if row['gold'] not in support:raise ValueError('current gold outside ontology')
        found=index.get(row['audio_sha256'],[])
        if not found:missing.append(row['id']);continue
        matched.append(dict(id=row['id'],audio_sha256=row['audio_sha256'],reference_requests=[dict(
            sample_id=j['sample_id'],candidate_count=len(j['ranking']),
            same_candidate_set={r['label'] for r in j['ranking']}==support,
            reviewed_gold_in_candidates=row['gold'] in {r['label'] for r in j['ranking']}) for j in found]))
    return dict(current_recordings=len(roster),reference_requests=len(jobs),reference_unique_audio=len(index),
        reference_candidate_counts=dict(sorted(sizes.items())),reference_union_labels=len(all_labels),
        current_candidates=len(support),exact_audio_matched_recordings=len(matched),matches=matched,
        unmatched_ids=missing,matched_full_candidate_recordings=sum(any(j['same_candidate_set'] for j in r['reference_requests']) for r in matched),
        scope='Exact audio hash overlap and candidate support only. No semantic score or equal-template/equal-knowledge claim.')
