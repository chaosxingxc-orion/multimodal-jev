"""Freeze epoch-specific, per-recording hard negatives from source TRAIN scores and audio readouts."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.speech_alignment_pairs import Partners
from tools.reports.audit_stage1_negative_sampling import load_panel
from tools.reports.card_scoring_benchmark import pinned


def rows(path):
    with Path(path).open() as stream:
        yield from (json.loads(line) for line in stream)


def build(c):
    from threadpoolctl import threadpool_limits
    out = Path(c['output'])
    check_paths({'frozen instance negatives': out})
    score_dir, ep = map(Path, (c['score'], c['endpoint']))
    audit, audit_sha, proof, labels, population, gold, scores = load_panel(score_dir, ep)
    embeddings = pinned(c['embeddings'])
    if (embeddings['schema'] != 'jev-audited-card-speech-embeddings-v1'
            or embeddings['policy_id'] != proof['policy_id']
            or embeddings['policy_sha256'] != proof['manifest_sha256']
            or embeddings['roster_sha256'] != audit['full_sha256']):
        raise ValueError('instance mining requires identical endpoint and TRAIN roster')
    ids = list(population)
    index = {u:i for i,u in enumerate(ids)}
    n = len(ids)
    matrix = np.empty((n, embeddings['width']), dtype=np.float32)
    seen = set()
    for block in embeddings['blocks']:
        root = Path(c['embeddings']['path']).parent/'collected'/block['name']
        rec = pinned(dict(path=root/'receipt.json',sha256=block['receipt_sha256']))
        vectors = root/'vectors.npy'
        if file_hash(vectors) != block['vectors_sha256'] or seen.intersection(rec['ids']):
            raise ValueError('speech blocks changed or overlap')
        matrix[[index[u] for u in rec['ids']]] = np.load(vectors, allow_pickle=False)
        seen.update(rec['ids'])
    if seen != population.keys() or not np.isfinite(matrix).all():
        raise ValueError('incomplete speech vectors')
    norms = np.linalg.norm(matrix, axis=1)
    if not (norms>0).all():
        raise ValueError('empty speech vectors')
    matrix /= norms[:,None]
    synthetic = pinned(c['synthetic'])
    rel = Path(c['real']['path'])
    release = pinned(dict(path=rel/'MANIFEST.json',sha256=c['real']['sha256']))
    roster = rel/'learning.jsonl'
    if file_hash(roster) != release['files']['learning.jsonl']:
        raise ValueError('real TRAIN roster changed')
    meta = {r['id']:r for r in [*rows(roster), *synthetic]}
    if (meta.keys() != population.keys() or any(
            m['role']!='learning' or m['official_split']!='train' or m['known_pending_source'] is not False
            or any(m[k]!=population[u][k] for k in ('intent','audio_sha256'))
            for u,m in meta.items())):
        raise ValueError('hard negatives require admitted source TRAIN')
    semantic = Partners(pinned(c['semantic']), population)
    counts = Counter(gold)
    # Collapse identical-label, byte-identical rows for candidate search. A conflicting or ambiguous
    # recording remains an anchor if admitted, but cannot be chosen as a negative.
    hashes = [population[u]['audio_sha256'] for u in ids]
    source = [meta[u]['source_group'] for u in ids]
    groups = {key:i for i,key in enumerate(sorted(set(source)))}
    source = np.array([groups[g] for g in source])
    by_hash = defaultdict(set)
    for i, key in enumerate(hashes):
        by_hash[key].add(int(gold[i]))
    dedup = set()
    valid = np.zeros(n,dtype=bool)
    for i, key in enumerate(hashes):
        if len(by_hash[key]) == 1 and key not in dedup and semantic.keys[ids[i]] >= 0:
            valid[i] = True
            dedup.add(key)
    clean = np.flatnonzero(valid)
    by_class = {y:np.flatnonzero(gold==y).tolist() for y in counts}
    order_by_class = {y:clean[np.argsort(-scores[clean,y],kind='stable')] for y in counts}
    k, block, epoch, seed = (c[key] for key in ('top_k','search_block','epoch','seed'))
    if not 2 <= k < len(labels) or block < 1 or not 0 < c['false_accept_fraction'] <= 1:
        raise ValueError('invalid instance negative sampling scope')
    strongest = scores.copy()
    strongest[np.arange(n),gold] = -np.inf
    card_top = np.argsort(-strongest, kind='stable', axis=1)[:,:k]
    exposure, chosen, diagnostics = defaultdict(Counter), {}, Counter()

    def eligible(i, j, used):
        return j >= 0 and valid[j] and gold[j] != gold[i] and source[j] != source[i] and source[j] not in used

    def unique_source(order, i, used, limit):
        result = []
        for j in order:
            if eligible(i,j,used):
                result.append(int(j))
                used.add(source[j])
                if len(result) == limit:
                    break
        return result

    processing = sorted(range(n),key=lambda i:hashlib.sha256(f'{seed}:{epoch}:{ids[i]}'.encode()).digest())
    with threadpool_limits(limits=c['search_threads'],user_api='blas'):
        for begin in range(0,n,block):
            anchors = processing[begin:begin+block]
            similarities = matrix[anchors] @ matrix.T
            for similarity,i in zip(similarities,anchors,strict=True):
                y, uid = int(gold[i]), ids[i]
                rng = random.Random(f'{seed}:{epoch}:{uid}')
                _,pos,control = semantic.draws(uid,epoch,seed)
                if pos is None:
                    same = by_class[y]
                    start = rng.randrange(len(same))
                    p = next((j for j in (same[(start+t)%len(same)] for t in range(len(same)))
                              if j!=i and hashes[j]!=hashes[i]),None)
                    if p is None:
                        raise ValueError('no independent positive recording for '+uid)
                else:
                    p = index[pos]
                used = {source[i],source[p]}
                search = order_by_class[y]
                cap = math.ceil(counts[y]*c['false_accept_fraction'])
                false_pool = unique_source(search,i,used.copy(),k)
                f = next((j for j in false_pool if exposure[y][j] < cap),None)
                if f is None:
                    f = next((int(j) for j in search if eligible(i,j,used) and exposure[y][j]<cap),None)
                if f is None:
                    raise ValueError('fixed-card false-accept cap exhausts eligible negatives')
                exposure[y][f] += 1
                used.add(source[f])
                diagnostics['first_false_accept'] += bool(false_pool and f == false_pool[0])
                # Deduplicate by source before picking deterministic top-k. Taking the best
                # sorted candidate per source makes ties independent of BLAS block shape.
                mask = valid & (gold!=y)
                for g in used:
                    mask &= source!=g
                similarity[~mask] = -np.inf
                limit = min(n, 2*k)
                near = []
                while len(near) < k:
                    cut = np.partition(similarity,n-limit)[n-limit]
                    t = np.flatnonzero((similarity>=cut) & mask)
                    order = t[np.lexsort((t,-similarity[t]))]
                    near = unique_source(order,i,used.copy(),k)
                    if len(near)>=k or limit==n:
                        break
                    limit = min(n,2*limit)
                if not near:
                    raise ValueError('no eligible speech neighbor')
                nn = near[0]
                used.add(source[nn])
                cards = [int(card_top[i,0]), int(rng.choice(card_top[i,1:]))]
                audio = [f,nn]
                ctl = index[control] if control is not None else -1
                # Keep mandatory slots. Reuse a hard slot if it already supplies the control.
                if (pos is not None and ctl >= 0 and
                        (ctl in audio or eligible(i,ctl,used))):
                    j = int(gold[ctl])
                    if j not in cards:
                        cards.append(j)
                    if ctl not in audio:
                        audio.append(ctl)
                        used.add(source[ctl])
                    control_slots = [cards.index(j),audio.index(ctl)]
                    diagnostics['semantic_controls'] += 1
                else:
                    control_slots = None
                if len(cards) < 3:
                    tail = [j for j in range(len(labels)) if j!=y and j not in card_top[i]]
                    cards.append(rng.choice(tail))
                if len(audio) < 3:
                    start = rng.randrange(len(clean))
                    tail = (int(clean[(start+j)%len(clean)]) for j in range(len(clean)))
                    extra = next((j for j in tail if j not in false_pool and j not in near
                                  and eligible(i,j,used)),None)
                    if extra is None:
                        raise ValueError('cannot fill independently sourced long-tail slot')
                    audio.append(extra)
                diagnostics['member_margin_violations'] += bool(scores[i,y]-scores[i,cards[0]] < c['probability_margin'])
                diagnostics['recording_margin_violations'] += bool(scores[i,y]-scores[f,y] < c['probability_margin'])
                chosen[uid] = [ids[p],[labels[j] for j in cards],[ids[j] for j in audio],control_slots]
            if begin % (block*c['progress_every_blocks']) == 0:
                print(f'mined {min(n,begin+block)}/{n} TRAIN anchors',flush=True)
    out.mkdir(parents=True,exist_ok=False)
    data=dict(schema='jev-stage1-instance-negatives-v1',epoch=epoch,labels=labels,
              rows=dict(sorted(chosen.items())),settings={k:c[k] for k in ('top_k','seed','false_accept_fraction')},
              reference=dict(policy_id=proof['policy_id'], checkpoint_sha256=proof['checkpoint_sha256'],
                             score_sha256=audit_sha, roster_sha256=audit['full_sha256'],
                             embeddings_sha256=c['embeddings']['sha256'],semantic_sha256=c['semantic']['sha256']))
    commit_json(out/'manifest.json',data)
    report=dict(schema='jev-audited-instance-negative-manifest-v1', epoch=epoch, recordings=n,
                manifest_sha256=file_hash(out/'manifest.json'),reference=data['reference'],coverage=dict(diagnostics))
    commit_json(out/'audit.json',report)
    return report


if __name__ == '__main__':
    print(json.dumps(build(load_config(*sys.argv[1:])[0])))
