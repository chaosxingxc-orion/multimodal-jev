"""Authenticated full-population schedule shared by RL probes and future trainer.

Reuses instruction data ordering; does not treat its CE manifest as RL launch
authorization. The caller separately authenticates release labels and policies.
"""
import json,math
from pathlib import Path
from .baseline import file_hash
from .slurp_training import content_hash
from .span_hybrid_objective import sft_coefficient


class SimpleRLSchedule:
    def __init__(self, directory, *, manifest_sha256, release_sha256):
        directory=Path(directory);path=directory/'MANIFEST.json'
        if file_hash(path)!=manifest_sha256:raise ValueError('schedule manifest changed')
        manifest=json.loads(path.read_text())
        expected=dict(global_batch=128,world_size=4,task_mode='direct_final',class_weighting='sqrt_cap4',
            loss_reduction='real_request_mean',unique_recordings=40513,total_nonpadding_recording_uses=81026,
            manifest_sha256=release_sha256)
        if any(manifest.get(k)!=v for k,v in expected.items()) or len(manifest['epochs'])!=2:
            raise ValueError('fixed full40513 two-epoch direct Final schedule required')
        self.sha256=manifest_sha256;self.release_sha256=release_sha256
        self.files={str(path):manifest_sha256};self.batches=[];self.metadata=[];identities=None
        self.population_hashes=[]
        for epoch,declared in enumerate(manifest['epochs']):
            path=directory/f'epoch-{epoch}.json'
            if file_hash(path)!=declared['file_sha256']:raise ValueError('epoch file changed')
            self.files[str(path)]=declared['file_sha256'];wire=json.loads(path.read_text());report=wire['report']
            if report!={k:v for k,v in declared.items() if k!='file_sha256'}:
                raise ValueError('epoch report differs from manifest')
            batches=wire['batches']
            self.population_hashes.append(report.get('population_sha256'))
            expected_report=dict(epoch=epoch,updates=317,recordings=40513,padding_slots=63,
                global_batch=128,world_size=4,seed=317,task_mode='direct_final',class_weighting='sqrt_cap4')
            if (any(report.get(k)!=v for k,v in expected_report.items()) or len(batches)!=317
                    or content_hash(batches)!=report['batches_sha256']):
                raise ValueError('full authenticated epoch layout required')
            real={};padding=0;weight_sum=0.
            for index,slots in enumerate(batches):
                if len(slots)!=128:raise ValueError('complete global128 batch required')
                count=0;ended=False
                for slot in slots:
                    if (slot['view']!='direct' or slot['variant']!='native' or type(slot['padding']) is not bool
                            or type(slot['weight']) not in (int,float) or not math.isfinite(slot['weight'])
                            or not isinstance(slot['id'],str) or not isinstance(slot['source_group'],str)):
                        raise ValueError('finite native schedule slot required')
                    if slot['padding']:
                        if index!=316 or slot['weight']!=0:raise ValueError('zero terminal padding required')
                        ended=True;padding+=1
                    else:
                        if ended or slot['weight']<=0 or slot['id'] in real:
                            raise ValueError('each learning recording exactly once per epoch')
                        real[slot['id']]=(slot['source_group'],slot['weight']);count+=1;weight_sum+=slot['weight']
                if count!=(65 if index==316 else 128):raise ValueError('real request denominator differs')
                self.batches.append(slots);self.metadata.append(dict(epoch=epoch,batch_index=index,real_requests=count))
            if len(real)!=40513 or padding!=63 or not math.isclose(weight_sum,40513,rel_tol=0,abs_tol=1e-6):
                raise ValueError('complete population and population-normalized weights required')
            if identities is not None and real!=identities:raise ValueError('epoch population or frozen weights changed')
            for slot in batches[-1]:
                if slot['padding'] and (slot['id'] not in real or real[slot['id']][0]!=slot['source_group']):
                    raise ValueError('padding must reference the same authenticated learning population')
            identities=real
        self.total_updates=len(self.batches)
        self.identities=identities

    def validate_population(self, rows):
        """Check admitted release population and independently recompute weights."""
        from collections import Counter
        if (len(rows)!=40513 or len({r['id'] for r in rows})!=40513
                or any(r['role']!='learning' or r['known_pending_source'] for r in rows)):
            raise ValueError('complete admitted learning population required')
        digest=content_hash(sorted(rows,key=lambda r:r['id']))
        if self.population_hashes!=[digest,digest]:raise ValueError('schedule release population differs')
        counts=Counter(r['intent'] for r in rows)
        if len(counts)!=60:raise ValueError('all native60 classes required')
        raw={c:min(4.,math.sqrt(len(rows)/(60*n))) for c,n in counts.items()}
        normalizer=sum(raw[c]*n for c,n in counts.items())/len(rows)
        expected={r['id']:(r['source_group'],raw[r['intent']]/normalizer) for r in rows}
        if expected.keys()!=self.identities.keys():raise ValueError('recording membership differs')
        for uid,(group,weight) in expected.items():
            actual=self.identities[uid]
            if actual[0]!=group or not math.isclose(actual[1],weight,rel_tol=0,abs_tol=1e-12):
                raise ValueError('full-population class weight or source binding differs')
        return dict(passed=True,recordings=len(rows),classes=60,epochs=2,updates=self.total_updates,
            uses=81026,population_sha256=digest,schedule_sha256=self.sha256)

    def at(self, completed_updates):
        if type(completed_updates) is not int or not 0<=completed_updates<self.total_updates:
            raise ValueError('unfinished exact optimizer cursor required')
        import copy
        return copy.deepcopy(self.batches[completed_updates]),dict(self.metadata[completed_updates],
            update=completed_updates,next_optimizer_update=completed_updates+1,total_updates=self.total_updates,
            coefficient=sft_coefficient(completed_updates,self.total_updates),schedule_sha256=self.sha256)
