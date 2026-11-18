"""Independent full-selection identity and label-free task binding audit."""
import json
from .slurp_intent_structure import compile_intent
from .slurp_training import content_hash


def verify(artifact, population, task, release_sha256):
    if (artifact.get('passed') is not True or artifact.get('population_mode')!='full_selection'
            or artifact.get('release_sha256')!=release_sha256
            or artifact.get('recordings')!=len(population) or artifact.get('requests')!=len(population)):
        raise ValueError('complete matching selection processor artifact required')
    expected={r['id']:r for r in population};seen=set();lengths=[]
    if not expected or len(expected)!=len(population):
        raise ValueError('nonempty unique selection population required')
    for record in artifact['records']:
        uid=record['id']
        if uid not in expected or uid in seen:
            raise ValueError('duplicate or foreign selection input')
        seen.add(uid);row=expected[uid]
        request=compile_intent(row['audio'],task,'D')
        # Normalize tuples to the actual JSON artifact representation.
        wire=json.loads(json.dumps(request.model_input()))
        length=record['input_tokens'];positions=record['positions'];mapping=record['candidate_mapping']
        if (record['recording_id']!=uid or record['audio_sha256']!=row['audio_sha256']
                or record['view']!='direct' or record['variant']!='native'
                or record['request']!=wire or len(mapping)!=1
                or mapping[0]['values']!=list(request.questions[0].values)
                or len(mapping[0]['token_ids'])!=60 or len(set(mapping[0]['token_ids']))!=60
                or type(length)!=int or not 0<length<=4096
                or len(positions)!=1 or len(positions[0])!=1 or not 0<=positions[0][0]<length
                or record['input_ids']['shape']!=[1,length]):
            raise ValueError('selection task, waveform or readout binding differs')
        lengths.append(length)
    if seen!=expected.keys() or max(lengths)!=artifact['maximum_input_tokens']:
        raise ValueError('selection coverage or maximum length differs')
    return dict(passed=True,requests=len(seen),maximum_input_tokens=max(lengths),
                input_identity_sha256=content_hash([(r['id'],r['audio_sha256']) for r in population]),
                scope='Independent full selection/task/readout binding; no processor rerun or model execution.')
