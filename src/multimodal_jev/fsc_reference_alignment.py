"""Align an existing published-model run with a frozen audio roster, no gold."""
from pathlib import Path
from .baseline import file_hash


def align(population, predictions, *, verify_audio=True):
    if (population['schema'] != 'fsc-official-span-population-v1'
            or population['split'] != 'test'
            or population['role'] != 'frozen_transfer_evaluation_inputs'):
        raise ValueError('frozen official TEST population required')
    sources = population['records']; root = Path(population['dataset_root']).resolve()
    if len(sources) != population['recordings'] or not sources:
        raise ValueError('complete roster required')
    found = {r['example_id']:r for r in predictions}
    if len(found) != len(predictions): raise ValueError('duplicate reference predictions')
    expected = set(); aligned = []
    for source in sources:
        audio = Path(source['audio']).resolve()
        relative = audio.relative_to(root).as_posix()
        if source['id'] != 'fsc:'+relative or relative in expected:
            raise ValueError('canonical unique recording identity required')
        expected.add(relative)
        if relative not in found: raise ValueError('missing reference prediction')
        row = found[relative]
        if row['audio_sha256'] != source['audio_sha256']:
            raise ValueError('reference audio identity differs')
        if verify_audio and file_hash(audio) != source['audio_sha256']:
            raise ValueError('current waveform file differs')
        if not isinstance(row['raw_prediction'],str): raise ValueError('preserved raw model output required')
        aligned.append(dict(id=source['id'], reference_example_id=relative,
                            audio_sha256=source['audio_sha256'], speaker_id=source['speaker_id']))
    if set(found) != expected: raise ValueError('extra reference predictions')
    speakers = len({r['speaker_id'] for r in sources})
    if speakers != population['speakers']: raise ValueError('speaker population differs')
    return dict(passed=True, recordings=len(aligned), speakers=speakers, alignment=aligned,
        scope='Same recording IDs and audio bytes only. No target read, rescoring, probability comparison or controlled training comparison.')
