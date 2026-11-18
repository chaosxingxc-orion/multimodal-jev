"""Read-only MINDS-14 audit; do not expose label-bearing paths to an actor."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pyarrow
import pyarrow.parquet as pq
import soundfile as sf


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def audit(root):
    files = sorted(root.glob('*/train-*.parquet'))
    if len(files) != 14: raise ValueError('Expected 14 local language shards')
    sources = {str(root / 'README.md'): sha_file(root / 'README.md')}
    all_rows = []; language_reports = []; roster = None; ontology = None
    for path in files:
        sources[str(path)] = sha_file(path)
        f = pq.ParquetFile(path)
        features = json.loads(f.schema_arrow.metadata[b'huggingface'])['info']['features']
        current_roster = features['lang_id']['names']; current_ontology = features['intent_class']['names']
        if roster is None: roster, ontology = current_roster, current_ontology
        if current_roster != roster or current_ontology != ontology:
            raise ValueError('Language/intent index meaning changes between shards')
        if features['audio']['sampling_rate'] != 8000 or len(ontology) != 14:
            raise ValueError('Unexpected published audio or ontology metadata')
        counts = Counter(); durations = []; rates = Counter(); channels = Counter(); failures = []
        for batch in f.iter_batches(batch_size=32, columns=['path', 'audio', 'intent_class', 'lang_id']):
            for row in batch.to_pylist():
                language = roster[row['lang_id']]; label = ontology[row['intent_class']]
                if language != path.parent.name: raise ValueError('Language shard identity mismatch')
                original = row['path']; payload = row['audio']['bytes']
                if not payload: raise ValueError('Audio bytes unavailable')
                uid = 'minds14:' + hashlib.sha256((language+'\0'+original).encode()).hexdigest()
                record = dict(id=uid, language=language, intent=label,
                              upstream_split='train', source_path_sha256=hashlib.sha256(original.encode()).hexdigest(),
                              encoded_audio_sha256=hashlib.sha256(payload).hexdigest())
                counts[label] += 1
                try:
                    waveform, rate = sf.read(io.BytesIO(payload), dtype='float32', always_2d=True)
                    if not len(waveform) or not np.isfinite(waveform).all(): raise ValueError('Empty/nonfinite audio')
                    duration = len(waveform)/rate; durations.append(duration)
                    rates[rate] += 1; channels[waveform.shape[1]] += 1
                    identity = f'{rate}:{waveform.shape[1]}:'.encode()+waveform.astype('<f4').tobytes()
                    record.update(sample_rate=rate,channels=waveform.shape[1],samples=len(waveform),
                                  seconds=duration, waveform_sha256=hashlib.sha256(identity).hexdigest(),
                                  peak=float(np.max(np.abs(waveform))))
                except (RuntimeError, ValueError) as error:
                    record['decode_error'] = type(error).__name__ + ': ' + str(error)
                    failures.append(uid)
                all_rows.append(record)
        if sum(counts.values()) != f.metadata.num_rows: raise ValueError('Parquet row count mismatch')
        language_reports.append(dict(language=path.parent.name,recordings=sum(counts.values()),
                                     classes=dict(sorted(counts.items())),missing_classes=sorted(set(ontology)-set(counts)),
                                     sample_rates=dict(rates),channels=dict(channels),decode_failures=failures,
                                     hours=sum(durations)/3600,max_seconds=max(durations,default=None)))
    if len({r['id'] for r in all_rows}) != len(all_rows): raise ValueError('Duplicate source identity')
    dup = defaultdict(list)
    for row in all_rows:
        if 'waveform_sha256' in row: dup[row['waveform_sha256']].append(row)
    duplicates = [dict(waveform_sha256=k,ids=[r['id'] for r in rs],
                       languages=sorted({r['language'] for r in rs}),labels=sorted({r['intent'] for r in rs}))
                  for k,rs in dup.items() if len(rs)>1]
    return dict(schema='minds-local-audio-audit-v1', sources=sources,
                dependencies=dict(pyarrow=pyarrow.__version__,numpy=np.__version__,soundfile=sf.__version__),
                recordings=len(all_rows),language_variants=roster,intent_ids=ontology,
                languages=language_reports,waveform_duplicates=duplicates,records=all_rows,
                training_dispatch_allowed=False,evaluation_dispatch_allowed=False,
                scope='Local upstream TRAIN inventory and complete audio decode only; not official TEST, '
                      'speaker/semantic deduplication, task-definition admission, processor or model inference. '
                      'Transcription columns were not read. Original paths contain labels and must stay outside actor input.')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.root);result['sources'][str(Path(__file__))]=sha_file(Path(__file__))
    with a.output.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps(dict(recordings=result['recordings'],languages=len(result['languages']),
                         decode_failures=sum(len(x['decode_failures']) for x in result['languages']),
                         duplicate_waveform_groups=len(result['waveform_duplicates']),
                         total_hours=sum(x['hours'] for x in result['languages'])),indent=2))
