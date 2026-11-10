"""Task-specific frozen inference IO; no target or provenance reader."""
import hashlib
import json
from pathlib import Path
from .consumption import canonical, validate_request, inference_messages
from .policy_evaluation import prediction_record
from .task_schema import DecisionSchema
from . import slurp_native


class FrozenContract:
    def __init__(self, config):
        self.kind = config.get('task_format', 'jev')
        self.taxonomy = None
        self.taxonomy_sha256 = None
        if self.kind == 'slurp-native':
            if config['route'] != 'asr-text':
                raise ValueError('native SLURP currently requires ASR-text route')
            path = Path(config['taxonomy'])
            self.taxonomy = json.loads(path.read_text())
            self.taxonomy_sha256 = hashlib.sha256(canonical(self.taxonomy)).hexdigest()
            if self.taxonomy_sha256 != config['taxonomy_sha256']:
                raise ValueError('pinned SLURP taxonomy changed')
        elif self.kind != 'jev':
            raise ValueError('unsupported task format')

    def validate(self, row):
        if self.kind == 'jev': return validate_request(row)
        slurp_native.messages(row, self.taxonomy, '', '')
        if (row['split'] not in ('train', 'valid', 'test') or
                any(type(row[k]) is not str or not row[k] for k in ('id', 'source_group', 'audio_sha256'))):
            raise ValueError('native request identity required')
        if type(row['input']['audio']) is not str or not row['input']['audio']:
            raise ValueError('native audio path required')
        return row

    def messages(self, row, evidence, instruction):
        if self.kind == 'jev': return inference_messages(row, evidence, instruction)
        return slurp_native.messages(row, self.taxonomy, evidence, instruction)

    def prediction(self, row, raw):
        if self.kind == 'jev':
            record = prediction_record(row['id'], raw)
            schema = DecisionSchema.from_dict(row['input']['schema'])
            record['schema_valid'] = record['format_valid'] and schema.accepts(record.get('value'))
            return record
        record = {'example_id': row['id'], 'raw_text': raw, 'format_valid': False, 'schema_valid': False}
        try:
            value = slurp_native.parse_prediction(raw)
        except (ValueError, TypeError):
            return record
        record.update(prediction=value, format_valid=True, schema_valid=slurp_native.accepts(value, self.taxonomy))
        return record
