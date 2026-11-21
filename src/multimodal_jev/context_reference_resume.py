"""Resume inference into a fresh journal from a pinned, complete prefix."""
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

from multimodal_jev.context_logits_wire import validate_logits_record
from multimodal_jev.context_reference_dataset import digest
from multimodal_jev.context_replica_macro_rollout import plan_rollouts
from multimodal_jev.local_checkpoint import commit_file


def batch_dirs(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('regular source journal required')
    batches = sorted(root.glob('batch-*'))
    if not batches:
        raise ValueError('source journal has no macro attempts')
    for index, path in enumerate(batches):
        if path.name != f'batch-{index:08d}' or path.is_symlink() or not path.is_dir():
            raise ValueError('contiguous macro attempt journal required')
    return batches


def read_files(directory):
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file() or not re.fullmatch(
                r'(INPUT|COMPLETE|rollout-[0-9]{4})\.json', path.name):
            raise ValueError('unexpected file in committed macro: ' + path.name)
        files[path.name] = path.read_bytes()
    return files


def seal(journal, lineage):
    """Create a hash index from an independently bound, stopped attempt."""
    batches = batch_dirs(journal)
    commits = []
    for index, path in enumerate(batches):
        if not (path/'COMPLETE.json').exists():
            if index != len(batches)-1:
                raise ValueError('uncommitted macro before journal tail')
            break
        files = read_files(path)
        commits.append({name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()})
    if not commits:
        raise ValueError('no complete macro to resume')
    return dict(schema='jev-reference-prefix-v1', lineage=lineage, commits=commits)


def archive_prefix(journal, manifest, stream):
    """Transport the stopped attempt without shipping its incomplete tail."""
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as out:
        for index, expected in enumerate(manifest['commits']):
            name = f'batch-{index:08d}'
            files = read_files(Path(journal)/name)
            if {key: hashlib.sha256(raw).hexdigest() for key, raw in files.items()} != expected:
                raise ValueError('prefix changed during packaging')
            for key, raw in files.items():
                out.writestr(name+'/'+key, raw)


def lineage(build, *, endpoint_sha256):
    pool = build['macro_client']
    return dict(endpoint_sha256=endpoint_sha256,
        **{key: build[key] for key in (
            'base_id', 'release_sha256', 'source_policy_id', 'source_contract',
            'input_proof_sha256', 'input_receipts_sha256', 'expected_population',
            'audio_token_id', 'max_tokens')},
        prompt_layout=build.get('prompt_layout'), runtime_identity=pool['runtime_identity'],
        max_queries=pool['max_queries'], rollout_queries=pool['rollout_queries'],
        replicas=len(pool['replicas']))


def positive_seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


class ResumePool:
    """Reuse exactly the declared prefix; inference starts at the first missing macro."""

    def __init__(self, engine, *, manifest, manifest_sha256, lineage, replay_last=False,
                 journal=None, archive=None):
        path = Path(manifest)
        raw = path.read_bytes()
        if path.is_symlink() or hashlib.sha256(raw).hexdigest() != manifest_sha256:
            raise ValueError('pinned resume manifest required')
        source = json.loads(raw)
        commits = source.get('commits')
        if (source != dict(schema='jev-reference-prefix-v1', lineage=lineage, commits=commits)
                or not isinstance(commits, list) or not commits or type(replay_last) is not bool):
            raise ValueError('resume lineage or commit list differs')
        if (journal is None) == (archive is None):
            raise ValueError('exactly one read-only resume source required')
        dst = engine.root.resolve()
        src = Path(journal if journal is not None else archive).resolve()
        if src.is_relative_to(dst) or dst.is_relative_to(src) or any(dst.glob('batch-*')):
            raise ValueError('fresh disjoint destination journal required')
        self.batches = None
        self.archive = archive
        if journal is not None:
            self.batches = batch_dirs(journal)
            if (len(self.batches) not in (len(commits), len(commits)+1)
                    or (len(self.batches) > len(commits) and (self.batches[-1]/'COMPLETE.json').exists())):
                raise ValueError('resume journal differs from sealed prefix')
        else:
            with zipfile.ZipFile(archive) as payload:
                names = payload.namelist()
            expected = {f'batch-{i:08d}/{file}' for i, batch in enumerate(commits) for file in batch}
            if len(names) != len(expected) or set(names) != expected:
                raise ValueError('resume archive differs from sealed prefix')
        self.engine = engine
        self.commits = commits
        self.index = 0
        self.reused = 0
        self.replayed = 0
        self.replay_last = replay_last
        self.manifest_sha256 = manifest_sha256

    @property
    def bundle(self):
        return self.engine.bundle

    @property
    def runtime_identity(self):
        return self.engine.runtime_identity

    def score_many(self, rows, *, audio_token_id):
        index = self.index
        name = f'batch-{index:08d}'
        if index >= len(self.commits):
            result = self.engine.score_macro(name, rows, audio_token_id=audio_token_id)
        else:
            if self.batches is not None:
                files = read_files(self.batches[index])
            else:
                with zipfile.ZipFile(self.archive) as payload:
                    files = {key: payload.read(name+'/'+key) for key in self.commits[index]}
            if {k: hashlib.sha256(v).hexdigest() for k, v in files.items()} != self.commits[index]:
                raise ValueError('committed resume macro content changed')
            result = self.read_macro(name, rows, files)
            if self.replay_last and index == len(self.commits)-1:
                replay = self.engine.score_macro(name, rows, audio_token_id=audio_token_id)
                if replay != result:
                    raise ValueError('last committed reference macro replay differs')
                self.replayed += 1
            else:
                dest = self.engine.root/name
                dest.mkdir(exist_ok=False)
                for key in sorted(files.keys()-{'COMPLETE.json'}) + ['COMPLETE.json']:
                    commit_file(dest/key, lambda stream: stream.write(files[key]))
                self.reused += 1
        self.index += 1
        return result

    def read_macro(self, name, rows, files):
        policy = self.bundle['policy_id']
        inputs, costs = self.engine._inputs(rows)
        jobs = plan_rollouts(costs, self.engine.rollout_queries, replicas=len(self.engine.clients))
        manifest = dict(batch_id=name, policy_id=policy, runtime_identity=self.runtime_identity,
                        inputs=inputs, expected_indices=list(range(len(rows))), jobs=jobs)
        expected_files = {'INPUT.json', 'COMPLETE.json', *(j['rollout_id']+'.json' for j in jobs)}
        if set(files) != expected_files or json.loads(files['INPUT.json']) != manifest:
            raise ValueError('resume macro input, policy, runtime or topology differs')
        result = [None]*len(rows)
        hashes = {}
        for job in jobs:
            rid = job['rollout_id']
            row = json.loads(files[rid+'.json'])
            records = row['records']
            if (not positive_seconds(row['seconds']) or len(records) != len(job['indices'])
                    or row != dict(job, records=records, seconds=row['seconds'], policy_id=policy)):
                raise ValueError('resume macro rollout differs')
            hashes[rid] = digest(row)
            for slot, record in zip(job['indices'], records, strict=True):
                validate_logits_record(record)
                positions = rows[slot]['positions'][0].tolist()
                binding = dict(policy_id=policy, positions=positions,
                    inputs={k:v for k,v in inputs[slot].items() if k != 'positions'})
                if (record['request_id'] != digest(binding) or record['request_identity'] != binding
                        or record['policy_id'] != policy or len(record['prob']) != len(positions)):
                    raise ValueError('resume macro response does not bind to its input')
                result[slot] = record
        complete = json.loads(files['COMPLETE.json'])
        if (not positive_seconds(complete['seconds'])
                or complete != dict(batch_id=name, policy_id=policy, status='complete_macro_inference',
                    input_sha256=digest(manifest), rollout_hashes=hashes,
                    completed_indices=manifest['expected_indices'], result_sha256=digest(result),
                    seconds=complete['seconds'], actor_update_performed=False)):
            raise ValueError('resume macro commit differs')
        return result

    def finish(self):
        if self.index < len(self.commits):
            raise ValueError('sealed resume macro beyond declared TRAIN population')
        return dict(manifest_sha256=self.manifest_sha256, reused_macros=self.reused,
                    replayed_macros=self.replayed, new_macros=self.index-len(self.commits))
