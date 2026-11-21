"""Identity, admission and publication for shared native evaluation inputs."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import re

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_decision_contract import PREFIX_LAYOUT
from multimodal_jev.context_execution import content_key, map_io
from multimodal_jev.local_checkpoint import commit_json, sync_directory
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.model_provider import cache_namespace

CACHE_SCHEMA = "jev-native-seal-encoded-cache-v1"
PROOF_STATUS = "full_stable_prefix_cache_verified"
# No experiment, endpoint, GPU placement, or scheduling knobs enter this identity.
CODE = (
    "src/multimodal_jev/audio_policy.py",
    "src/multimodal_jev/audio_preprocess.py",
    "src/multimodal_jev/context_decision_contract.py",
    "src/multimodal_jev/cached_context_preparation.py",
    "src/multimodal_jev/encoded_audio_cache.py",
    "src/multimodal_jev/model_provider.py",
    "src/multimodal_jev/readonly_encoded_audio_cache.py",
    "src/multimodal_jev/span_request_identity.py",
    "src/multimodal_jev/qwen3_asr_provider.py",
    "tools/reports/prepare_native_seal_cache.py",
    "tools/reports/native_evaluation_cache.py",
)


def read(path):
    return json.loads(path.read_bytes())


def identity(cache):
    return dict(dataset=cache.get("dataset"), panel_size=cache.get("panel_size"),
                seal_sha256=cache["seal"]["sha256"], processor_sha256=cache["processor"]["sha256"],
                model_provider_sha256=cache_namespace(cache), prompt_layout=PREFIX_LAYOUT,
                encoding=cache["encoding"])


def read_verified(cache, verification=None, *, strict=True):
    root = Path(cache["output"])
    manifest_path = root / "MANIFEST.json"
    proof_path = Path(verification) if verification else root / "VERIFIED.json"
    m, proof = read(manifest_path), read(proof_path)
    pins = cache.get("cache_identity")
    manifest_sha = file_hash(manifest_path)
    if pins is not None and (manifest_sha != pins["manifest_sha256"]
                             or file_hash(proof_path) != pins["verification_sha256"]):
        raise ValueError("pinned evaluation cache or audit receipt changed")
    i = identity(cache)
    basic_invalid = (
        m.get("schema") != CACHE_SCHEMA
        or m.get("encoding") != cache.get("encoding")
        or m.get("seal_sha256") != cache["seal"]["sha256"]
        or m.get("processor_sha256") != cache["processor"]["sha256"]
        or m.get("model_provider_sha256") != cache_namespace(cache)
        or proof.get("status") != PROOF_STATUS
        or proof.get("manifest_sha256") != manifest_sha
        or proof.get("recordings") != m.get("recordings")
    )
    if strict:
        basic_invalid = basic_invalid or (
            proof.get("model_calls") != 0
            or proof.get("npu_executed") is not False
            or type(m.get("recordings")) is not int
            or m["recordings"] < 1
            or m.get("inputs_sha256") != file_hash(root / "inputs.jsonl")
            or any(m.get(k) != v for k, v in i.items() if k not in ("dataset", "panel_size"))
        )
    if basic_invalid:
        raise ValueError("audited stable-prefix cache requires a matching independent receipt")
    if "publication_key" in cache and (m.get("publication_key") != cache["publication_key"]
                                       or proof.get("publication_key") != cache["publication_key"]
                                       or m.get("cache_input") != i):
        raise ValueError("cache publication binds a different input or auditor")
    return m, proof


def catalog_match(cache, catalog_root, seal_root, entries):
    i = identity(cache)
    match = [row for row in entries if row["input"] == i
             and Path(cache["seal"]["root"]) == Path(seal_root) / row["seal_relative_root"]]
    if len(match) > 1:
        raise ValueError("duplicate verified input cache registration")
    if match:
        row = match[0]
        name = Path(row["relative_root"])
        if name.is_absolute() or ".." in name.parts or not name.parts:
            raise ValueError("relative verified cache path required")
        return dict(cache, output=str(Path(catalog_root) / name), cache_identity=row["pins"])
    return None


def shared_publication(cache, store, *, image, source):
    """Address an encoded panel on the current machine by its input identity."""
    # The current request hash includes the canonical audio path. Preserve
    # that binding when distinguishing otherwise identical relocated seals.
    key = content_key(CACHE_SCHEMA, identity(cache), cache["seal"]["root"], image,
                      {name: source[name] for name in CODE})
    return dict(cache, output=str(Path(store) / key), publication_key=key), "shared"


def preflight(cache, *, io_workers):
    """Check the entire admitted tensor inventory without loading a processor."""
    m, proof = read_verified(cache, strict=True)
    root = Path(cache["output"])
    from multimodal_jev.cached_context_preparation import CachedContextPreparer
    from multimodal_jev.encoded_audio_cache import key_for
    from tools.reports.prepare_native_seal_cache import plan

    prepare = CachedContextPreparer(None, None, processor_sha256=m["processor_sha256"],
                                     prompt_layout=PREFIX_LAYOUT, provider_namespace=cache_namespace(cache),
                                     **cache["encoding"])
    jobs = plan(cache)
    files = {}
    records = set()
    with (root / "inputs.jsonl").open() as stream:
        for line, (name, asset, request) in zip(stream, jobs, strict=True):
            row = json.loads(line)
            if (row["dataset"], row["id"], row["audio_sha256"]) != (name, asset["id"], asset["audio_sha256"]):
                raise ValueError("cache and sealed audio roster differ")
            key, sha = row["cache_key"], row["cache_sha256"]
            record = (row["dataset"], row["id"])
            if ((cache.get("dataset") is not None and record[0] != cache["dataset"])
                    or record in records or key != key_for(prepare.identity(request, asset["audio_sha256"]))
                    or not re.fullmatch(r"[0-9a-f]{64}", sha)
                    or files.setdefault(key, sha) != sha):
                raise ValueError("invalid cache inventory")
            records.add(record)
    if len(records) != m["recordings"] or not records:
        raise ValueError("incomplete cache inventory")

    def check(item):
        key, expected = item
        path = root / "cache" / (key + ".pt")
        if path.is_symlink() or file_hash(path) != expected:
            raise ValueError("encoded cache content changed: " + key)

    map_io(check, sorted(files.items()), workers=io_workers)
    return dict(proof, cache_reused=True, cache_action="reused", verified_tensor_files=len(files))


@contextmanager
def locked_publication(root, key):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("shared evaluation cache root must be a directory")
    check_paths({"evaluation cache publication": root})
    with (root / ("." + key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def materialize(cache, provenance, *, run, io_workers):
    """Encode missing entries, audit once and expose only the audited directory."""
    target = Path(cache["output"])
    key = cache["publication_key"]
    if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key) or target.name != key:
        raise ValueError("invalid evaluation cache publication key")
    if target.is_symlink():
        raise ValueError("invalid cache publication")
    if target.is_dir():
        return preflight(cache, io_workers=io_workers)
    with locked_publication(target.parent, key):
        if target.is_symlink():
            raise ValueError("invalid cache publication")
        if target.exists():
            return preflight(cache, io_workers=io_workers)
        pending = target.with_name("." + key + ".building")
        if pending.is_symlink():
            raise ValueError("invalid pending cache publication")
        action = "resumed" if pending.is_dir() else "built"
        build = dict(cache, output=str(pending))
        if not (pending / "MANIFEST.json").is_file():
            run(build, provenance, resume=True)
        proof_path = pending / "VERIFIED.json"
        if not proof_path.exists():
            proof = run(build, provenance, verify=True)
            commit_json(proof_path, proof)
            _, proof = read_verified(build)
        else:
            proof = preflight(build, io_workers=io_workers)
        pending.rename(target)
        sync_directory(target.parent)
        return dict(proof, cache_reused=False, cache_action=action)
