import json
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_decision_contract import ApplicabilityQuestion, ContextDecisionRequest, PREFIX_LAYOUT
from multimodal_jev.cached_context_preparation import CachedContextPreparer
from multimodal_jev.encoded_audio_cache import key_for
from tools.reports.native_evaluation_cache import CODE, catalog_match, identity, materialize, preflight, read_verified, shared_publication


def cache_spec(tmp_path):
    return dict(dataset="SIB", seal={"sha256":"a"*64, "root":str(tmp_path/"data/sib")},
                processor={"sha256":"b"*64}, encoding={
                    "decision_token_id":151706,"audio_token_id":151676,"max_tokens":4096},
                output=str(tmp_path/"cache"))


def test_content_addressed_choice_is_independent_of_experiment(tmp_path):
    spec = cache_spec(tmp_path)
    source = dict.fromkeys(CODE, 'a' * 64)
    first, mode = shared_publication(spec, str(tmp_path/"store"), image="image-a", source=source)
    second, mode2 = shared_publication(dict(spec, output=str(tmp_path/"other")),
                                       str(tmp_path/"store"), image="image-a", source=source)
    assert mode == mode2 == "shared"
    assert first["publication_key"] == second["publication_key"]
    assert Path(first["output"]).name == first["publication_key"]
    changed, _ = shared_publication(dict(spec, encoding=dict(spec["encoding"], max_tokens=8192)),
                                    str(tmp_path/"store"), image="image-a", source=source)
    assert changed["publication_key"] != first["publication_key"]
    registered = [{"input":identity(spec), "seal_relative_root":"sib", "relative_root":"known-sib",
                   "pins":{"manifest_sha256":"c"*64,"verification_sha256":"d"*64}}]
    cached = catalog_match(spec, str(tmp_path/"catalog"), str(tmp_path/"data"), registered)
    assert Path(cached["output"]).name == "known-sib"


def test_verified_cache_binds_complete_manifest(tmp_path, monkeypatch):
    spec = cache_spec(tmp_path)
    root = Path(spec["output"])
    root.mkdir()
    inputs = root/"inputs.jsonl"
    from tools.reports import prepare_native_seal_cache
    req = request("/audio/one.wav")
    monkeypatch.setattr(prepare_native_seal_cache, "plan", lambda _: [
        ("SIB", {"id":"one", "audio_sha256":"f"*64}, req)])
    prep = CachedContextPreparer(None, None, processor_sha256="b"*64, prompt_layout=PREFIX_LAYOUT,
                                 **spec["encoding"])
    key = key_for(prep.identity(req, "f"*64))
    (root/"cache").mkdir()
    tensor = root/"cache"/(key+".pt")
    tensor.write_bytes(b"sealed tensor")
    inputs.write_text(json.dumps(dict(dataset="SIB", id="one", audio_sha256="f"*64, cache_key=key,
                                      cache_sha256=file_hash(tensor))) + "\n")
    m = dict(schema="jev-native-seal-encoded-cache-v1", recordings=1,
             seal_sha256=spec["seal"]["sha256"], processor_sha256=spec["processor"]["sha256"],
             prompt_layout="jev-stable-prefix-audio-readout-v1", encoding=spec["encoding"],
             inputs_sha256=file_hash(inputs), model_provider_sha256=None)
    (root/"MANIFEST.json").write_text(json.dumps(m))
    proof = dict(status="full_stable_prefix_cache_verified", recordings=1,
                 manifest_sha256=file_hash(root/"MANIFEST.json"), model_calls=0, npu_executed=False)
    (root/"VERIFIED.json").write_text(json.dumps(proof))
    checked, _ = read_verified(spec)
    assert checked["recordings"] == 1
    assert preflight(spec, io_workers=2)["verified_tensor_files"] == 1
    tensor.write_bytes(b"replaced tensor")
    with pytest.raises(ValueError, match="cache content changed"):
        preflight(spec, io_workers=2)
    inputs.write_text('tampered\n')
    with pytest.raises(ValueError, match="audited stable-prefix"):
        read_verified(spec)


def request(path):
    return ContextDecisionRequest(str(path), (ApplicabilityQuestion("a", "Label A", "Is it A?"),), ("Listen.",))


def test_partial_publication_resumes_and_hit_never_encodes(tmp_path, monkeypatch):
    import torch
    from multimodal_jev import cached_context_preparation, local_checkpoint, model_provider
    from tools.reports import native_evaluation_cache, prepare_native_seal_cache as encoder

    for module in (native_evaluation_cache, local_checkpoint, encoder):
        monkeypatch.setattr(module, "check_paths", lambda _: None)
    store = tmp_path/"publications"
    store.mkdir()
    cfg, _ = shared_publication(cache_spec(tmp_path), str(store), image="image-a",
                                source=dict.fromkeys(CODE, "a"*64))
    cfg.update(io_workers=1, prefetch=1, progress_interval=1)
    cfg["processor"]["policy"] = str(tmp_path/"processor")
    monkeypatch.setattr(cached_context_preparation, "processor_cache_identity", lambda _: "b"*64)
    jobs = []
    for uid in ("one", "two"):
        wav = tmp_path/(uid+".wav")
        wav.write_bytes(uid.encode())
        jobs.append(("SIB", {"id":uid,"audio_sha256":file_hash(wav)}, request(wav)))
    monkeypatch.setattr(encoder, "plan", lambda _: jobs)
    monkeypatch.setattr(cached_context_preparation, "load_audio", lambda *a, **k: (None, {}))
    encoded = []
    stopped = True

    class Provider:
        processor = staticmethod(lambda _: None)

        @staticmethod
        def encode(_, req, audio, **kw):
            uid = Path(req.audio).stem
            if uid == "two" and stopped:
                raise RuntimeError("test interruption")
            encoded.append(uid)
            ids = torch.tensor([[kw["audio_token_id"],kw["decision_token_id"]]])
            return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                        input_features=torch.ones((1,1)), input_features_mask=torch.ones((1,1))), torch.tensor([[1]])

    monkeypatch.setattr(model_provider, "resolve_provider", lambda _: Provider())
    target = Path(cfg["output"])
    with pytest.raises(RuntimeError, match="test interruption"):
        materialize(cfg, {}, run=encoder.run, io_workers=1)
    assert not target.exists()
    assert len(list((store/("."+target.name+".building")/"cache").glob("*.pt"))) == 1
    stopped = False
    resumed = materialize(cfg, {}, run=encoder.run, io_workers=1)
    assert resumed["cache_reused"] is False and resumed["cache_action"] == "resumed"
    assert encoded == ["one", "two"]

    def no_build(*a, **kw):
        pytest.fail("published input must not invoke the encoder")

    proof = materialize(cfg, {}, run=no_build, io_workers=2)
    assert proof["cache_reused"] and proof["cache_action"] == "reused" and proof["verified_tensor_files"] == 2
    next((target/"cache").glob("*.pt")).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="cache content changed"):
        materialize(cfg, {}, run=no_build, io_workers=2)
