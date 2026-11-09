# Multimodal Jev runtime: inherit the NPU-compatible NGA/verl/vLLM stack.
# Base image digest is intentionally immutable; update only with a new release record.
FROM sha256:8190b3ec51929baca2388452039ccf05d681c01b7c91cc55dd1e7ac5570a76b7

LABEL org.opencontainers.image.title="multimodal-jev" \
      org.opencontainers.image.description="Speech multimodal Jev baselines and RL runtime" \
      org.opencontainers.image.source="https://github.com/chaosxingxc-orion/multimodal-jev"

COPY src/ /opt/multimodal-jev/src/
COPY pyproject.toml README.md /opt/multimodal-jev/
COPY tools/ /opt/multimodal-jev/tools/
COPY configs/ /opt/multimodal-jev/configs/
COPY docker/multimodal-jev/runtime-entrypoint.sh /opt/multimodal-jev/runtime-entrypoint.sh
COPY docker/multimodal-jev/requirements-runtime.txt /opt/multimodal-jev/requirements-runtime.txt
COPY docker/multimodal-jev/IMAGE-MANIFEST.json /opt/multimodal-jev/IMAGE-MANIFEST.json

# Runtime dependencies are intentionally limited: the base already pins torch-NPU,
# transformers, verl, vLLM and vLLM-Ascend. qwen-asr is optional because its
# official package currently has a separate transformers compatibility range.
RUN python3 -m pip install --no-cache-dir --no-deps --no-build-isolation /opt/multimodal-jev \
    && chmod 0555 /opt/multimodal-jev/runtime-entrypoint.sh \
    && python3 /opt/multimodal-jev/tools/jev/check_versions.py

ENV PYTHONPATH=/opt/multimodal-jev/src:/opt/multimodal-jev/tools:$PYTHONPATH \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/mnt/cephfs/xingchao/.cache/huggingface \
    TRANSFORMERS_CACHE=/mnt/cephfs/xingchao/.cache/huggingface/transformers
ENTRYPOINT ["/opt/multimodal-jev/runtime-entrypoint.sh"]
