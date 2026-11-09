# Separate official-baseline environment; build on an allocated x86_64 host.
FROM python:3.12-slim-bookworm@sha256:0f086103b026048cc50cb5da25ec435c023c5241a83fc44daea593377abc61c4
LABEL org.opencontainers.image.title="multimodal-jev-audiojev-official" \
      org.opencontainers.image.description="Isolated official AudioJev CUDA baseline"
RUN test "$(uname -m)" = x86_64 \
    && apt-get update \
    && apt-get install -y --no-install-recommends git libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/audiojev-official/requirements.txt /opt/audiojev-runtime/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/audiojev-runtime/requirements.txt \
    && python -m pip check \
    && python -m pip freeze --all > /opt/audiojev-runtime/pip-freeze.txt \
    && python -m pip inspect > /opt/audiojev-runtime/pip-inspect.json
COPY src/ /opt/multimodal-jev/src/
COPY tools/ /opt/multimodal-jev/tools/
COPY configs/ /opt/multimodal-jev/configs/
COPY docker/audiojev-official/check_runtime.py /opt/audiojev-runtime/check_runtime.py
ENV PYTHONPATH=/opt/multimodal-jev/src \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
# Imports must pass at build time without a GPU. Device execution is a separate gate.
RUN python /opt/audiojev-runtime/check_runtime.py > /opt/audiojev-runtime/import-check.json
CMD ["python", "/opt/audiojev-runtime/check_runtime.py", "--require-cuda"]
