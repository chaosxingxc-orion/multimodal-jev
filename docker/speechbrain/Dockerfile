FROM sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c
# Add a separate environment; preserve the inherited NGA/verl/vLLM packages.
USER root
COPY upstream /opt/speechbrain-source
COPY wheels /opt/speechbrain-wheels
COPY requirements-extra.txt /opt/speechbrain-extra.txt
RUN python3 -m venv --system-site-packages /opt/speechbrain-env && \
    /opt/speechbrain-env/bin/pip install --no-index --find-links=/opt/speechbrain-wheels \
        --no-deps --require-hashes -r /opt/speechbrain-extra.txt && \
    /opt/speechbrain-env/bin/pip install --no-index --no-deps --no-build-isolation /opt/speechbrain-source && \
    /opt/speechbrain-env/bin/pip freeze > /opt/speechbrain-requirements.freeze
ENV PATH="/opt/speechbrain-env/bin:${PATH}" PYTHONDONTWRITEBYTECODE=1
# Inherit entrypoint: Ascend driver libraries are needed even for CPU imports.
