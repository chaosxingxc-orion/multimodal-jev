FROM sha256:fdd4a28bb7515717e9ff1988669b4ae7c24588091cf0b73f05abff1f6ff6f87b
# Independent environment: do not alter the pinned vLLM/verl runtime.
USER root
COPY upstream /opt/tau2-source
COPY wheels /opt/tau2-wheels
RUN python3 -m venv /opt/tau2-env && \
    /opt/tau2-env/bin/pip install --no-index --find-links=/opt/tau2-wheels --no-cache-dir /opt/tau2-source && \
    /opt/tau2-env/bin/pip check && \
    /opt/tau2-env/bin/pip freeze > /opt/tau2-requirements.freeze
ENV PATH="/opt/tau2-env/bin:${PATH}" PYTHONDONTWRITEBYTECODE=1
ENV LITELLM_LOCAL_MODEL_COST_MAP=True
ENTRYPOINT ["/opt/tau2-env/bin/python"]
