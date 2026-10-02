FROM python:3.12-slim@sha256:dddfd7e07f9d15aeeca61529320492139d21cac7f0070c00609243e51e4e0016
WORKDIR /app
RUN pip install --no-cache-dir httpx==0.28.1 pydantic==2.11.7
COPY litellm/proxy/engine/__init__.py litellm/proxy/engine/models.py litellm/proxy/engine/trace_store.py litellm/proxy/engine/analysis.py litellm/proxy/engine/worker.py /app/engine/
VOLUME /tmp
USER 65532:65532
CMD ["python", "-m", "engine.worker"]
