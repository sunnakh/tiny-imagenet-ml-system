FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY tiny_imagenet/ tiny_imagenet/
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.7,<3" \
    && pip install --no-cache-dir .
RUN useradd --system --create-home app
COPY --chown=app:app release/ /app/initial-release/
USER app
ENV TINY_STATE=/app/initial-release PORT=8080 PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python", "-m", "tiny_imagenet", "serve", "--host", "0.0.0.0"]
