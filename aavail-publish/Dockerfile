FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    FORECAST_STATE=/var/lib/forecast
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --gid 10001 forecast \
    && useradd --uid 10001 --gid 10001 --no-create-home forecast \
    && install -d -o 10001 -g 10001 /var/lib/forecast
COPY forecast ./forecast
COPY tests ./tests
COPY --chmod=0755 run_tests.sh .
COPY --chmod=0644 artifacts/model.joblib ./artifacts/model.joblib
COPY data/cs-train ./data/cs-train
USER 10001:10001
RUN ./run_tests.sh
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=4s --start-period=10s --retries=3 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3).close()"]
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "180", "forecast.api:create_app()"]
