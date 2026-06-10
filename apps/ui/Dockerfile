# apps/ui/Dockerfile — Streamlit dashboard image (thin client of the API).
#
# Build:  docker build -f apps/ui/Dockerfile -t kyc-ui .  (run from repo root)
# Run:    docker compose up ui
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Recreate repo layout — dashboard.py resolves personas via parents[2].
WORKDIR /repo/apps/ui

COPY apps/ui/requirements.txt /repo/apps/ui/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY apps/ui/ /repo/apps/ui/
COPY personas/ /repo/personas/

EXPOSE 8501

# Headless + no telemetry. CORS/XSRF off so the browser can talk to the API
# (the API is on a different docker-compose port; same flags as run-dashboard.sh).
CMD ["streamlit", "run", "dashboard.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
