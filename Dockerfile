FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m sentinel && chown -R sentinel /app
USER sentinel
EXPOSE 8000
# Trains on the real PhiUSIIL data on first start (downloads ~55 MB), then serves.
CMD ["sh", "-c", "[ -f models/sentinel_mlp.npz ] || python -m sentinel train --quiet; python -m sentinel serve --host 0.0.0.0 --port 8000"]
