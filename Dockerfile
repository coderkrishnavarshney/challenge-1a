
FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY src/ src/
COPY run.sh .

# models will be mounted by user
VOLUME ["/app/models", "/app/input", "/app/output"]

ENTRYPOINT ["bash", "run.sh"]
