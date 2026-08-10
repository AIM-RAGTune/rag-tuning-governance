FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt pyproject.toml README.md /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
RUN pip install --no-cache-dir -e .
CMD ["make", "validate-publication"]
