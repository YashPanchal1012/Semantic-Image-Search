FROM python:3.10-slim

WORKDIR /app

# Install git for openai-clip installation
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure Images directory exists
RUN mkdir -p Images

ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
