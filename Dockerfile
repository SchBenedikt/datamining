FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY visualization/requirements_streamlit.txt ./visualization/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r visualization/requirements_streamlit.txt

# Copy application code
COPY . .

# Default command (can be overridden by docker-compose)
CMD ["python3"]
