FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    redis-server \
    build-essential \
    libffi-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Copy project files
COPY . /app

# Install dependencies and local package
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -e .

# Expose FastAPI and Redis ports
EXPOSE 8000 6379

# Create start script
RUN echo '#!/bin/bash\nredis-server --daemonize yes --bind 0.0.0.0\nuvicorn solace.main:app --host 0.0.0.0 --port 8000' > /start.sh \
    && chmod +x /start.sh
    
# Start services using the startup script
CMD ["/start.sh"]