# Base image: python:3.9-slim (lightweight and secure)
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any) and clean up to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY src/ src/

# Create a non-root user and switch to it (Security Best Practice)
RUN useradd -m appuser
USER appuser

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application using Gunicorn (Production Server)
# We bind to 0.0.0.0 so it is accessible outside the container
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.inference_api:app"]