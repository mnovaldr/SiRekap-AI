FROM python:3.12-slim

WORKDIR /app

# Install dependencies needed for some packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy app code
COPY . /app

# Install required python packages
RUN pip install --no-cache-dir fastapi uvicorn pandas scikit-learn Sastrawi

# Expose the port the app runs on
EXPOSE 8001

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
