# Use Python 3.10 with a lightweight Debian-based image
FROM python:3.10-slim-buster

# Update system packages and install required dependencies
RUN apt update -y && apt install -y awscli

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the training script to train and save the model
CMD ["python3", "training.py"]

# Start the application (user inputs & prediction)
CMD ["python3", "app.py"]
