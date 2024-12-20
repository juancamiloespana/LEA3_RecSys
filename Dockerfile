# Use an official lightweight Python image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install system dependencies for the container
# For SQLite and Openpyxl dependencies, these may include gcc for some library installations
RUN apt-get update && apt-get install -y \
    gcc \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies specified in the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (optional, for web services)
EXPOSE 8080

# Command to run the Python script when the container starts
CMD ["python", "./despliegue2.py"]