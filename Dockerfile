# Use Python 3.12 as base image
FROM python:3.12-slim

# Set working directory in the container
WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (including app.py and models/)
COPY . .

# Expose the Flask port
EXPOSE 5000

# Run the app
CMD ["python", "prod_models_v2.py"]
