# Use full Python image (includes build tools & common libs)
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy your entire application code into the container
COPY . .

RUN pip install --no-cache-dir flask transformers torch

# Upgrade pip and install required Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir flask transformers torch

# Expose the port Cloud Run expects
EXPOSE 8080

# Start the Flask application
CMD ["python", "app.py"]
