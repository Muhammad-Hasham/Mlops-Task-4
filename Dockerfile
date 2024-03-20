# Use an official Python runtime as a base image
FROM python:3.11.5

# Set the working directory in the container
WORKDIR /app

# Copy the contents of the local src directory to the working directory in the container
COPY src/ /app

# Copy the requirements.txt file from the local directory to the working directory in the container
COPY requirements.txt /app/requirements.txt

# Install required Python packages
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose the Flask port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py

# Command to run the Flask application
CMD ["flask", "run", "--host", "0.0.0.0"]
