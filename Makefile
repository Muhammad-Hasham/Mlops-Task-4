# Define variables
VENV_NAME := myenv
PYTHON := python3
PIP := $(VENV_NAME)/bin/pip
DOCKER := docker
IMAGE_NAME := my_flask_app

# Set default target
.PHONY: all
all: venv install docker-build docker-run

# Create a virtual environment
venv:
    $(PYTHON) -m venv $(VENV_NAME)

# Install dependencies
install:
    $(PIP) install -r requirements.txt

# Build Docker image
docker-build:
    $(DOCKER) build -t $(IMAGE_NAME) .

# Run Docker container
docker-run:
    $(DOCKER) run -p 5000:5000 $(IMAGE_NAME)

# Clean up
clean:
    rm -rf $(VENV_NAME)

# Define targets that do not create files
.PHONY: venv install docker-build docker-run clean
