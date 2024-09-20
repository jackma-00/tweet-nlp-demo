# Name of the Docker image
IMAGE_NAME = tweet-nlp-demo

# Create a virtual environment
venv:
    python3 -m venv .venv

# Install the required Python packages
install:
    source ./.venv/bin/activate && \
    pip install --upgrade pip &&\
    pip install -r requirements.txt

# Build the Docker image
build:
    docker build -t $(IMAGE_NAME) .

# Run the Docker container
run:
    docker run --rm -it $(IMAGE_NAME)

# Clean up Docker images
clean:
    docker rmi $(IMAGE_NAME)