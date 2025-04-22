# Use an official PyTorch base image with CUDA support
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Set the working directory in the container
WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    ca-certificates \
    git \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application code (including scripts like downloads.sh)
COPY . .

# Set environment variables for CUDA
ENV CUDA_VISIBLE_DEVICES=0

# Make the downloads.sh script executable
RUN chmod +x ./downloads.sh

# Download required datasets using the downloads.sh script
RUN ./downloads.sh

# Set up the .env file
RUN cp ./.env.template ./.env

# Expose the necessary ports (optional, based on your application)
EXPOSE 5000

# Command to run the application
# Set the default command as the one mentioned in README for `kvcache.py`
CMD ["python", "./kvcache.py", "--kvcache", "file", "--dataset", "squad-train", "--similarity", "bertscore", \
    "--maxKnowledge", "5", "--maxParagraph", "100", "--maxQuestion", "1000", \
    "--modelname", "meta-llama/Llama-3.1-8B-Instruct", "--randomSeed", "0", "--output", "./result_kvcache.txt"]
