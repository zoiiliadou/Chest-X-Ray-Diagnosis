# Use the standard slim Python 3.12 image for optimized container size
FROM python:3.12-slim

# Hugging Face Spaces mandates applications to run under a non-root 'user' with UID 1000
RUN useradd -m -u 1000 user
USER user

# Configure environment variables for local bin execution path
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory for the application runtime
WORKDIR $HOME/app

# Transfer dependency specifications and install them without caching to minimize image bloat
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Synchronize the remaining application codebase
COPY --chown=user . .

# Expose and bind the FastAPI interface to port 7860 (Hugging Face standard)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
