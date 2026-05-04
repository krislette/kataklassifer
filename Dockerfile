# Base image — slim Python, matches your dev environment
FROM python:3.11-slim

# HuggingFace Spaces runs as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Copy and install dependencies first (layer cache benefit)
COPY --chown=user api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Pre-download the emotion model into the image during build
# This is the KEY step that eliminates cold-start downloads
RUN python -c "\
from transformers import pipeline; \
pipeline('text-classification', \
         model='j-hartmann/emotion-english-distilroberta-base', \
         top_k=1)"

# Copy model artifacts and API code
COPY --chown=user models/ ./models/
COPY --chown=user api/ ./api/

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Start the server on port 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
