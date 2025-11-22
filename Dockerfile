FROM python:3.12-slim



# System deps for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app



# Copy requirements first (better cache)
COPY requirements.txt .



RUN pip install --no-cache-dir -r requirements.txt


# Copy project



COPY . .
RUN python -m src.download_model




# Port
EXPOSE 8000



# Run FastAPI
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
