FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY models/ models/
COPY src/inference.py .
CMD ["python", "inference.py"]