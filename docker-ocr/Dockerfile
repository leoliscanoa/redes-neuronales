FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    libsm6 libxext6 libxrender-dev \
    && apt-get clean


WORKDIR /app

COPY .. /app


COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]