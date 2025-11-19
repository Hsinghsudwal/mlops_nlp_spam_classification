# Use official Python base image
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create folder for uploaded files
RUN mkdir -p user_app/uploads

# Expose the port
EXPOSE 9000

CMD ["gunicorn", "--bind", "0.0.0.0:9000", "app:app"]

