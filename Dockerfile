FROM python:3.12

# Install system-level dependencies
RUN apt update && apt install -y python3.12-distutils

# Set working directory and copy files
WORKDIR /app
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

CMD ["python", "app.py"]