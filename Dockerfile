FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema necesarias para automatizar la BD
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Creamos directorios necesarios para prevenir errores de FileNotFoundError
RUN mkdir -p models data/processed

EXPOSE 80

CMD ["python", "src/bot.py"]
