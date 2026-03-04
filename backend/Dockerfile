# Usar una imagen de Python ligera
FROM python:3.12-slim

# Evita que Python genere archivos .pyc y habilita el log en tiempo real
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema (necesarias para MySQL y compilación)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    ca-certificates \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto por el que escuchará (Render inyecta PORT)
EXPOSE 8000

# Comando para ejecutar la aplicación (Render asigna el puerto dinámicamente)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

