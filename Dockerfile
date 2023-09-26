# Utiliza una imagen base de Python
FROM python:3.10.6

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia la carpeta src al directorio de trabajo en el contenedor
COPY src /app/src

# Copia el archivo de requerimientos a la imagen
COPY requirements.txt .

# Instala las dependencias especificadas
RUN pip install -r requirements.txt

# Instala las dependencias necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copia el archivo main.py a la imagen
COPY main.py .

# Ejecuta el script cuando el contenedor se inicie
CMD ["python", "main.py"]
