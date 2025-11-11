# Sistema de Reconocimiento de Lenguaje de Señas

Sistema completo de reconocimiento de lenguaje de señas usando MediaPipe Holistic y LSTM bidireccional.

## Características

- Extracción de 543 landmarks (pose, rostro, manos) con MediaPipe Holistic
- Clasificador LSTM bidireccional temporal
- Pipeline completo de entrenamiento y evaluación
- Inferencia en tiempo real desde webcam
- Data augmentation y normalización robusta
- Métricas completas (Top-k accuracy, matriz de confusión)

## Instalación

# Crear entorno virtual
python -m venv venv
source venv/bin/activate 
En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

### 1. Recolectar datos
python scripts/collect_data.py

- Ingresa las señas separadas por coma (ej: hola,adios,gracias)
- Define número de muestras por seña
- Sigue las instrucciones en pantalla

### 2. Entrenar modelo
python scripts/train_model.py

El modelo se guardará automáticamente en `models/best_model.pth`.

### 3. Inferencia en tiempo real

python scripts/run_inference.py

