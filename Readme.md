# 🧠 Sistema de Reconocimiento de Lenguaje de Señas

Reconoce **lenguaje de señas en tiempo real** usando **MediaPipe Holistic** y un **clasificador LSTM bidireccional**.  
El sistema es **modular, extensible y adecuado para investigación, docencia y prototipado avanzado** en visión por computador.

---

## 🚀 Características

- 🎯 Extracción en tiempo real de **543 landmarks** (manos, rostro y pose corporal) usando **MediaPipe Holistic**.  
- 🔁 **Clasificador LSTM bidireccional** para secuencias temporales de keypoints.  
- 🧩 **Pipeline modular** para recolección de datos, preprocesamiento, entrenamiento y evaluación.  
- 📸 **Inferencia interactiva desde webcam** con visualización de confianza y predicción.  
- 🧠 **Data augmentation** y **normalización avanzada** para mejorar la robustez del modelo.  
- 📊 Métricas: *Top-k accuracy*, matriz de confusión y reporte por clase.  
- ⚙️ Arquitectura **escalable y fácil de mantener**, con soporte a datasets externos (WLASL, PHOENIX-2014T, propios).

---

## 🧰 Instalación

### 1. Crea un entorno virtual

python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

### 2. Instala las dependencias

pip install -r requirements.txt
⚡ Uso rápido
# 1. Recolectar datos

python scripts/collect_data.py
Ingresa las señas separadas por coma (ejemplo: hola,adios,gracias)

Define el número de muestras por seña.

Sigue las instrucciones en pantalla.

# 2. Entrenar el modelo
python scripts/train_model.py
El modelo entrenado se guarda en:
models/best_model.pth
# 3. Inferencia en tiempo real
python scripts/run_inference.py
Pulsa q para salir.
Pulsa r para reiniciar el buffer de predicción.

## 🗂️ Estructura del proyecto
sign_language_translator/
├── data/           # Datos crudos y secuencias procesadas
├── src/            # Código fuente modular
│   ├── data/       # Captura, datasets y procesamiento
│   ├── models/     # Modelos LSTM, Transformer, etc.
│   ├── training/   # Entrenamiento y métricas
│   ├── inference/  # Inferencia en tiempo real
│   └── utils/      # Configuración y utilidades
├── scripts/        # Scripts CLI
└── notebooks/      # Notebooks para exploración y análisis
## ⚙️ Configuración
Edita el archivo:
src/utils/config.py
Puedes cambiar:

Hiperparámetros del modelo (capas, tamaño de secuencia, etc.)

Rutas de datos, salidas y modelos

Parámetros de entrenamiento y tolerancia de inferencia

## 🔧 Extensión y Escalabilidad
🔹 Agregar Transformer con CTC
Implementa src/models/transformer.py como arquitectura tipo PHOENIX-2014T para traducción continua de video → glosas → texto.

🔹 Soporte a LSC (Lengua de Señas Colombiana)
Recolecta tu propio corpus y anótalo con ELAN.

Ajusta la lista de glosas y vocabulario en la configuración del pipeline.

## 📚 Ejemplo de sistemas soportados
Extracción robusta de landmarks con MediaPipe Holistic (documentación oficial)

Entrenamiento en tu propio lenguaje de señas o datasets públicos:
WLASL
PHOENIX-2014T
Integración directa con notebooks para prototipado y visualización avanzada.

## 🔗 Referencias
📘 MediaPipe Holistic
📄 Sign Language Transformers (Paper)
📂 WLASL Dataset
📂 PHOENIX-2014T Dataset (traducción continua)

## 🧩 Ejemplos comunitarios: busca proyectos SignLanguageRecognition en GitHub.
👥 Créditos
Proyecto desarrollado con fines académicos, de docencia e investigación en:
Visión por computador
Aprendizaje profundo
Accesibilidad e inclusión tecnológica
