"""Configuración global del proyecto."""
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    """Hiperparámetros y rutas del proyecto."""
    
    # Rutas
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    
    # Captura de datos
    VIDEO_WIDTH: int = 640
    VIDEO_HEIGHT: int = 480
    FPS: int = 30
    SAMPLES_PER_SIGN: int = 50
    SEQUENCE_LENGTH: int = 30  # Frames por muestra
    
    # Landmarks MediaPipe Holistic
    POSE_LANDMARKS: int = 33
    FACE_LANDMARKS: int = 468
    HAND_LANDMARKS: int = 21  # Por mano
    TOTAL_LANDMARKS: int = POSE_LANDMARKS + FACE_LANDMARKS + 2 * HAND_LANDMARKS
    FEATURE_DIM: int = TOTAL_LANDMARKS * 3  # x, y, z por punto
    
    # Modelo
    HIDDEN_SIZE: int = 256
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.3
    BIDIRECTIONAL: bool = True
    
    # Entrenamiento
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-3
    NUM_EPOCHS: int = 50
    TRAIN_SPLIT: float = 0.8
    RANDOM_SEED: int = 42
    
    # Inferencia
    CONFIDENCE_THRESHOLD: float = 0.7
    SMOOTHING_WINDOW: int = 5  # Frames para suavizar predicción
    
    def __post_init__(self):
        """Crear directorios si no existen."""
        self.DATA_RAW.mkdir(parents=True, exist_ok=True)
        self.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

config = Config()
