"""Script para ejecutar inferencia en tiempo real."""
import sys
from pathlib import Path
import torch

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.capture import LandmarkExtractor
from src.data.preprocessing import LandmarkPreprocessor
from src.data.dataset import SignLanguageDataset
from src.models.lstm_classifier import LSTMSignClassifier
from src.inference.realtime import RealtimePredictor
from src.utils.config import config

def main():
    """Función principal de inferencia."""
    print("\n" + "="*60)
    print("INFERENCIA EN TIEMPO REAL")
    print("="*60 + "\n")
    
    # Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}\n")
    
    # Cargar dataset para obtener clases
    print("Cargando configuración...")
    temp_dataset = SignLanguageDataset(
        data_dir=config.DATA_PROCESSED / "train_split"
    )
    num_classes = temp_dataset.num_classes
    label_map = temp_dataset.idx_to_label
    
    print(f"Clases reconocidas: {list(label_map.values())}\n")
    
    # Cargar modelo
    print("Cargando modelo...")
    model = LSTMSignClassifier(
        input_size=config.FEATURE_DIM,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=num_classes,
        dropout=config.DROPOUT,
        bidirectional=config.BIDIRECTIONAL
    )
    
    checkpoint_path = config.MODELS_DIR / 'best_model.pth'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Modelo cargado (Val Acc: {checkpoint['val_acc']:.4f})\n")
    
    # Preprocesador
    preprocessor = LandmarkPreprocessor()
    # Cargar scaler ajustado (si fue guardado)
    # En producción, deberías guardar el scaler con el modelo
    
    # Extractor de landmarks
    extractor = LandmarkExtractor()
    
    # Predictor
    predictor = RealtimePredictor(
        model=model,
        extractor=extractor,
        preprocessor=preprocessor,
        label_map=label_map,
        device=device,
        sequence_length=config.SEQUENCE_LENGTH,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        smoothing_window=config.SMOOTHING_WINDOW
    )
    
    # Ejecutar
    try:
        predictor.run()
    except KeyboardInterrupt:
        print("\n\nInferencia interrumpida por el usuario")
    
    print("\n" + "="*60)
    print("INFERENCIA FINALIZADA")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
