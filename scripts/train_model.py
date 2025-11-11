"""Script para entrenar el modelo."""
import sys
from pathlib import Path
import torch

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import SignLanguageDataset
from src.data.preprocessing import LandmarkPreprocessor
from src.models.lstm_classifier import LSTMSignClassifier
from src.training.trainer import Trainer
from src.training.metrics import MetricsCalculator
from src.utils.config import config

def main():
    """Función principal de entrenamiento."""
    print("\n" + "="*60)
    print("ENTRENAMIENTO DE MODELO DE RECONOCIMIENTO DE SEÑAS")
    print("="*60 + "\n")
    
    # Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}\n")
    
    # Preprocesador
    preprocessor = LandmarkPreprocessor()
    
    # Dataloaders
    print("Cargando datos...")
    train_loader, val_loader = SignLanguageDataset.get_dataloaders(
        data_dir=config.DATA_PROCESSED,
        preprocessor=preprocessor,
        batch_size=config.BATCH_SIZE,
        train_split=config.TRAIN_SPLIT,
        random_seed=config.RANDOM_SEED
    )
    
    num_classes = train_loader.dataset.num_classes
    class_names = list(train_loader.dataset.label_to_idx.keys())
    
    print(f"\nNúmero de clases: {num_classes}")
    print(f"Clases: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}\n")
    
    # Modelo
    print("Creando modelo...")
    model = LSTMSignClassifier(
        input_size=config.FEATURE_DIM,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=num_classes,
        dropout=config.DROPOUT,
        bidirectional=config.BIDIRECTIONAL
    )
    
    print(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Entrenador
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=config.LEARNING_RATE
    )
    
    # Entrenar
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.NUM_EPOCHS,
        save_dir=config.MODELS_DIR
    )
    
    # Graficar historial
    print("\nGenerando gráficos...")
    metrics_calc = MetricsCalculator()
    metrics_calc.plot_training_history(
        train_losses=trainer.history['train_loss'],
        val_losses=trainer.history['val_loss'],
        train_accs=trainer.history['train_acc'],
        val_accs=trainer.history['val_acc'],
        save_path=config.MODELS_DIR / 'training_history.png'
    )
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print(f"Mejor Val Accuracy: {trainer.best_val_acc:.4f}")
    print(f"Modelo guardado en: {config.MODELS_DIR / 'best_model.pth'}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
