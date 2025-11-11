"""Script para recolectar datos de entrenamiento."""
import sys
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.capture import LandmarkExtractor, VideoCapture
from src.utils.config import config

def main():
    """Función principal de recolección."""
    print("\n" + "="*60)
    print("RECOLECCIÓN DE DATOS PARA ENTRENAMIENTO")
    print("="*60 + "\n")
    
    # Configuración
    signs = input("Ingresa las señas a capturar (separadas por coma): ").split(',')
    signs = [s.strip() for s in signs]
    
    samples_per_sign = int(input(f"Muestras por seña (default: {config.SAMPLES_PER_SIGN}): ") 
                          or config.SAMPLES_PER_SIGN)
    
    print(f"\nSeñas: {signs}")
    print(f"Muestras por seña: {samples_per_sign}")
    print(f"Longitud de secuencia: {config.SEQUENCE_LENGTH} frames\n")
    
    # Inicializar capturador
    extractor = LandmarkExtractor()
    capturer = VideoCapture(extractor, sequence_length=config.SEQUENCE_LENGTH)
    
    # Capturar por cada seña
    for sign in signs:
        print(f"\n{'='*60}")
        print(f"Capturando seña: {sign}")
        print(f"{'='*60}\n")
        
        for sample_id in range(samples_per_sign):
            capturer.capture_sequence(
                label=sign,
                sequence_id=sample_id,
                save_dir=config.DATA_PROCESSED
            )
            
            print(f"\nProgreso: {sample_id + 1}/{samples_per_sign}\n")
    
    print("\n" + "="*60)
    print("RECOLECCIÓN COMPLETADA")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
