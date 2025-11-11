"""Dataset PyTorch para secuencias de landmarks."""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split

class SignLanguageDataset(Dataset):
    """Dataset para secuencias de landmarks de señas."""
    
    def __init__(
        self,
        data_dir: Path,
        preprocessor=None,
        augment: bool = False
    ):
        """
        Inicializa el dataset.
        
        Args:
            data_dir: Directorio con archivos .npy organizados por clase
            preprocessor: Instancia de LandmarkPreprocessor
            augment: Si aplicar data augmentation
        """
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.augment = augment
        
        # Cargar archivos y etiquetas
        self.samples = []
        self.labels = []
        self.label_to_idx = {}
        
        # Escanear directorios
        for label_idx, label_dir in enumerate(sorted(self.data_dir.iterdir())):
            if label_dir.is_dir():
                label_name = label_dir.name
                self.label_to_idx[label_name] = label_idx
                
                for sample_file in label_dir.glob("*.npy"):
                    self.samples.append(sample_file)
                    self.labels.append(label_idx)
        
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        
        print(f"Dataset cargado: {len(self.samples)} muestras, {self.num_classes} clases")
        print(f"Clases: {list(self.label_to_idx.keys())}")
    
    def __len__(self) -> int:
        """Retorna el número de muestras."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Obtiene una muestra.
        
        Args:
            idx: Índice de la muestra
            
        Returns:
            Tupla (secuencia, etiqueta)
        """
        # Cargar secuencia
        sequence = np.load(self.samples[idx])
        label = self.labels[idx]
        
        # Preprocesar
        if self.preprocessor:
            sequence = self.preprocessor.normalize_by_reference(sequence)
            if self.preprocessor.fitted:
                sequence = self.preprocessor.transform(sequence)
            
            # Augmentation
            if self.augment:
                sequence = self.preprocessor.augment_sequence(sequence)
        
        # Convertir a tensor
        sequence_tensor = torch.FloatTensor(sequence)
        
        return sequence_tensor, label
    
    @staticmethod
    def get_dataloaders(
        data_dir: Path,
        preprocessor,
        batch_size: int = 32,
        train_split: float = 0.8,
        random_seed: int = 42
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Crea dataloaders de entrenamiento y validación.
        
        Args:
            data_dir: Directorio con datos procesados
            preprocessor: Instancia de LandmarkPreprocessor
            batch_size: Tamaño de batch
            train_split: Proporción para entrenamiento
            random_seed: Semilla aleatoria
            
        Returns:
            Tupla (train_loader, val_loader)
        """
        # Cargar todos los archivos
        all_files = []
        all_labels = []
        
        for label_dir in sorted(data_dir.iterdir()):
            if label_dir.is_dir():
                files = list(label_dir.glob("*.npy"))
                all_files.extend(files)
                all_labels.extend([label_dir.name] * len(files))
        
        # Split estratificado
        train_files, val_files, train_labels, val_labels = train_test_split(
            all_files,
            all_labels,
            train_size=train_split,
            random_state=random_seed,
            stratify=all_labels
        )
        
        # Ajustar preprocessor con datos de entrenamiento
        if preprocessor and not preprocessor.fitted:
            train_sequences = np.array([np.load(f) for f in train_files])
            preprocessor.fit_scaler(train_sequences)
        
        # Crear datasets
        # Guardar archivos temporales para cada split
        train_dir = data_dir / "train_split"
        val_dir = data_dir / "val_split"
        
        for split_dir, files, labels in [
            (train_dir, train_files, train_labels),
            (val_dir, val_files, val_labels)
        ]:
            split_dir.mkdir(exist_ok=True)
            for file, label in zip(files, labels):
                label_dir = split_dir / label
                label_dir.mkdir(exist_ok=True)
                
                # Copiar archivo
                import shutil
                shutil.copy(file, label_dir / file.name)
        
        train_dataset = SignLanguageDataset(
            train_dir,
            preprocessor=preprocessor,
            augment=True
        )
        
        val_dataset = SignLanguageDataset(
            val_dir,
            preprocessor=preprocessor,
            augment=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader
