"""Preprocesamiento y normalización de landmarks."""
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler

class LandmarkPreprocessor:
    """Normaliza y aumenta secuencias de landmarks."""
    
    def __init__(self):
        """Inicializa el preprocesador."""
        self.scaler = StandardScaler()
        self.fitted = False
    
    def normalize_by_reference(
        self,
        sequence: np.ndarray,
        reference_idx: int = 0
    ) -> np.ndarray:
        """
        Normaliza landmarks respecto a un punto de referencia (ej. hombro).
        
        Args:
            sequence: Array de shape (seq_len, feature_dim)
            reference_idx: Índice del landmark de referencia
            
        Returns:
            Secuencia normalizada
        """
        # Reshape para separar coordenadas
        seq_len, feature_dim = sequence.shape
        n_landmarks = feature_dim // 3
        landmarks = sequence.reshape(seq_len, n_landmarks, 3)
        
        # Centrar respecto al punto de referencia
        reference_point = landmarks[:, reference_idx, :].reshape(seq_len, 1, 3)
        landmarks_centered = landmarks - reference_point
        
        # Escalar por distancia característica (ej. ancho de hombros)
        shoulder_left_idx = 11  # En MediaPipe Pose
        shoulder_right_idx = 12
        
        shoulder_width = np.linalg.norm(
            landmarks[:, shoulder_left_idx, :] - landmarks[:, shoulder_right_idx, :],
            axis=1
        ).reshape(seq_len, 1, 1)
        
        # Evitar división por cero
        shoulder_width = np.where(shoulder_width > 0, shoulder_width, 1.0)
        
        landmarks_scaled = landmarks_centered / shoulder_width
        
        return landmarks_scaled.reshape(seq_len, feature_dim)
    
    def fit_scaler(self, sequences: np.ndarray):
        """
        Ajusta el scaler a las secuencias de entrenamiento.
        
        Args:
            sequences: Array de shape (n_samples, seq_len, feature_dim)
        """
        # Aplanar para ajustar el scaler
        n_samples, seq_len, feature_dim = sequences.shape
        flat_sequences = sequences.reshape(-1, feature_dim)
        
        self.scaler.fit(flat_sequences)
        self.fitted = True
    
    def transform(self, sequence: np.ndarray) -> np.ndarray:
        """
        Normaliza una secuencia usando el scaler ajustado.
        
        Args:
            sequence: Array de shape (seq_len, feature_dim)
            
        Returns:
            Secuencia normalizada
        """
        if not self.fitted:
            raise ValueError("Scaler no ajustado. Llama a fit_scaler primero.")
        
        seq_len, feature_dim = sequence.shape
        sequence_flat = sequence.reshape(-1, feature_dim)
        normalized = self.scaler.transform(sequence_flat)
        
        return normalized.reshape(seq_len, feature_dim)
    
    def augment_sequence(
        self,
        sequence: np.ndarray,
        noise_std: float = 0.01,
        time_stretch_range: Tuple[float, float] = (0.9, 1.1)
    ) -> np.ndarray:
        """
        Aumenta una secuencia con ruido y estiramiento temporal.
        
        Args:
            sequence: Array de shape (seq_len, feature_dim)
            noise_std: Desviación estándar del ruido gaussiano
            time_stretch_range: Rango para estiramiento temporal
            
        Returns:
            Secuencia aumentada
        """
        seq_len, feature_dim = sequence.shape
        
        # Ruido gaussiano
        noise = np.random.normal(0, noise_std, sequence.shape)
        sequence_noisy = sequence + noise
        
        # Estiramiento temporal (time warping simple)
        stretch_factor = np.random.uniform(*time_stretch_range)
        new_len = int(seq_len * stretch_factor)
        indices = np.linspace(0, seq_len - 1, new_len)
        
        sequence_stretched = np.array([
            np.interp(indices, np.arange(seq_len), sequence_noisy[:, i])
            for i in range(feature_dim)
        ]).T
        
        # Redimensionar al tamaño original
        if new_len != seq_len:
            final_indices = np.linspace(0, new_len - 1, seq_len)
            sequence_stretched = np.array([
                np.interp(final_indices, np.arange(new_len), sequence_stretched[:, i])
                for i in range(feature_dim)
            ]).T
        
        return sequence_stretched
