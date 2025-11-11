"""Métricas de evaluación."""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score
)
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsCalculator:
    """Calcula métricas de clasificación."""
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None,
        class_names: List[str] = None
    ) -> Dict[str, float]:
        """
        Calcula métricas de clasificación.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones
            y_prob: Probabilidades (para top-k)
            class_names: Nombres de las clases
            
        Returns:
            Diccionario con métricas
        """
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Top-k accuracy
        if y_prob is not None:
            for k in [3, 5]:
                if y_prob.shape[1] >= k:
                    metrics[f'top_{k}_accuracy'] = top_k_accuracy_score(
                        y_true,
                        y_prob,
                        k=k
                    )
        
        # Reporte por clase
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        metrics['precision_macro'] = report['macro avg']['precision']
        metrics['recall_macro'] = report['macro avg']['recall']
        metrics['f1_macro'] = report['macro avg']['f1-score']
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_path: str = None
    ):
        """
        Grafica la matriz de confusión.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones
            class_names: Nombres de las clases
            save_path: Ruta para guardar la figura
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Matriz de Confusión')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Predicción')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
    
    @staticmethod
    def plot_training_history(
        train_losses: List[float],
        val_losses: List[float],
        train_accs: List[float],
        val_accs: List[float],
        save_path: str = None
    ):
        """
        Grafica el historial de entrenamiento.
        
        Args:
            train_losses: Pérdidas de entrenamiento
            val_losses: Pérdidas de validación
            train_accs: Accuracy de entrenamiento
            val_accs: Accuracy de validación
            save_path: Ruta para guardar la figura
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Pérdida
        ax1.plot(train_losses, label='Entrenamiento')
        ax1.plot(val_losses, label='Validación')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Pérdida')
        ax1.set_title('Evolución de la Pérdida')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(train_accs, label='Entrenamiento')
        ax2.plot(val_accs, label='Validación')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Evolución del Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
