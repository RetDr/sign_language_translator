"""Loop de entrenamiento del modelo."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, List
from .metrics import MetricsCalculator

class Trainer:
    """Entrena el clasificador de señas."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        """
        Inicializa el entrenador.
        
        Args:
            model: Modelo a entrenar
            device: Dispositivo (CPU/GPU)
            learning_rate: Tasa de aprendizaje
            weight_decay: Regularización L2
        """
        self.model = model.to(device)
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
        self.metrics_calc = MetricsCalculator()
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Entrena una época.
        
        Args:
            dataloader: DataLoader de entrenamiento
            
        Returns:
            Diccionario con métricas de la época
        """
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc="Entrenamiento")
        for batch_idx, (sequences, labels) in enumerate(pbar):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Métricas
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Actualizar barra
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Valida el modelo.
        
        Args:
            dataloader: DataLoader de validación
            
        Returns:
            Diccionario con métricas de validación
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(dataloader, desc="Validación")
        for sequences, labels in pbar:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            
            # Métricas
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        metrics = self.metrics_calc.calculate_metrics(
            all_labels,
            all_preds,
            all_probs
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: Path
    ):
        """
        Loop completo de entrenamiento.
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            num_epochs: Número de épocas
            save_dir: Directorio para guardar checkpoints
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Iniciando entrenamiento por {num_epochs} épocas")
        print(f"Dispositivo: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nÉpoca {epoch}/{num_epochs}")
            print("-" * 40)
            
            # Entrenar
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            
            # Validar
            val_metrics = self.validate(val_loader)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Scheduler
            self.scheduler.step(val_metrics['accuracy'])
            
            # Imprimir resumen
            print(f"\nTrain Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            if 'top_3_accuracy' in val_metrics:
                print(f"Val Top-3 Acc: {val_metrics['top_3_accuracy']:.4f}")
            
            # Guardar mejor modelo
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_metrics['accuracy'],
                    'history': self.history
                }
                torch.save(
                    checkpoint,
                    save_dir / 'best_model.pth'
                )
                print(f"✓ Mejor modelo guardado (Val Acc: {self.best_val_acc:.4f})")
            
            # Guardar checkpoint periódico
            if epoch % 10 == 0:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    },
                    save_dir / f'checkpoint_epoch_{epoch}.pth'
                )
        
        print(f"\n{'='*60}")
        print(f"Entrenamiento completado")
        print(f"Mejor Val Accuracy: {self.best_val_acc:.4f}")
        print(f"{'='*60}\n")
