"""Clasificador LSTM para secuencias de landmarks."""
import torch
import torch.nn as nn
from typing import Tuple

class LSTMSignClassifier(nn.Module):
    """Clasificador LSTM bidireccional para señas."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Inicializa el modelo.
        
        Args:
            input_size: Dimensión de features por frame
            hidden_size: Tamaño de la capa oculta LSTM
            num_layers: Número de capas LSTM
            num_classes: Número de clases a predecir
            dropout: Probabilidad de dropout
            bidirectional: Si usar LSTM bidireccional
        """
        super(LSTMSignClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # Capa de entrada
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Factor de bidireccionalidad
        self.direction_factor = 2 if bidirectional else 1
        
        # Capas fully connected
        self.fc1 = nn.Linear(
            hidden_size * self.direction_factor,
            hidden_size
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de shape (batch, seq_len, input_size)
            hidden: Estado oculto inicial (opcional)
            
        Returns:
            Logits de shape (batch, num_classes)
        """
        batch_size, seq_len, _ = x.size()
        
        # Normalización
        x = x.permute(0, 2, 1)  # (batch, input_size, seq_len)
        x = self.input_bn(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, input_size)
        
        # LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        
        # Tomar el último output
        # Si es bidireccional, concatenar última salida forward y backward
        if self.bidirectional:
            # lstm_out: (batch, seq_len, hidden*2)
            last_output = lstm_out[:, -1, :]
        else:
            last_output = lstm_out[:, -1, :]
        
        # Clasificación
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def init_hidden(self, batch_size: int, device: torch.device):
        """
        Inicializa el estado oculto.
        
        Args:
            batch_size: Tamaño del batch
            device: Dispositivo (CPU/GPU)
            
        Returns:
            Tupla (h_0, c_0) de estados ocultos
        """
        num_directions = self.direction_factor
        
        h_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size
        ).to(device)
        
        c_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size
        ).to(device)
        
        return (h_0, c_0)
