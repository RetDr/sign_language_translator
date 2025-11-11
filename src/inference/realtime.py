"""Inferencia en tiempo real desde webcam."""
import cv2
import torch
import numpy as np
from collections import deque
from typing import List, Dict
import time

class RealtimePredictor:
    """Predictor en tiempo real para señas."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        extractor,
        preprocessor,
        label_map: Dict[int, str],
        device: torch.device,
        sequence_length: int = 30,
        confidence_threshold: float = 0.7,
        smoothing_window: int = 5
    ):
        """
        Inicializa el predictor en tiempo real.
        
        Args:
            model: Modelo entrenado
            extractor: LandmarkExtractor
            preprocessor: LandmarkPreprocessor
            label_map: Mapeo de índice a nombre de clase
            device: Dispositivo (CPU/GPU)
            sequence_length: Longitud de secuencia
            confidence_threshold: Umbral de confianza
            smoothing_window: Ventana para suavizar predicciones
        """
        self.model = model.to(device)
        self.model.eval()
        
        self.extractor = extractor
        self.preprocessor = preprocessor
        self.label_map = label_map
        self.device = device
        
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Buffer para secuencia
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # Buffer para suavizar predicciones
        self.prediction_buffer = deque(maxlen=smoothing_window)
        
        # FPS tracking
        self.fps_buffer = deque(maxlen=30)
        self.prev_time = time.time()
    
    def predict_frame(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Procesa un frame y retorna predicción.
        
        Args:
            frame: Frame BGR de OpenCV
            
        Returns:
            Diccionario con predicción y confianza
        """
        # Extraer landmarks
        landmarks = self.extractor.extract_landmarks(frame)
        
        if landmarks is None:
            return {'label': None, 'confidence': 0.0, 'ready': False}
        
        # Agregar al buffer
        self.frame_buffer.append(landmarks)
        
        # Esperar a tener secuencia completa
        if len(self.frame_buffer) < self.sequence_length:
            return {
                'label': 'Capturando...',
                'confidence': 0.0,
                'ready': False,
                'progress': len(self.frame_buffer) / self.sequence_length
            }
        
        # Preparar secuencia
        sequence = np.array(list(self.frame_buffer))
        
        # Preprocesar
        sequence = self.preprocessor.normalize_by_reference(sequence)
        if self.preprocessor.fitted:
            sequence = self.preprocessor.transform(sequence)
        
        # Inferencia
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            outputs = self.model(sequence_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            
            confidence = confidence.item()
            pred_idx = pred_idx.item()
        
        # Suavizar predicción
        if confidence >= self.confidence_threshold:
            self.prediction_buffer.append(pred_idx)
            
            if len(self.prediction_buffer) >= 3:
                # Votar por la predicción más frecuente
                smoothed_pred = max(set(self.prediction_buffer), 
                                   key=list(self.prediction_buffer).count)
                label = self.label_map[smoothed_pred]
            else:
                label = self.label_map[pred_idx]
        else:
            label = "Incierto"
        
        return {
            'label': label,
            'confidence': confidence,
            'ready': True,
            'all_probs': {self.label_map[i]: probs[0][i].item() 
                         for i in range(len(self.label_map))}
        }
    
    def run(self):
        """Ejecuta el loop de inferencia en tiempo real."""
        cap = cv2.VideoCapture(0)
        
        print("\n" + "="*60)
        print("RECONOCIMIENTO DE SEÑAS EN TIEMPO REAL")
        print("="*60)
        print("Presiona 'q' para salir")
        print("Presiona 'r' para reiniciar buffer\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time)
            self.prev_time = curr_time
            self.fps_buffer.append(fps)
            avg_fps = np.mean(self.fps_buffer)
            
            # Predicción
            prediction = self.predict_frame(frame)
            
            # Dibujar landmarks
            frame_viz = self.extractor.draw_landmarks(frame.copy(), None)
            
            # Overlay de información
            overlay = frame_viz.copy()
            alpha = 0.6
            
            # Panel superior
            cv2.rectangle(overlay, (0, 0), (frame_viz.shape[1], 120), (0, 0, 0), -1)
            frame_viz = cv2.addWeighted(overlay, alpha, frame_viz, 1 - alpha, 0)
            
            # FPS
            cv2.putText(
                frame_viz,
                f"FPS: {avg_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Predicción
            if prediction['ready']:
                label = prediction['label']
                confidence = prediction['confidence']
                
                color = (0, 255, 0) if confidence >= self.confidence_threshold else (0, 165, 255)
                
                cv2.putText(
                    frame_viz,
                    f"Seña: {label}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    color,
                    3
                )
                
                cv2.putText(
                    frame_viz,
                    f"Confianza: {confidence:.2%}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )
                
                # Barra de confianza
                bar_width = int(300 * confidence)
                cv2.rectangle(
                    frame_viz,
                    (frame_viz.shape[1] - 320, 30),
                    (frame_viz.shape[1] - 320 + bar_width, 50),
                    color,
                    -1
                )
                cv2.rectangle(
                    frame_viz,
                    (frame_viz.shape[1] - 320, 30),
                    (frame_viz.shape[1] - 20, 50),
                    (255, 255, 255),
                    2
                )
            else:
                if 'progress' in prediction:
                    progress = prediction['progress']
                    cv2.putText(
                        frame_viz,
                        f"Capturando: {progress:.0%}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 0),
                        2
                    )
            
            cv2.imshow("Reconocimiento de Señas", frame_viz)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.frame_buffer.clear()
                self.prediction_buffer.clear()
                print("Buffer reiniciado")
        
        cap.release()
        cv2.destroyAllWindows()
