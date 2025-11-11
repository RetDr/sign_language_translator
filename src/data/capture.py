"""Captura de video y extracción de landmarks con MediaPipe Holistic."""
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, Dict
from pathlib import Path

class LandmarkExtractor:
    """Extrae landmarks de pose, manos y rostro usando MediaPipe Holistic."""
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Inicializa el extractor de landmarks.
        
        Args:
            min_detection_confidence: Umbral de confianza para detección
            min_tracking_confidence: Umbral de confianza para seguimiento
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1  # 0=lite, 1=full, 2=heavy
        )
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae landmarks de un frame.
        
        Args:
            frame: Frame BGR de OpenCV
            
        Returns:
            Array de shape (543, 3) con coordenadas (x, y, z) o None si falla
        """
        # Convertir BGR a RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Procesar con Holistic
        results = self.holistic.process(image_rgb)
        
        # Extraer coordenadas
        landmarks = []
        
        # Pose (33 puntos)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 33 * 3)
        
        # Rostro (468 puntos)
        if results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 468 * 3)
        
        # Mano izquierda (21 puntos)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 21 * 3)
        
        # Mano derecha (21 puntos)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 21 * 3)
        
        return np.array(landmarks, dtype=np.float32)
    
    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Dibuja landmarks sobre el frame.
        
        Args:
            frame: Frame BGR de OpenCV
            landmarks: Landmarks extraídos
            
        Returns:
            Frame con landmarks dibujados
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)
        
        # Dibujar pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles
                .get_default_pose_landmarks_style()
            )
        
        # Dibujar manos
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )
        
        return frame
    
    def __del__(self):
        """Libera recursos."""
        self.holistic.close()


class VideoCapture:
    """Captura videos y extrae secuencias de landmarks."""
    
    def __init__(
        self,
        extractor: LandmarkExtractor,
        sequence_length: int = 30
    ):
        """
        Inicializa el capturador de video.
        
        Args:
            extractor: Instancia de LandmarkExtractor
            sequence_length: Número de frames por secuencia
        """
        self.extractor = extractor
        self.sequence_length = sequence_length
    
    def capture_sequence(
        self,
        label: str,
        sequence_id: int,
        save_dir: Path
    ) -> Optional[np.ndarray]:
        """
        Captura una secuencia de landmarks desde la webcam.
        
        Args:
            label: Etiqueta de la seña
            sequence_id: ID de la secuencia
            save_dir: Directorio donde guardar
            
        Returns:
            Array de shape (sequence_length, feature_dim) o None
        """
        cap = cv2.VideoCapture(0)
        sequence = []
        
        print(f"Capturando seña '{label}' (muestra {sequence_id})")
        print("Presiona ESPACIO cuando estés listo, ESC para cancelar")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mostrar instrucciones
            cv2.putText(
                frame,
                f"Seña: {label} | Muestra: {sequence_id}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                "ESPACIO: comenzar | ESC: cancelar",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            cv2.imshow("Captura", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return None
            elif key == 32:  # ESPACIO
                break
        
        # Capturar secuencia
        frames_captured = 0
        while frames_captured < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extraer landmarks
            landmarks = self.extractor.extract_landmarks(frame)
            if landmarks is not None:
                sequence.append(landmarks)
                frames_captured += 1
            
            # Dibujar landmarks
            frame_viz = self.extractor.draw_landmarks(frame.copy(), landmarks)
            
            # Mostrar progreso
            progress = int((frames_captured / self.sequence_length) * 100)
            cv2.putText(
                frame_viz,
                f"Progreso: {progress}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            cv2.imshow("Captura", frame_viz)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(sequence) == self.sequence_length:
            sequence_array = np.array(sequence)
            
            # Guardar
            save_path = save_dir / label / f"{label}_{sequence_id}.npy"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, sequence_array)
            
            print(f"✓ Guardado: {save_path}")
            return sequence_array
        else:
            print(f"✗ Secuencia incompleta ({len(sequence)}/{self.sequence_length})")
            return None
