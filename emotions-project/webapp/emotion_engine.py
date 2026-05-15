"""
Silnik detekcji emocji - reuzywalna logika dla aplikacji webowej.

Laduje raz model CNN (EmotionCNN) oraz wszystkie dostepne detektory twarzy MediaPipe.
Udostepnia metode process_frame(frame_bgr, detector_name) ktora zwraca:
  - annotated frame (BGR, z naniesionymi ramkami i etykietami)
  - liste detekcji (bbox + emocja + prawdopodobienstwa per klasa)
"""

import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Zaladuj model CNN z katalogu projektu (dwa poziomy wyzej znajduje sie model.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from model import EmotionCNN  # noqa: E402


EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

EMOTION_COLORS_BGR = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 100, 0),
    'Fear': (128, 0, 128),
    'Happy': (0, 255, 255),
    'Neutral': (200, 200, 200),
    'Sad': (255, 0, 0),
    'Surprise': (0, 165, 255),
}

# Mapowanie nazwa -> plik .tflite. Wszystkie modele lezace w detectors/
AVAILABLE_DETECTORS = {
    'blaze_short': {
        'label': 'BlazeFace Short Range',
        'file': 'blaze_face_short_range.tflite',
        'description': 'Szybki, twarze blisko kamery (do ~2 m). Domyslny w detect.py.',
    },
    'blaze_full': {
        'label': 'BlazeFace Full Range',
        'file': 'blaze_face_full_range.tflite',
        'description': 'Wiekszy zasieg (do ~5 m), pelna jakosc.',
    },
    'blaze_full_sparse': {
        'label': 'BlazeFace Full Range Sparse',
        'file': 'blaze_face_full_range_sparse.tflite',
        'description': 'Wiekszy zasieg, mniejszy model (~600 KB), szybszy.',
    },
}


class EmotionEngine:
    def __init__(self, model_path: str | None = None, detectors_dir: str | None = None,
                 min_detection_confidence: float = 0.6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or str(PROJECT_ROOT / 'models' / 'emotion_model.pth')
        self.detectors_dir = detectors_dir or str(PROJECT_ROOT / 'detectors')
        self.min_detection_confidence = min_detection_confidence

        self.cnn = EmotionCNN(num_classes=len(EMOTION_CLASSES))
        self.cnn.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.cnn.to(self.device)
        self.cnn.eval()

        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        # Cache detektorow - laduje przy pierwszym uzyciu
        self._detectors: dict[str, mp_vision.FaceDetector] = {}

    def list_detectors(self) -> list[dict]:
        return [
            {'id': key, **meta}
            for key, meta in AVAILABLE_DETECTORS.items()
        ]

    def _get_detector(self, detector_name: str) -> mp_vision.FaceDetector:
        if detector_name not in AVAILABLE_DETECTORS:
            raise ValueError(f"Nieznany detektor: {detector_name}")
        if detector_name not in self._detectors:
            cfg = AVAILABLE_DETECTORS[detector_name]
            asset_path = os.path.join(self.detectors_dir, cfg['file'])
            base_options = mp_python.BaseOptions(model_asset_path=asset_path)
            options = mp_vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=self.min_detection_confidence,
            )
            self._detectors[detector_name] = mp_vision.FaceDetector.create_from_options(options)
        return self._detectors[detector_name]

    def _classify_face(self, gray_full: np.ndarray, bbox: tuple[int, int, int, int]
                       ) -> tuple[str, dict[str, float]] | None:
        x, y, w, h = bbox
        h_img, w_img = gray_full.shape[:2]
        x, y = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        if x2 <= x or y2 <= y:
            return None
        roi = gray_full[y:y2, x:x2]
        if roi.size == 0:
            return None
        roi_pil = Image.fromarray(roi)
        input_tensor = self.transform(roi_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.cnn(input_tensor)
            probs = F.softmax(output, dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        emotion = EMOTION_CLASSES[idx]
        prob_dict = {cls: float(probs[i]) for i, cls in enumerate(EMOTION_CLASSES)}
        return emotion, prob_dict

    def detect(self, frame_bgr: np.ndarray, detector_name: str) -> list[dict]:
        """Zwraca tylko surowe detekcje (bez rysowania)."""
        detector = self._get_detector(detector_name)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        detections = []
        if not result.detections:
            return detections

        for det in result.detections:
            bbox = det.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            cls = self._classify_face(gray, (x, y, w, h))
            if cls is None:
                continue
            emotion, probs = cls
            detections.append({
                'bbox': [x, y, w, h],
                'emotion': emotion,
                'probabilities': probs,
            })
        return detections

    def draw(self, frame_bgr: np.ndarray, detections: list[dict],
             show_bars: bool = True, label_prefix: str = '') -> np.ndarray:
        """Rysuje ramki + napisy + ewentualne paski prawdopodobienstw."""
        out = frame_bgr.copy()
        for det in detections:
            x, y, w, h = det['bbox']
            emotion = det['emotion']
            color = EMOTION_COLORS_BGR.get(emotion, (255, 255, 255))
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            label = f"{label_prefix}{emotion}"
            cv2.putText(out, label, (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if show_bars and detections:
            # Paski tylko dla najwiekszej twarzy (najwazniejsza)
            largest = max(detections, key=lambda d: d['bbox'][2] * d['bbox'][3])
            probs = largest['probabilities']
            y_off = 30
            bar_w = 150
            overlay = out.copy()
            cv2.rectangle(overlay, (5, 5), (200, 310), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)
            for i, name in enumerate(EMOTION_CLASSES):
                p = probs[name] * 100
                color = EMOTION_COLORS_BGR[name]
                cv2.putText(out, f"{name}: {p:.1f}%", (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.rectangle(out, (10, y_off + 8), (10 + bar_w, y_off + 18),
                              (50, 50, 50), -1)
                fill = int((p / 100) * bar_w)
                cv2.rectangle(out, (10, y_off + 8), (10 + fill, y_off + 18),
                              color, -1)
                y_off += 40
        return out

    def process_frame(self, frame_bgr: np.ndarray, detector_name: str,
                      draw: bool = True, show_bars: bool = True
                      ) -> tuple[np.ndarray, list[dict]]:
        detections = self.detect(frame_bgr, detector_name)
        if draw:
            frame_out = self.draw(frame_bgr, detections, show_bars=show_bars)
        else:
            frame_out = frame_bgr
        return frame_out, detections


# Globalna instancja - laduje sie raz przy starcie aplikacji
_engine: EmotionEngine | None = None


def get_engine() -> EmotionEngine:
    global _engine
    if _engine is None:
        _engine = EmotionEngine()
    return _engine
