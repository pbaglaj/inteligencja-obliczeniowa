import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from model import EmotionCNN

# --- INICJALIZACJA MODELU EMOCJI ---
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
device = torch.device("cpu")

model = EmotionCNN(num_classes=len(emotion_classes))
model.load_state_dict(torch.load('models/emotion_model.pth', map_location=device))
model.to(device)
model.eval() 

transform = transforms.Compose([
    transforms.Resize((48, 48)),         
    transforms.ToTensor(),               
    transforms.Normalize((0.5,), (0.5,)) 
])

# --- INICJALIZACJA NOWEGO MEDIAPIPE TASKS API ---
base_options = python.BaseOptions(model_asset_path='assets/detectors/blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.6)
detector = vision.FaceDetector.create_from_options(options)

# --- USTAWIENIA KAMERY I STABILIZACJI ---
cap = cv2.VideoCapture(0)
print("Kamera uruchomiona! Wciśnij klawisz 'q' na klawiaturze, aby wyłączyć.")

frame_skip = 1
frame_count = 0

emotion_buffer = deque(maxlen=5)
prob_buffer = deque(maxlen=5)
last_face_rect = None     
frames_without_face = 0   

# --- ZMIENNE DLA STABILNEGO FPS ---
prev_frame_time = time.time()
fps_buffer = deque(maxlen=15)  # Przechowuje 15 ostatnich pomiarów do uśrednienia
TARGET_FPS = 60                # Nasz limit klatek
frame_time_target = 1.0 / TARGET_FPS

emotion_colors = {
    'Angry': (0, 0, 255),      # Czerwony
    'Disgust': (0, 100, 0),    # Ciemnozielony
    'Fear': (128, 0, 128),     # Fioletowy
    'Happy': (0, 255, 255),    # Żółty
    'Neutral': (200, 200, 200),# Szary
    'Sad': (255, 0, 0),        # Niebieski
    'Surprise': (0, 165, 255)  # Pomarańczowy
}

# --- GŁÓWNA PĘTLA PROGRAMU ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    
    if frame_count % frame_skip == 0:
        # Konwersja obrazu dla nowego API
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Wykrywanie twarzy
        detection_result = detector.detect(mp_image)
        
        if detection_result.detections:
            # Sortujemy wykryte twarze po wielkości (wybieramy największą)
            def get_area(det):
                return det.bounding_box.width * det.bounding_box.height
            
            largest_detection = max(detection_result.detections, key=get_area)
            bbox = largest_detection.bounding_box
            
            # Nowe API od razu podaje nam piksele
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            
            # --- STABILIZACJA KWADRATU ---
            if last_face_rect is None:
                last_face_rect = [x, y, w, h]
            else:
                last_face_rect[0] = int(0.7 * x + 0.3 * last_face_rect[0])
                last_face_rect[1] = int(0.7 * y + 0.3 * last_face_rect[1])
                last_face_rect[2] = int(0.7 * w + 0.3 * last_face_rect[2])
                last_face_rect[3] = int(0.7 * h + 0.3 * last_face_rect[3])
                
            frames_without_face = 0 
            
            # --- PRZYGOTOWANIE OBRAZU DLA SIECI CNN ---
            sx, sy, sw, sh = last_face_rect
            sy, sx = max(0, sy), max(0, sx) 
            
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = gray_full[sy:sy+sh, sx:sx+sw]
            
            if roi_gray.size > 0:
                roi_pil = Image.fromarray(roi_gray)
                input_tensor = transform(roi_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted = torch.max(output.data, 1)
                    emotion = emotion_classes[predicted.item()]
                    emotion_buffer.append(emotion)

                    # Softmax
                    # output ma kształt [1, 7], więc bierzemy [0] i mnożymy przez 100
                    probabilities = F.softmax(output, dim=1)[0] * 100
                    
                    # Zapisujemy do bufora (jako listę liczb), aby paski nie skakały
                    prob_buffer.append(probabilities.cpu().numpy())
        else:
            frames_without_face += 1
            if frames_without_face > 10:
                last_face_rect = None
                emotion_buffer.clear()
                prob_buffer.clear()

    # --- RYSOWANIE W KAŻDEJ KLATCE ---
    if last_face_rect is not None:
        rx, ry, rw, rh = last_face_rect
        
        # Domyślny kolor (niebieski), gdyby bufor był jeszcze pusty (pierwsze ułamki sekund)
        box_color = (255, 0, 0)
        
        if len(emotion_buffer) > 0:
            # 1. Najpierw wyciągamy stabilną emocję
            smooth_emotion = max(set(emotion_buffer), key=emotion_buffer.count)
            
            # 2. Pobieramy dynamiczny kolor ze słownika
            box_color = emotion_colors[smooth_emotion]
            
            # 3. Rysujemy tekst (ustawiłem jego kolor na taki sam jak ramki - wygląda to dużo lepiej!)
            cv2.putText(frame, smooth_emotion, (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
            
        # 4. Rysujemy ramkę z przypisanym, dynamicznym kolorem
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), box_color, 2)
        
        if len(prob_buffer) > 0:
            # Uśredniamy procenty z 5 ostatnich klatek
            avg_probs = np.mean(prob_buffer, axis=0)
            
            y_offset = 30 # Początkowa wysokość na ekranie
            bar_max_width = 150 # Maksymalna długość paska (100%)
            
            # Ciemne półprzezroczyste tło pod interfejsem (opcjonalnie, poprawia czytelność)
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (170, 310), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            for i, emotion_name in enumerate(emotion_classes):
                prob = avg_probs[i]
                color = emotion_colors[emotion_name]
                
                # Tekst z nazwą emocji i wartością procentową
                text = f"{emotion_name}: {prob:.1f}%"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                
                # Tło paska (ciemnoszary)
                cv2.rectangle(frame, (10, y_offset + 8), (10 + bar_max_width, y_offset + 18), (50, 50, 50), -1)
                
                # Wypełnienie paska (kolorowe)
                bar_width = int((prob / 100) * bar_max_width)
                cv2.rectangle(frame, (10, y_offset + 8), (10 + bar_width, y_offset + 18), color, -1)
                
                y_offset += 40 # Przesunięcie w dół dla kolejnego paska
        
    # Ogranicznik (Cap) i stabilny FPS ---
    process_time = time.time() - prev_frame_time
    
    # Jeśli klatka wykonała się za szybko (np. zignorowaliśmy AI), 
    # usypiamy program na brakujący ułamek sekundy, aby utrzymać stałe 60 FPS.
    if process_time < frame_time_target:
        time.sleep(frame_time_target - process_time)
        
    # Pobieramy czas po ewentualnym uśpieniu
    new_frame_time = time.time()
    instant_fps = 1.0 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    
    # Dodajemy aktualny FPS do bufora i wyciągamy średnią (zapobiega migotaniu)
    fps_buffer.append(instant_fps)
    smooth_fps = sum(fps_buffer) / len(fps_buffer)
    
    # Rysowanie na ekranie
    frame_width = frame.shape[1] 
    fps_text = f"FPS: {int(smooth_fps)}"
    
    cv2.putText(frame, fps_text, (frame_width - 140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(frame, fps_text, (frame_width - 140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Detekcja Emocji AI', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()