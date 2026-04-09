import os
import json
import cv2
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

def process_image(img_path, model, thresholds, output_dir):
    print(f"\n--- Przetwarzanie zdjęcia: {img_path.name} ---")
    
    for conf in thresholds:
        print(f"Próg confidence: {conf}")
        json_filename = output_dir / f"zdjecie_conf_{conf}.json"
        img_filename = output_dir / f"zdjecie_conf_{conf}.jpg"

        if json_filename.exists() and img_filename.exists():
            print(f"Pliki już istnieją, pomijam: {json_filename.name}, {img_filename.name}")
            continue

        # Detekcja (b, c)
        results = model(img_path, conf=conf)
        result = results[0]
        
        json_data = []
        for box in result.boxes:
            json_data.append({
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox_xyxy": box.xyxy[0].tolist()
            })
            
        # Zapis do JSON
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)
            
        # Zapis zdjęcia z wyrysowanymi bounding boxami
        result.save(filename=str(img_filename))
        print(f"Zapisano: {json_filename.name} oraz {img_filename.name}")

def process_video(video_path, model, thresholds, output_dir):
    """Przetwarza wideo klatka po klatce dla różnych progów confidence."""
    print(f"\n--- Przetwarzanie wideo: {video_path.name} ---")
    
    for conf in thresholds:
        print(f"Próg confidence: {conf}")
        video_filename = output_dir / f"wideo_conf_{conf}.mp4"
        json_filename = output_dir / f"wideo_conf_{conf}.json"
        stats_filename = output_dir / f"wideo_statystyki_conf_{conf}.txt"

        if video_filename.exists() and json_filename.exists() and stats_filename.exists():
            print(f"Pliki już istnieją, pomijam: {video_filename.name}, {json_filename.name}, {stats_filename.name}")
            continue

        cap = cv2.VideoCapture(str(video_path))
        
        # Przygotowanie do zapisu wideo (e)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_filename), fourcc, fps, (width, height))
        
        video_json_data = {}
        class_stats = defaultdict(int) # Słownik do zliczania wystąpień klas (e)
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detekcja na pojedynczej klatce
            results = model(frame, conf=conf, verbose=False)
            result = results[0]
            
            frame_detections = []
            for box in result.boxes:
                class_name = model.names[int(box.cls[0])]
                frame_detections.append({
                    "class_id": int(box.cls[0]),
                    "class_name": class_name,
                    "confidence": float(box.conf[0]),
                    "bbox_xyxy": box.xyxy[0].tolist()
                })
                # Aktualizacja statystyk dla całego wideo (e)
                class_stats[class_name] += 1
                
            video_json_data[f"frame_{frame_idx}"] = frame_detections
            
            # Zapis klatki z wyrysowanymi obiektami do pliku wyjściowego wideo
            annotated_frame = result.plot()
            out.write(annotated_frame)
            
            frame_idx += 1
            
        cap.release()
        out.release()
        
        # Zapis json dla wideo (d)
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(video_json_data, f, indent=4)
            
        # Zapis statystyk (e)
        with open(stats_filename, 'w', encoding='utf-8') as f:
            f.write(f"Statystyki detekcji dla wideo (próg {conf}):\n")
            for cls_name, count in class_stats.items():
                f.write(f"Klasa '{cls_name}': {count} detekcji w sumie na wszystkich klatkach\n")
                
        print(f"Zapisano: {video_filename.name}, {json_filename.name}, {stats_filename.name}")

if __name__ == "__main__":
    model = YOLO('yolov8n.pt')
    
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "wyniki_zad_1"
    output_dir.mkdir(exist_ok=True)
    
    IMAGE_NAME = "office_yolo.png"
    VIDEO_NAME = "office_yolo.mp4"
    
    img_path = base_dir / IMAGE_NAME
    video_path = base_dir / VIDEO_NAME
    
    thresholds = [0.1, 0.3, 0.5, 0.7]
    
    if img_path.exists():
        process_image(img_path, model, thresholds, output_dir)
    else:
        print(f"Brak pliku zdjęcia: {img_path}")
        
    if video_path.exists():
        process_video(video_path, model, thresholds, output_dir)
    else:
        print(f"Brak pliku wideo: {video_path}")