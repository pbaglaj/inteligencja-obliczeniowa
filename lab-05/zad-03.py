import cv2
from pathlib import Path
from ultralytics import YOLO

def count_flying_objects_yolo(input_folder, output_folder, model, conf_threshold=0.05):
    print(f"\n--- Analiza YOLO (Próg pewności: {conf_threshold}) ---")
    print(f"{'Nazwa obrazu':<30} | {'Wykryto (Ptak/Samolot/Latawiec)':<10}")
    print("-" * 65)

    # Klasy COCO: 14 (bird), 4 (airplane), 33 (kite)
    target_classes = [4, 14, 33] 
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    results_list = []

    for img_path in input_folder.iterdir():
        if img_path.suffix.lower() in extensions:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # preprocessing: Zwiększenie kontrastu
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

            # Detekcja
            results = model(enhanced_img, conf=conf_threshold, classes=target_classes, imgsz=640, verbose=False)
            
            detections = len(results[0].boxes)
            
            print(f"{img_path.name:<30} | {detections:<10}")
            results_list.append((img_path.name, detections))

            debug_img = results[0].plot()
            cv2.imwrite(str(output_folder / f"yolo_{img_path.name}"), debug_img)

    return results_list

if __name__ == "__main__":
    model = YOLO('yolov8n.pt')
    
    base_dir = Path(__file__).resolve().parent
    input_folder = base_dir / "bird_miniatures"
    output_folder = base_dir / "wyniki_yolo"
    output_folder.mkdir(exist_ok=True)

    if input_folder.exists():
        count_flying_objects_yolo(input_folder, output_folder, model, conf_threshold=0.01)
    else:
        print(f"Brak folderu: {input_folder}")