import cv2
import numpy as np
from pathlib import Path

def count_birds_in_image(img_path, output_dir):
    """
    Przetwarza obraz tradycyjnymi metodami CV: 
    Grayscale -> Blur -> Threshold -> Connected Components.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    # Konwersja na skalę szarości (wymagane do progowania)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Usuwanie szumu (rozmycie medianowe pomaga pozbyć się małych kropek/zakłóceń)
    blurred = cv2.medianBlur(gray, 5)

    # Progowanie (Thresholding)
    # THRESH_BINARY_INV sprawi, że ptaki będą białe (obiekt), a tło czarne dla algorytmu.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Operacje morfologiczne (opcjonalnie - łączenie dziur wewnątrz ptaków)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Liczenie czarnych plam (obiektów)
    # Wynik obejmuje tło jako etykietę 0, więc odejmujemy 1.
    num_labels, labels = cv2.connectedComponents(thresh)
    bird_count = num_labels - 1

    debug_path = output_dir / f"processed_{img_path.name}"
    cv2.imwrite(str(debug_path), thresh)

    return bird_count

def main():
    base_dir = Path(__file__).resolve().parent
    input_folder = base_dir / "bird_miniatures"
    output_folder = base_dir / "wyniki_detekcji"
    output_folder.mkdir(exist_ok=True)

    if not input_folder.exists():
        print(f"Błąd: Folder {input_folder} nie istnieje!")
        return

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    print(f"{'Nazwa obrazu':<30} | {'Liczba ptaków':<15}")
    print("-" * 50)

    results = []
    
    for img_path in input_folder.iterdir():
        if img_path.suffix.lower() in extensions:
            count = count_birds_in_image(img_path, output_folder)
            
            if count is not None:
                print(f"{img_path.name:<30} | {count:<15}")
                results.append((img_path.name, count))

    with open(output_folder / "raport_ptaki.txt", "w", encoding="utf-8") as f:
        f.write("Raport z liczenia ptaków\n")
        f.write("=" * 30 + "\n")
        for name, count in results:
            f.write(f"{name}: {count}\n")

if __name__ == "__main__":
    main()