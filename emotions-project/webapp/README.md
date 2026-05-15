# Detekcja emocji — aplikacja webowa

Lokalna aplikacja w przeglądarce nad istniejącym pipeline'em z [detect.py](../detect.py).
Trzy tryby:

- **Zdjęcie** — wgraj obraz, otrzymaj wersję z naniesionymi ramkami i emocjami.
- **Film** — wgraj plik wideo, backend przetwarza klatka po klatce i zwraca gotowy MP4.
- **Kamera (live)** — strumień z przeglądarki przez `getUserMedia`, klatki są wysyłane do backendu i wracają zanotowane.

W każdym trybie można wybrać jeden z trzech detektorów twarzy MediaPipe BlazeFace
(`short_range`, `full_range`, `full_range_sparse`) albo włączyć **tryb porównania** —
wtedy wszystkie trzy detektory działają równolegle na tym samym wejściu i wyniki
pokazywane są obok siebie.

## Instalacja

```powershell
cd webapp
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> Zakładam, że plik [models/emotion_model.pth](../models/emotion_model.pth) jest już
> wytrenowany (skryptem [train.py](../train.py)) — backend ładuje go przy starcie.

## Uruchomienie

```powershell
python app.py
```

Następnie otwórz w przeglądarce: <http://127.0.0.1:5000>

## Architektura

- [emotion_engine.py](emotion_engine.py) — wrapper na model `EmotionCNN` + MediaPipe.
  Ładuje model raz, cache'uje detektory, ma metodę `process_frame()`.
- [app.py](app.py) — Flask, endpointy JSON + statyczne pliki.
- [templates/index.html](templates/index.html), [static/app.js](static/app.js),
  [static/style.css](static/style.css) — frontend (vanilla JS).
- `uploads/`, `results/` — tymczasowe pliki przy przetwarzaniu wideo.

## Endpointy

| Endpoint | Opis |
|---|---|
| `GET /api/detectors` | Lista detektorów + nazwy emocji |
| `POST /api/detect/image` | `file=...`, `detector=...` → przetworzone JPEG |
| `POST /api/detect/image_compare` | `file=...` → wynik 3 detektorów |
| `POST /api/detect/frame` | JSON `{image: base64, mode: single\|compare, detector}` |
| `POST /api/detect/video` | `file=...`, `detector=...` → `{job_id}` |
| `GET /api/video/<id>/status` | Status (`processing`/`done`/`error`) + progres |
| `GET /api/video/<id>/result` | Pobranie gotowego MP4 |

Struktura:


emotions-project/
├── webapp/
│   ├── app.py                  # Flask backend
│   ├── emotion_engine.py       # Reusable wrapper (CNN + 3 detektory MediaPipe)
│   ├── requirements.txt
│   ├── README.md               # Instrukcja uruchomienia
│   ├── templates/index.html
│   ├── static/
│   │   ├── style.css
│   │   └── app.js
│   ├── uploads/                # tymczasowe wideo wejściowe
│   └── results/                # przetworzone MP4
3 tryby dostępne w przeglądarce:

Zdjęcie — wgrywasz JPG/PNG → dostajesz przetworzony obraz z ramkami i paskami emocji
Film — wgrywasz mp4 → backend tworzy nowy MP4 (progress bar w trakcie), zwraca odtwarzacz i link do pobrania
Kamera (live) — getUserMedia z przeglądarki, klatki lecą przez /api/detect/frame, wyświetlają się z FPS i RTT
Selektor detektora + checkbox „Tryb porównania" — działa w trybie zdjęcia ORAZ kamery (3 modele jednocześnie side-by-side, jak ustaliliśmy).

Co potwierdziłem testami:

emotion_engine.py:131 ładuje model z models/emotion_model.pth i wszystkie 3 detektory z detectors/
GET /api/detectors zwraca listę
POST /api/detect/image — Happy 99.99% na FER2013 test/happy
POST /api/detect/image_compare — 3 wyniki obok siebie (short_range wykrył twarz 48×48, full_range nie — sensowne, bo full_range jest tunowany na większe twarze)
POST /api/detect/frame — base64 round-trip ~5 ms
Jak odpalić

"c:\Users\Admin\github\inteligencja-obliczeniowa\venv\Scripts\python.exe" webapp\app.py
Następnie http://127.0.0.1:5000. Flask już dodany do twojego istniejącego venv.