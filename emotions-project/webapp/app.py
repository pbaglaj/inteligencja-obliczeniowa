"""
Flask backend dla detekcji emocji.

Endpointy:
  GET  /                          - strona glowna
  GET  /api/detectors             - lista dostepnych detektorow twarzy
  POST /api/detect/image          - wgranie zdjecia, zwrot przetworzonego JPEG
  POST /api/detect/image_compare  - wgranie zdjecia, zwrot 3 wersji (po jednej na detektor)
  POST /api/detect/frame          - pojedyncza klatka z kamery (base64), zwrot JSON z detekcjami
  POST /api/detect/video          - wgranie filmu, zwrot id zadania
  GET  /api/video/<job_id>/status - status przetwarzania filmu
  GET  /api/video/<job_id>/result - pobranie przetworzonego MP4
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file, abort

from emotion_engine import get_engine, AVAILABLE_DETECTORS, EMOTION_CLASSES

APP_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = APP_DIR / 'uploads'
RESULTS_DIR = APP_DIR / 'results'
UPLOADS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder=str(APP_DIR / 'templates'),
            static_folder=str(APP_DIR / 'static'))
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB

# Eager load aby pierwszy request nie czekal kilku sekund
engine = get_engine()

# In-memory rejestr zadan video: job_id -> {status, progress, result_path, error}
video_jobs: dict[str, dict] = {}
video_jobs_lock = threading.Lock()


def _decode_image_from_request(file_storage) -> np.ndarray:
    data = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Nie udalo sie zdekodowac obrazu")
    return img


def _decode_base64_image(b64: str) -> np.ndarray:
    # Akceptuj zarowno "data:image/jpeg;base64,...." jak i sam base64
    if ',' in b64:
        b64 = b64.split(',', 1)[1]
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Nie udalo sie zdekodowac obrazu base64")
    return img


def _encode_image_jpeg_b64(img_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Nie udalo sie zakodowac JPEG")
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')


# ---------- Routes ----------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/detectors')
def list_detectors():
    return jsonify({
        'detectors': engine.list_detectors(),
        'emotions': EMOTION_CLASSES,
    })


@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        return jsonify({'error': 'Brak pliku w polu "file"'}), 400
    detector_name = request.form.get('detector', 'blaze_short')
    if detector_name not in AVAILABLE_DETECTORS:
        return jsonify({'error': f'Nieznany detektor: {detector_name}'}), 400

    try:
        img = _decode_image_from_request(request.files['file'])
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    t0 = time.time()
    annotated, detections = engine.process_frame(img, detector_name, draw=True)
    elapsed_ms = (time.time() - t0) * 1000

    return jsonify({
        'image': _encode_image_jpeg_b64(annotated),
        'detections': detections,
        'detector': detector_name,
        'elapsed_ms': round(elapsed_ms, 1),
        'faces': len(detections),
    })


@app.route('/api/detect/image_compare', methods=['POST'])
def detect_image_compare():
    if 'file' not in request.files:
        return jsonify({'error': 'Brak pliku w polu "file"'}), 400
    try:
        img = _decode_image_from_request(request.files['file'])
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    results = []
    for det_id, meta in AVAILABLE_DETECTORS.items():
        t0 = time.time()
        annotated, detections = engine.process_frame(img, det_id, draw=True, show_bars=False)
        elapsed_ms = (time.time() - t0) * 1000
        results.append({
            'detector': det_id,
            'label': meta['label'],
            'image': _encode_image_jpeg_b64(annotated),
            'faces': len(detections),
            'detections': detections,
            'elapsed_ms': round(elapsed_ms, 1),
        })
    return jsonify({'results': results})


@app.route('/api/detect/frame', methods=['POST'])
def detect_frame():
    """Klatka z kamery jako base64 JPEG. Tryb single lub compare."""
    data = request.get_json(silent=True) or {}
    b64 = data.get('image')
    if not b64:
        return jsonify({'error': 'Brak pola "image"'}), 400
    mode = data.get('mode', 'single')  # 'single' lub 'compare'
    detector_name = data.get('detector', 'blaze_short')

    try:
        img = _decode_base64_image(b64)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    if mode == 'compare':
        results = []
        for det_id, meta in AVAILABLE_DETECTORS.items():
            t0 = time.time()
            annotated, detections = engine.process_frame(img, det_id, draw=True, show_bars=False)
            elapsed_ms = (time.time() - t0) * 1000
            results.append({
                'detector': det_id,
                'label': meta['label'],
                'image': _encode_image_jpeg_b64(annotated, quality=70),
                'faces': len(detections),
                'elapsed_ms': round(elapsed_ms, 1),
            })
        return jsonify({'mode': 'compare', 'results': results})

    if detector_name not in AVAILABLE_DETECTORS:
        return jsonify({'error': f'Nieznany detektor: {detector_name}'}), 400
    t0 = time.time()
    annotated, detections = engine.process_frame(img, detector_name, draw=True)
    elapsed_ms = (time.time() - t0) * 1000
    return jsonify({
        'mode': 'single',
        'image': _encode_image_jpeg_b64(annotated, quality=75),
        'detections': detections,
        'detector': detector_name,
        'elapsed_ms': round(elapsed_ms, 1),
        'faces': len(detections),
    })


# ---------- Video pipeline ----------

def _process_video_job(job_id: str, input_path: Path, output_path: Path,
                       detector_name: str):
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError("Nie mozna otworzyc pliku wideo")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # mp4v - dziala out-of-the-box bez ffmpeg w wiekszosci buildow opencv-python
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Nie mozna utworzyc pliku wyjsciowego MP4")

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, _ = engine.process_frame(frame, detector_name, draw=True, show_bars=False)
            writer.write(annotated)
            idx += 1
            if total > 0:
                with video_jobs_lock:
                    video_jobs[job_id]['progress'] = round(100 * idx / total, 1)
                    video_jobs[job_id]['frames_done'] = idx
                    video_jobs[job_id]['frames_total'] = total

        cap.release()
        writer.release()

        with video_jobs_lock:
            video_jobs[job_id]['status'] = 'done'
            video_jobs[job_id]['progress'] = 100.0
            video_jobs[job_id]['result_path'] = str(output_path)
    except Exception as e:
        with video_jobs_lock:
            video_jobs[job_id]['status'] = 'error'
            video_jobs[job_id]['error'] = str(e)
    finally:
        # Sprzatamy plik wejsciowy
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.route('/api/detect/video', methods=['POST'])
def detect_video():
    if 'file' not in request.files:
        return jsonify({'error': 'Brak pliku w polu "file"'}), 400
    detector_name = request.form.get('detector', 'blaze_short')
    if detector_name not in AVAILABLE_DETECTORS:
        return jsonify({'error': f'Nieznany detektor: {detector_name}'}), 400

    file = request.files['file']
    job_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename or '')[1].lower() or '.mp4'
    input_path = UPLOADS_DIR / f'{job_id}{ext}'
    output_path = RESULTS_DIR / f'{job_id}.mp4'
    file.save(str(input_path))

    with video_jobs_lock:
        video_jobs[job_id] = {
            'status': 'processing',
            'progress': 0.0,
            'detector': detector_name,
            'result_path': None,
            'error': None,
        }

    thread = threading.Thread(target=_process_video_job,
                              args=(job_id, input_path, output_path, detector_name),
                              daemon=True)
    thread.start()
    return jsonify({'job_id': job_id})


@app.route('/api/video/<job_id>/status')
def video_status(job_id: str):
    with video_jobs_lock:
        job = video_jobs.get(job_id)
        if job is None:
            return jsonify({'error': 'Nieznane zadanie'}), 404
        return jsonify({'job_id': job_id, **job})


@app.route('/api/video/<job_id>/result')
def video_result(job_id: str):
    with video_jobs_lock:
        job = video_jobs.get(job_id)
    if job is None:
        abort(404)
    if job['status'] != 'done' or not job.get('result_path'):
        abort(404)
    return send_file(job['result_path'], mimetype='video/mp4',
                     as_attachment=False, download_name=f'{job_id}.mp4')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
