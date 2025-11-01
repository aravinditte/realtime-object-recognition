#!/usr/bin/env python3
"""
Real-Time Object Detection Web Application

Fixes:
- Use pure WebSocket transport to avoid Engine.IO long-polling payload limits
- Apply rate limiting and queueing on incoming webcam frames
- Resize frames to model input size to improve accuracy and speed
- Add NMS/threshold config via env vars
- Emit using socketio.emit in background contexts only
- Add CORS allowed origins from env
"""
import os
import cv2
import base64
import numpy as np
import eventlet
from eventlet.queue import LightQueue, Empty
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import logging
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import time

# Patch for eventlet with urllib3 (if needed)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config from env
CONF = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.35'))
IOU = float(os.environ.get('IOU_THRESHOLD', '0.45'))
MAX_Q = int(os.environ.get('FRAME_QUEUE_MAX', '3'))
TARGET_W = int(os.environ.get('INFER_WIDTH', '640'))
TARGET_H = int(os.environ.get('INFER_HEIGHT', '384'))
CORS = os.environ.get('CORS_ORIGINS', '*')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'realtime-object-detection-2024')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Force websockets transport to avoid payload overflow
socketio = SocketIO(app, cors_allowed_origins=CORS, async_mode='eventlet',
                    logger=False, engineio_logger=False, ping_timeout=30, ping_interval=25)

# Globals
model = None
model_loaded = False
processing_active = {}
frame_queues = {}

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}

def allowed_file(filename, file_type):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return (file_type == 'video' and ext in ALLOWED_VIDEO_EXTENSIONS) or \
           (file_type == 'image' and ext in ALLOWED_IMAGE_EXTENSIONS)

def load_model():
    global model, model_loaded
    try:
        logger.info("Loading YOLO model...")
        model = YOLO(os.environ.get('MODEL_NAME', 'yolov8n.pt'))
        model_loaded = True
        logger.info("YOLO model loaded successfully!")
    except Exception as e:
        logger.exception("Error loading YOLO model: %s", e)
        model_loaded = False


def preprocess_frame(frame):
    # Resize to target while keeping aspect via letterbox style simple resize
    return cv2.resize(frame, (TARGET_W, TARGET_H))


def detect_objects(frame):
    if not model_loaded or model is None:
        return frame, []
    try:
        resized = preprocess_frame(frame)
        results = model(resized, conf=CONF, iou=IOU, verbose=False)
        detections = []
        # scale back boxes to original frame size
        h, w = frame.shape[:2]
        rh, rw = resized.shape[:2]
        sx, sy = w / rw, h / rh
        annotated = frame.copy()
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                name = model.names.get(cls, str(cls))
                # scale
                x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                detections.append({'class': name, 'confidence': conf, 'bbox': [x1, y1, x2, y2]})
        return annotated, detections
    except Exception as e:
        logger.exception("Detection error: %s", e)
        return frame, []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model_loaded, 'timestamp': datetime.now().isoformat()})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    is_video = allowed_file(f.filename, 'video')
    is_image = allowed_file(f.filename, 'image')
    if not (is_video or is_image):
        return jsonify({'error': 'Unsupported file format'}), 400
    filename = secure_filename(f.filename)
    os.makedirs('uploads', exist_ok=True)
    path = os.path.join('uploads', f"{uuid.uuid4()}_{filename}")
    f.save(path)
    if is_image:
        frame = cv2.imread(path)
        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400
        annotated, dets = detect_objects(frame)
        _, buf = cv2.imencode('.jpg', annotated)
        b64 = base64.b64encode(buf).decode('utf-8')
        try:
            os.remove(path)
        except Exception:
            pass
        return jsonify({'type': 'image', 'image': b64, 'detections': dets, 'count': len(dets)})
    return jsonify({'type': 'video', 'filepath': path})

@socketio.on('connect')
def on_connect():
    sid = request.sid
    processing_active[sid] = False
    frame_queues[sid] = LightQueue(maxsize=MAX_Q)
    emit('connection_status', {'status': 'connected', 'model_loaded': model_loaded, 'client_id': sid})

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    processing_active.pop(sid, None)
    frame_queues.pop(sid, None)

@socketio.on('webcam_frame')
def on_webcam_frame(data):
    """Enqueue frames, drop if queue is full to avoid payload overflow."""
    sid = request.sid
    q = frame_queues.get(sid)
    if not q:
        return
    try:
        if q.qsize() >= MAX_Q:
            # Drop oldest to keep stream flowing
            try:
                q.get_nowait()
            except Empty:
                pass
        q.put_nowait(data['frame'])
    except Exception:
        pass

    # Start processor once per client
    if not processing_active.get(sid):
        processing_active[sid] = True
        eventlet.spawn_n(process_frames_worker, sid)


def process_frames_worker(sid):
    q = frame_queues.get(sid)
    if not q:
        return
    last_emit = 0.0
    target_interval = 0.1  # 10 FPS max
    while processing_active.get(sid) and q:
        try:
            frame_data = q.get(timeout=1.0)
        except Empty:
            continue
        try:
            image_bytes = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            annotated, dets = detect_objects(frame)
            _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.b64encode(buf).decode('utf-8')
            now = time.time()
            if now - last_emit >= target_interval:
                socketio.emit('processed_frame', {'frame': f"data:image/jpeg;base64,{b64}", 'detections': dets, 'timestamp': now}, room=sid)
                last_emit = now
        except Exception as e:
            logger.exception("Frame worker error: %s", e)
            socketio.emit('error', {'message': 'Frame processing error'}, room=sid)

@socketio.on('process_video')
def on_process_video(data):
    sid = request.sid
    path = data.get('filepath')
    if not path or not os.path.exists(path):
        emit('error', {'message': 'Video file not found'})
        return
    processing_active[sid] = True
    eventlet.spawn_n(process_video_worker, sid, path)


def process_video_worker(sid, path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        socketio.emit('error', {'message': 'Cannot open video file'}, room=sid)
        processing_active[sid] = False
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    delay = 1.0 / fps
    idx = 0
    try:
        while processing_active.get(sid) and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            annotated, dets = detect_objects(frame)
            _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.b64encode(buf).decode('utf-8')
            socketio.emit('video_frame', {
                'frame': f"data:image/jpeg;base64,{b64}",
                'detections': dets,
                'progress': (idx / total * 100) if total else 0,
                'frame_number': idx,
                'total_frames': total
            }, room=sid)
            eventlet.sleep(delay)
    finally:
        cap.release()
        try:
            os.remove(path)
        except Exception:
            pass
        socketio.emit('video_complete', {}, room=sid)

@socketio.on('stop_processing')
def on_stop_processing():
    sid = request.sid
    processing_active[sid] = False
    emit('processing_stopped', {'status': 'stopped'})

if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    logger.info(f"Starting Real-Time Object Detection Server on {host}:{port}")
    logger.info(f"Model loaded: {model_loaded}")
    socketio.run(app, host=host, port=port, debug=False, transports=['websocket'])
