#!/usr/bin/env python3
"""
Real-Time Object Recognition Web Application

Optimized for true realtime camera and video detection with minimal latency.
No external environment variables required beyond HOST, PORT, FLASK_ENV, Python version.

Author: Aravind Itte
Repository: https://github.com/aravinditte/realtime-object-recognition
Technology Stack: Flask, SocketIO, OpenCV, Ultralytics YOLO
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded configuration for optimal realtime performance
CONFIDENCE_THRESHOLD = 0.30
IOU_THRESHOLD = 0.45
FRAME_QUEUE_MAX = 3
INFER_WIDTH = 640
INFER_HEIGHT = 384
EMIT_FPS_CAP = 10  # Max 10 FPS emission to client
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'realtime-object-recognition-2024'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Initialize SocketIO for realtime WebSocket communication
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='eventlet',
    logger=False, 
    engineio_logger=False, 
    ping_timeout=30, 
    ping_interval=25,
    allow_upgrades=False,
    cookie=False
)

# Global variables
model = None
model_loaded = False
processing_active = {}
frame_queues = {}
frame_counts = {}

# Allowed file extensions
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}

def allowed_file(filename, file_type):
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return (file_type == 'video' and ext in ALLOWED_VIDEO_EXTENSIONS) or \
           (file_type == 'image' and ext in ALLOWED_IMAGE_EXTENSIONS)

def load_model():
    """Load YOLO model for realtime object recognition"""
    global model, model_loaded
    try:
        logger.info("Loading YOLO model for realtime recognition...")
        model = YOLO('yolov8n.pt')  # Fastest model for realtime performance
        model_loaded = True
        logger.info("YOLO model loaded successfully for realtime recognition!")
    except Exception as e:
        logger.exception(f"Error loading YOLO model: {e}")
        model_loaded = False

def preprocess_frame(frame):
    """Resize frame to inference size for consistent realtime processing"""
    return cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))

def recognize_objects_realtime(frame):
    """Perform realtime object recognition with optimized scaling"""
    if not model_loaded or model is None:
        return frame, []
    
    try:
        # Get original dimensions
        orig_h, orig_w = frame.shape[:2]
        
        # Preprocess for inference
        resized_frame = preprocess_frame(frame)
        
        # Run YOLO inference
        results = model(resized_frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        
        recognitions = []
        annotated_frame = frame.copy()
        
        # Calculate scale factors
        scale_x = orig_w / INFER_WIDTH
        scale_y = orig_h / INFER_HEIGHT
        
        # Process results
        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                # Get detection data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names.get(class_id, str(class_id))
                
                # Scale back to original frame size
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Draw bounding box with optimized thickness
                color = (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                label = f"{class_name} {confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, max(0, y1 - th - 6)), 
                            (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1 + 3, y1 - 4), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Store recognition
                recognitions.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
        
        return annotated_frame, recognitions
    
    except Exception as e:
        logger.exception(f"Realtime recognition error: {e}")
        return frame, []

@app.route('/')
def index():
    """Main realtime object recognition page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check for realtime recognition service"""
    return jsonify({
        'status': 'healthy',
        'service': 'Real-Time Object Recognition',
        'model_loaded': model_loaded,
        'realtime_config': {
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'iou_threshold': IOU_THRESHOLD,
            'inference_size': f"{INFER_WIDTH}x{INFER_HEIGHT}",
            'canvas_size': f"{CANVAS_WIDTH}x{CANVAS_HEIGHT}",
            'max_fps': EMIT_FPS_CAP
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for realtime processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    is_video = allowed_file(file.filename, 'video')
    is_image = allowed_file(file.filename, 'image')
    
    if not (is_video or is_image):
        return jsonify({'error': 'Unsupported file format'}), 400
    
    try:
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        os.makedirs('uploads', exist_ok=True)
        filepath = os.path.join('uploads', unique_filename)
        file.save(filepath)
        
        if is_image:
            frame = cv2.imread(filepath)
            if frame is None:
                return jsonify({'error': 'Invalid image file'}), 400
            
            annotated_frame, recognitions = recognize_objects_realtime(frame)
            
            # Encode result
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Cleanup
            try:
                os.remove(filepath)
            except Exception:
                pass
            
            return jsonify({
                'type': 'image',
                'image': img_base64,
                'detections': recognitions,
                'count': len(recognitions)
            })
        
        else:  # video
            return jsonify({
                'type': 'video',
                'filepath': filepath,
                'message': 'Starting realtime video processing...'
            })
    
    except Exception as e:
        logger.exception(f"Upload processing error: {e}")
        return jsonify({'error': 'Failed to process file'}), 500

@socketio.on('connect')
def on_connect():
    """Handle realtime client connection"""
    sid = request.sid
    processing_active[sid] = False
    frame_queues[sid] = LightQueue(maxsize=FRAME_QUEUE_MAX)
    frame_counts[sid] = 0
    
    logger.info(f"Realtime client {sid} connected")
    emit('connection_status', {
        'status': 'connected',
        'service': 'Real-Time Object Recognition',
        'model_loaded': model_loaded,
        'client_id': sid,
        'realtime_ready': True
    })

@socketio.on('disconnect')
def on_disconnect():
    """Handle client disconnection"""
    sid = request.sid
    processing_active.pop(sid, None)
    frame_queues.pop(sid, None)
    frame_counts.pop(sid, None)
    logger.info(f"Realtime client {sid} disconnected")

@socketio.on('webcam_frame')
def on_webcam_frame(data):
    """Handle realtime webcam frame with optimized queueing"""
    sid = request.sid
    queue = frame_queues.get(sid)
    
    if not queue:
        return
    
    try:
        # Drop oldest frame if queue full (maintains realtime flow)
        while queue.qsize() >= FRAME_QUEUE_MAX:
            try:
                queue.get_nowait()
            except Empty:
                break
        
        # Enqueue new frame
        queue.put_nowait(data['frame'])
        
        # Send periodic queue status
        frame_counts[sid] += 1
        if frame_counts[sid] % 20 == 0:
            emit('queue_status', {'frames_received': frame_counts[sid], 'queue_size': queue.qsize()})
    
    except Exception:
        pass
    
    # Start processing worker
    if not processing_active.get(sid):
        processing_active[sid] = True
        eventlet.spawn_n(realtime_frame_worker, sid)

def realtime_frame_worker(sid):
    """Realtime frame processing worker with optimized throughput"""
    queue = frame_queues.get(sid)
    if not queue:
        return
    
    last_emit = 0.0
    emit_interval = 1.0 / EMIT_FPS_CAP  # Cap at 10 FPS emission
    processed_count = 0
    
    while processing_active.get(sid) and queue:
        try:
            frame_data = queue.get(timeout=1.0)
        except Empty:
            continue
        
        try:
            # Decode frame safely
            if ',' in frame_data:
                image_data = frame_data.split(',')[1]
            else:
                image_data = frame_data
                
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            # Realtime object recognition
            annotated_frame, recognitions = recognize_objects_realtime(frame)
            
            # Encode with optimized quality
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            processed_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Emit with rate limiting
            current_time = time.time()
            if current_time - last_emit >= emit_interval:
                socketio.emit('processed_frame', {
                    'frame': f"data:image/jpeg;base64,{processed_base64}",
                    'detections': recognitions,
                    'timestamp': current_time,
                    'realtime': True
                }, room=sid)
                last_emit = current_time
                
                processed_count += 1
                if processed_count % 50 == 0:
                    logger.info(f"Realtime processed {processed_count} frames for {sid}")
                
        except Exception as e:
            logger.exception(f"Realtime frame worker error: {e}")
            socketio.emit('error', {'message': 'Realtime processing error', 'recoverable': True}, room=sid)

@socketio.on('process_video')
def on_process_video(data):
    """Handle realtime video processing"""
    sid = request.sid
    filepath = data.get('filepath')
    
    if not filepath or not os.path.exists(filepath):
        emit('error', {'message': 'Video file not found'})
        return
    
    processing_active[sid] = True
    eventlet.spawn_n(realtime_video_worker, sid, filepath)

def realtime_video_worker(sid, filepath):
    """Realtime video processing worker"""
    cap = cv2.VideoCapture(filepath)
    
    if not cap.isOpened():
        socketio.emit('error', {'message': 'Cannot open video file'}, room=sid)
        processing_active[sid] = False
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_delay = max(0.033, 1.0 / fps)  # Cap at ~30 FPS for realtime feel
    frame_count = 0
    
    logger.info(f"Starting realtime video processing: {total_frames} frames at {fps} FPS")
    
    try:
        while processing_active.get(sid) and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame with realtime recognition
            annotated_frame, recognitions = recognize_objects_realtime(frame)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Emit realtime video frame
            socketio.emit('video_frame', {
                'frame': f"data:image/jpeg;base64,{frame_base64}",
                'detections': recognitions,
                'progress': (frame_count / total_frames * 100) if total_frames > 0 else 0,
                'frame_number': frame_count,
                'total_frames': total_frames,
                'realtime': True
            }, room=sid)
            
            # Control realtime playback speed
            eventlet.sleep(frame_delay)
            
    finally:
        cap.release()
        try:
            os.remove(filepath)
        except Exception:
            pass
        
        socketio.emit('video_complete', {'message': 'Realtime video processing completed'}, room=sid)
        logger.info(f"Realtime video processing completed for {sid}: {frame_count} frames")

@socketio.on('stop_processing')
def on_stop_processing():
    """Stop realtime processing"""
    sid = request.sid
    processing_active[sid] = False
    emit('processing_stopped', {'status': 'stopped', 'realtime': True})

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Get port and host
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting Real-Time Object Recognition Server on {host}:{port}")
    logger.info(f"Model loaded: {model_loaded}")
    logger.info(f"Realtime config: conf={CONFIDENCE_THRESHOLD}, iou={IOU_THRESHOLD}, "f"inference={INFER_WIDTH}x{INFER_HEIGHT}, canvas={CANVAS_WIDTH}x{CANVAS_HEIGHT}")
    
    # Start realtime recognition server
    socketio.run(app, host=host, port=port, debug=False)