# Real-Time Object Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com)
[![YOLO](https://img.shields.io/badge/YOLO-v8-orange.svg)](https://ultralytics.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://github.com/aravinditte/realtime-object-recognition/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/aravinditte/realtime-object-recognition/actions)

A comprehensive, production-ready real-time object detection system that processes live camera feeds, uploaded videos, and images using state-of-the-art YOLO models. Built with modern web technologies and optimized for both development and production environments.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [MVP Implementation](#mvp-implementation)
- [Web Application](#web-application)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Performance](#performance)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Functionality
- **Real-Time Detection**: Live webcam processing with WebSocket streaming
- **Video Analysis**: Upload and process video files with object tracking
- **Image Processing**: Single image analysis with bounding box annotations
- **Multi-Model Support**: YOLOv8 variants (nano to extra-large)
- **80+ Object Classes**: Comprehensive COCO dataset object detection

### Technical Features
- **High Performance**: Optimized for low-latency real-time processing
- **Scalable Architecture**: Modular design with containerization support
- **Modern UI**: Responsive Bootstrap 5 interface with real-time updates
- **Production Ready**: Docker deployment with CI/CD pipeline
- **Security First**: Non-root containers, input validation, rate limiting
- **Monitoring**: Health checks, metrics, and comprehensive logging

### Deployment Options
- **Local Development**: Simple Python script execution
- **Docker**: Containerized deployment with docker-compose
- **Cloud Ready**: Support for AWS, GCP, Azure deployments
- **CI/CD Pipeline**: Automated testing and deployment

## Quick Start

### Prerequisites

- Python 3.8+ with pip
- Webcam or camera device
- 2GB+ RAM (4GB+ recommended)
- OpenCV-compatible system libraries

### 1. MVP Demo (5-minute setup)

The fastest way to see object detection in action:

```bash
# Clone repository
git clone https://github.com/aravinditte/realtime-object-recognition.git
cd realtime-object-recognition

# Setup and run MVP
python setup.py
python detect.py
```

Press 'q' to quit the MVP demo.

### 2. Full Web Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web application
python app.py
# or
python run.py
```

Open your browser to `http://localhost:5000`

### 3. Docker Deployment

```bash
# Quick start with Docker Compose
docker-compose up

# Or build and run manually
docker build -t realtime-object-detection .
docker run -p 5000:5000 realtime-object-detection
```

## Architecture

### System Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   AI Engine    │
│                 │    │                 │    │                 │
│ • Bootstrap 5   │◄──►│ • Flask 3.0     │◄──►│ • YOLOv8       │
│ • Socket.IO     │    │ • SocketIO      │    │ • OpenCV       │
│ • Real-time UI  │    │ • WebSockets    │    │ • Ultralytics   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Interface│    │  File Handling  │    │  Model Cache    │
│                 │    │                 │    │                 │
│ • Camera access │    │ • Upload/Process│    │ • Model storage │
│ • File uploads  │    │ • Validation    │    │ • Optimization  │
│ • Statistics    │    │ • Security      │    │ • GPU support   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack

**Backend**
- **Framework**: Flask 3.0 with SocketIO for real-time communication
- **AI/ML**: Ultralytics YOLO v8, OpenCV 4.8, PyTorch
- **Async**: Eventlet for WebSocket handling
- **Security**: Input validation, CORS protection, rate limiting

**Frontend**
- **UI Framework**: Bootstrap 5.3 with custom CSS
- **Real-time**: Socket.IO client for WebSocket communication
- **Icons**: Font Awesome 6.4
- **Animations**: Animate.css for smooth transitions

**Infrastructure**
- **Containerization**: Docker with multi-stage builds
- **CI/CD**: GitHub Actions with automated testing and deployment
- **Monitoring**: Health checks, metrics collection, logging
- **Security**: Non-root containers, security scanning

## MVP Implementation

The MVP (`detect.py`) demonstrates the core concept with minimal dependencies:

**What it does:**
1. Loads YOLOv8 nano model (`yolov8n.pt`)
2. Accesses default webcam (index 0)
3. Processes video frames in real-time
4. Draws bounding boxes and labels
5. Displays annotated video feed
6. Shows FPS counter

**Key features:**
- Single Python script
- No web interface required
- Immediate visual feedback
- Press 'q' to quit
- Optimized for laptop CPU

**Usage:**
```bash
python detect.py
```

**Requirements:**
- Python 3.8+
- ultralytics
- opencv-python
- Working webcam

## Web Application

The full web application (`app.py`) provides a comprehensive interface:

### Features

1. **Live Camera Feed**
   - Real-time object detection from webcam
   - WebSocket streaming for low latency
   - FPS monitoring and statistics
   - Start/stop controls

2. **File Upload Processing**
   - Support for images: JPG, PNG, BMP, TIFF
   - Support for videos: MP4, AVI, MOV, MKV, WebM
   - Drag-and-drop interface
   - Progress tracking for video processing
   - Maximum file size: 50MB

3. **Real-time Statistics**
   - Current FPS display
   - Objects detected count
   - Processing time metrics
   - Connection status monitoring

4. **Advanced Settings**
   - Confidence threshold adjustment (0.1 - 0.9)
   - Frame skip configuration for performance tuning
   - Statistics reset functionality

5. **Responsive Design**
   - Mobile-friendly interface
   - Dark theme with glass morphism effects
   - Smooth animations and transitions
   - Intuitive user experience

### Supported Object Classes

The system detects 80+ object classes from the COCO dataset:

**People & Animals**
- person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles**
- car, motorcycle, airplane, bus, train, truck, boat

**Sports & Recreation**
- bicycle, sports ball, kite, baseball bat, skateboard, surfboard, tennis racket

**Electronics**
- cell phone, laptop, mouse, remote, keyboard, microwave, oven, toaster

**Food Items**
- banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

**Household Objects**
- chair, couch, potted plant, bed, dining table, toilet, tv, refrigerator

**And many more...**

## Deployment

### Development Environment

1. **Setup Development Environment**
   ```bash
   # Clone and setup
   git clone https://github.com/aravinditte/realtime-object-recognition.git
   cd realtime-object-recognition
   python setup.py
   
   # Create virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Run Development Server**
   ```bash
   # Simple run
   python app.py
   
   # Production runner with options
   python run.py --host 0.0.0.0 --port 5000 --model yolov8s.pt
   
   # Debug mode
   python run.py --debug
   ```

### Docker Deployment

1. **Using Docker Compose (Recommended)**
   ```bash
   # Start all services
   docker-compose up -d
   
   # View logs
   docker-compose logs -f
   
   # Stop services
   docker-compose down
   ```

2. **Manual Docker Build**
   ```bash
   # Build image
   docker build -t realtime-object-detection .
   
   # Run container
   docker run -d \
     --name object-detection \
     -p 5000:5000 \
     -v $(pwd)/uploads:/app/uploads \
     realtime-object-detection
   
   # Check health
   docker exec object-detection python -c "import requests; print(requests.get('http://localhost:5000/health').json())"
   ```

### Production Deployment

1. **Environment Variables**
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit configuration
   nano .env
   ```

2. **Key Production Settings**
   ```bash
   FLASK_ENV=production
   SECRET_KEY=your-secure-random-key
   MODEL_NAME=yolov8s.pt  # Better accuracy for production
   MAX_CONTENT_LENGTH=104857600  # 100MB for larger files
   CONFIDENCE_THRESHOLD=0.25  # Lower for more detections
   ```

3. **Reverse Proxy Setup (Nginx)**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
       
       location /socket.io/ {
           proxy_pass http://127.0.0.1:5000;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_set_header Origin "";
       }
   }
   ```

### Cloud Deployment

**AWS ECS/Fargate**
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker build -t realtime-object-detection .
docker tag realtime-object-detection:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/realtime-object-detection:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/realtime-object-detection:latest
```

**Google Cloud Run**
```bash
# Deploy to Cloud Run
gcloud run deploy realtime-object-detection \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600
```

## Configuration

### Model Selection

| Model | Size | Speed | mAP | Use Case |
|-------|------|-------|-----|-----------|
| yolov8n.pt | 6.2MB | 80+ FPS | 37.3 | Real-time, mobile, demo |
| yolov8s.pt | 21.5MB | 50+ FPS | 44.9 | Balanced performance |
| yolov8m.pt | 49.7MB | 30+ FPS | 50.2 | High accuracy needs |
| yolov8l.pt | 83.7MB | 20+ FPS | 52.9 | Production quality |
| yolov8x.pt | 130.5MB | 15+ FPS | 53.9 | Maximum accuracy |

### Performance Tuning

**For Real-time Applications:**
- Model: `yolov8n.pt`
- Input size: 640x384 or smaller
- Frame skip: 1-2 frames
- Confidence threshold: 0.4+

**For High Accuracy:**
- Model: `yolov8l.pt` or `yolov8x.pt`
- Input size: 640x640 or larger
- Frame skip: 0
- Confidence threshold: 0.25

**For Resource-Constrained Environments:**
- Model: `yolov8n.pt`
- Input size: 320x320
- Frame skip: 3-5 frames
- Reduce frame queue size

### Environment Variables

Comprehensive configuration options in `.env` file:

```bash
# Core Settings
FLASK_ENV=production
HOST=0.0.0.0
PORT=5000
SECRET_KEY=change-this-in-production

# AI Model Settings
MODEL_NAME=yolov8n.pt
CONFIDENCE_THRESHOLD=0.30
IOU_THRESHOLD=0.45

# Performance Settings
INFER_WIDTH=640
INFER_HEIGHT=384
FRAME_QUEUE_MAX=3
MAX_CONTENT_LENGTH=52428800

# Security Settings
CORS_ORIGINS=*
RATE_LIMIT_PER_MINUTE=60
MAX_CONNECTIONS=10
```

## Performance

### Benchmarks

**Hardware:** Intel i7-8750H, 16GB RAM, GTX 1060

| Model | Input Size | CPU FPS | GPU FPS | Memory |
|-------|------------|---------|---------|--------|
| YOLOv8n | 640x640 | 45 | 120 | 1.2GB |
| YOLOv8s | 640x640 | 30 | 90 | 1.8GB |
| YOLOv8m | 640x640 | 20 | 60 | 2.5GB |

**Optimizations Implemented:**
- Multi-threaded frame processing
- Frame queue management
- Model caching and reuse
- WebSocket connection pooling
- Asynchronous file processing
- Memory management and garbage collection

### System Requirements

**Minimum:**
- 2GB RAM
- 1 CPU core
- 2GB storage
- Python 3.8+

**Recommended:**
- 4GB+ RAM
- 2+ CPU cores
- 4GB+ storage
- Python 3.10+
- GPU (optional but recommended)

**Production:**
- 8GB+ RAM
- 4+ CPU cores
- 10GB+ storage
- Load balancer
- Database for statistics
- Monitoring stack

## API Documentation

### REST Endpoints

**Health Check**
```http
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "yolov8n.pt",
  "connections": 2,
  "statistics": {
    "total_frames_processed": 1234,
    "total_detections": 567,
    "average_fps": 45.2,
    "average_processing_time": 22.1
  },
  "timestamp": "2024-11-01T12:00:00Z"
}
```

**File Upload**
```http
POST /upload
Content-Type: multipart/form-data

Form Data:
- file: (binary file data)

Response (Image):
{
  "type": "image",
  "result": "base64-encoded-image",
  "detections": 3,
  "message": "Detected 3 objects in the image"
}

Response (Video):
{
  "type": "video",
  "filepath": "/tmp/video_123456.mp4",
  "message": "Video uploaded successfully. Start processing via WebSocket."
}
```

### WebSocket Events

**Client to Server:**

```javascript
// Send webcam frame for processing
socket.emit('webcam_frame', {
  image: 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA...'
});

// Start video processing
socket.emit('process_video', {
  filepath: '/tmp/uploaded_video.mp4'
});

// Stop any active processing
socket.emit('stop_processing');
```

**Server to Client:**

```javascript
// Connection established
socket.on('connection_status', (data) => {
  console.log(data.message); // "Connected to real-time object detection server"
});

// Processed webcam frame
socket.on('processed_frame', (data) => {
  /*
  {
    image: 'base64-encoded-result',
    detections: 2,
    fps: 45.2,
    processing_time: 22.1
  }
  */
});

// Video processing frame
socket.on('video_frame', (data) => {
  /*
  {
    image: 'base64-encoded-frame',
    frame_number: 150,
    total_frames: 500,
    progress: 30.0,
    detections: 1
  }
  */
});

// Processing complete
socket.on('video_complete', (data) => {
  console.log(`Processed ${data.total_frames_processed} frames`);
});

// Error handling
socket.on('error', (data) => {
  console.error('Error:', data.message);
});
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/aravinditte/realtime-object-recognition.git
cd realtime-object-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 pre-commit

# Setup pre-commit hooks
pre-commit install

# Run setup script
python setup.py
```

### Code Structure

```
realtime-object-recognition/
├── app.py                 # Main Flask application
├── detect.py              # MVP script for quick demo
├── run.py                 # Production runner with monitoring
├── setup.py               # System setup and configuration
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container build configuration
├── docker-compose.yml    # Multi-container deployment
├── .env.example          # Environment configuration template
├── .github/
│   └── workflows/
│       └── ci-cd.yml     # CI/CD pipeline configuration
├── templates/
│   └── index.html        # Web application frontend
├── uploads/              # Temporary file storage
├── logs/                 # Application logs
└── README.md            # This file
```

### Development Commands

```bash
# Format code
black .

# Lint code
flake8 .

# Run tests
pytest

# Run with hot reload
FLASK_ENV=development python app.py

# Check application health
python run.py --check-health

# Run with custom model
python run.py --model yolov8s.pt

# Debug mode with logging
python run.py --debug --host 127.0.0.1
```

### Adding New Features

1. **Adding New Object Detection Models**
   ```python
   # In app.py, modify the ObjectDetector class
   def load_model(self, model_name):
       # Add support for new model formats
       if model_name.endswith('.onnx'):
           # ONNX model loading logic
       elif model_name.endswith('.pt'):
           # PyTorch model loading logic
   ```

2. **Adding New API Endpoints**
   ```python
   @app.route('/api/v1/models')
   def list_models():
       return jsonify({
           'available_models': ['yolov8n.pt', 'yolov8s.pt'],
           'current_model': CONFIG['MODEL_NAME']
       })
   ```

3. **Adding New WebSocket Events**
   ```python
   @socketio.on('custom_event')
   def handle_custom_event(data):
       # Process custom event
       emit('custom_response', {'status': 'processed'})
   ```

## Testing

### Unit Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_detection.py

# Run tests in verbose mode
pytest -v
```

### Integration Tests

```bash
# Test model loading
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Test application startup
python -c "from app import app, detector; print('Import successful')"

# Test OpenCV functionality
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera access OK' if cap.isOpened() else 'Camera access failed'); cap.release()"
```

### Performance Tests

```bash
# Load testing with hey (install: go install github.com/rakyll/hey@latest)
hey -n 1000 -c 10 http://localhost:5000/health

# Memory usage monitoring
ps aux | grep python

# Docker resource usage
docker stats realtime-object-detection
```

### Manual Testing Checklist

- [ ] MVP script runs successfully
- [ ] Web application starts without errors
- [ ] Webcam access works
- [ ] Real-time detection displays objects
- [ ] Image upload and processing works
- [ ] Video upload and processing works
- [ ] WebSocket connections stable
- [ ] Statistics update correctly
- [ ] Settings changes take effect
- [ ] Error handling works properly
- [ ] Mobile interface responsive
- [ ] Docker container builds and runs
- [ ] Health check endpoint responds
- [ ] File size limits enforced
- [ ] Security measures active

## Troubleshooting

### Common Issues

**1. Model Loading Fails**
```bash
# Clear model cache
rm -rf ~/.cache/torch/hub/ultralytics_yolov8*

# Reinstall ultralytics
pip uninstall ultralytics
pip install ultralytics

# Download model manually
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**2. Camera Access Denied**
```bash
# Check camera permissions
ls -la /dev/video*

# Test camera with OpenCV
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened()); cap.release()"

# For Linux users
sudo usermod -a -G video $USER
```

**3. WebSocket Connection Issues**
```bash
# Check SocketIO compatibility
pip install python-socketio==5.9.0 flask-socketio==5.3.6

# Verify CORS settings
# In app.py, check: socketio = SocketIO(app, cors_allowed_origins="*")

# Test in different browsers (Chrome recommended)
```

**4. Memory Issues**
```bash
# Use smaller model
MODEL_NAME=yolov8n.pt python app.py

# Reduce frame queue size
FRAME_QUEUE_MAX=1 python app.py

# Monitor memory usage
top -p $(pgrep -f "python app.py")
```

**5. Docker Build Issues**
```bash
# Clean Docker cache
docker system prune -a

# Build with no cache
docker build --no-cache -t realtime-object-detection .

# Check Docker logs
docker logs realtime-object-detection
```

**6. No Objects Detected**
```bash
# Lower confidence threshold
CONFIDENCE_THRESHOLD=0.1 python app.py

# Test with clear, well-lit objects
# Try pointing camera at common objects (person, car, chair, etc.)

# Verify model is correct version
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print(model.names)"
```

**7. Poor Performance**
```bash
# Use CPU-optimized settings
INFER_WIDTH=320 INFER_HEIGHT=320 python app.py

# Enable frame skipping
# In web interface: adjust "Frame Skip" slider

# Close unnecessary applications
# Monitor CPU usage: htop or Task Manager
```

### Debug Mode

```bash
# Enable debug logging
FLASK_ENV=development python run.py --debug

# Check application logs
tail -f logs/app.log

# Monitor WebSocket connections
# Open browser developer tools -> Network -> WS
```

### Performance Optimization

```bash
# Profile the application
python -m cProfile -o profile.stats app.py

# Analyze profile results
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"

# Memory profiling with memory_profiler
pip install memory_profiler
python -m memory_profiler app.py
```

### Getting Help

If you're still experiencing issues:

1. **Check the logs** in `logs/app.log`
2. **Review environment variables** in `.env`
3. **Test with MVP script** first: `python detect.py`
4. **Check GitHub Issues** for similar problems
5. **Create a new issue** with:
   - System information (OS, Python version)
   - Error messages and logs
   - Steps to reproduce
   - Expected vs actual behavior

## Contributing

We welcome contributions! Please follow these guidelines:

### Getting Started

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Run tests and linting**
   ```bash
   black .
   flake8 .
   pytest
   ```
5. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push to your branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Development Guidelines

- **Code Style**: Follow PEP 8, use Black for formatting
- **Documentation**: Update README and docstrings
- **Testing**: Add tests for new features
- **Commits**: Use conventional commit messages
- **Performance**: Profile performance-critical changes
- **Security**: Follow security best practices

### Areas for Contribution

- **Performance Optimization**: GPU acceleration, model optimization
- **New Features**: Additional object classes, tracking algorithms
- **UI/UX Improvements**: Better interface, mobile optimization
- **Deployment**: Kubernetes manifests, cloud templates
- **Testing**: More comprehensive test coverage
- **Documentation**: Tutorials, API documentation
- **Security**: Vulnerability fixes, security enhancements

### Pull Request Process

1. **Ensure tests pass** on your changes
2. **Update documentation** if needed
3. **Add/update tests** for new functionality
4. **Follow the coding standards**
5. **Provide clear description** of changes
6. **Link related issues** if applicable

### Code of Conduct

By participating in this project, you agree to:
- Be respectful and inclusive
- Provide constructive feedback
- Focus on what's best for the community
- Help others learn and grow

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### What this means:
- ✅ **Commercial use** - Use in commercial projects
- ✅ **Modification** - Modify and distribute
- ✅ **Distribution** - Share with others
- ✅ **Private use** - Use in private projects
- ❌ **Liability** - No warranty provided
- ❌ **Trademark use** - No trademark rights granted

## Acknowledgments

- **[Ultralytics](https://ultralytics.com)** - For the incredible YOLO models and framework
- **[OpenCV](https://opencv.org)** - For computer vision capabilities
- **[Flask](https://flask.palletsprojects.com)** - For the lightweight web framework
- **[Socket.IO](https://socket.io)** - For real-time communication
- **[Bootstrap](https://getbootstrap.com)** - For the responsive UI framework
- **[COCO Dataset](https://cocodataset.org)** - For the comprehensive object detection dataset
- **Open Source Community** - For the countless contributors who make projects like this possible

---

## Author

**Aravind Itte**
- 🌐 **GitHub**: [@aravinditte](https://github.com/aravinditte)
- 💼 **LinkedIn**: [Connect with me](https://linkedin.com/in/aravinditte)
- 📧 **Email**: aravinditte@gmail.com
- 🚀 **Portfolio**: Building innovative AI solutions

---

<div align="center">
  <h3>⭐ Star this repository if it helped you! ⭐</h3>
  <p>Made with ❤️ by <a href="https://github.com/aravinditte">Aravind Itte</a></p>
  <p><strong>Real-Time Object Detection System</strong> - Production-ready AI powered by YOLOv8</p>
</div>
