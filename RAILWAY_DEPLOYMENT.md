# Railway Deployment Guide

This guide will help you deploy the Real-Time Object Detection System on Railway for free using their trial credits or free tier.

## Quick Deploy

### Option 1: One-Click Deploy (Recommended)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/deploy?template=https://github.com/aravinditte/realtime-object-recognition)

Click the button above to deploy directly to Railway.

### Option 2: Manual Deploy from GitHub

1. **Create Railway Account**
   - Go to [Railway.app](https://railway.app)
   - Sign up with GitHub (recommended)
   - Verify your GitHub account to access deployment features

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose `aravinditte/realtime-object-recognition`
   - Railway will automatically detect the Dockerfile

3. **Configure Environment Variables**
   
   In your Railway dashboard, go to your service > Variables and add:
   
   ```bash
   FLASK_ENV=production
   HOST=0.0.0.0
   PORT=5000
   SECRET_KEY=your-secure-random-key-here
   MODEL_NAME=yolov8n.pt
   CONFIDENCE_THRESHOLD=0.30
   IOU_THRESHOLD=0.45
   INFER_WIDTH=640
   INFER_HEIGHT=384
   FRAME_QUEUE_MAX=3
   MAX_CONTENT_LENGTH=52428800
   ```

4. **Generate Domain**
   - Go to Settings > Networking
   - Click "Generate Domain"
   - Your app will be available at `https://your-app-name.up.railway.app`

5. **Deploy**
   - Railway will automatically build and deploy your application
   - First deployment may take 5-10 minutes due to YOLO model download

## Railway Free Tier Information

### Current Free Tier (2024)
- **Trial Credits**: New users get $5 in trial credits
- **Monthly Allowance**: $5/month worth of usage after trial
- **Resource Limits**: 
  - 512MB RAM per service
  - Shared CPU
  - Apps sleep after 30 minutes of inactivity
  - 100GB network bandwidth

### Resource Optimization for Free Tier

To stay within free limits, use these optimizations:

```bash
# Lightweight model for faster startup and lower memory usage
MODEL_NAME=yolov8n.pt

# Reduced inference resolution to save CPU/memory
INFER_WIDTH=320
INFER_HEIGHT=240

# Lower frame queue to reduce memory usage
FRAME_QUEUE_MAX=1

# Higher confidence threshold to reduce processing
CONFIDENCE_THRESHOLD=0.5
```

## Deployment Configuration Files

The repository includes several Railway-specific configuration files:

### `railway.json` (Main Configuration)
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "python app.py",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 300,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### `railway.toml` (Alternative Configuration)
Provides the same configuration in TOML format with additional watch patterns.

### `nixpacks.toml` (Fallback)
Used if Dockerfile deployment fails, provides Nixpacks-based build configuration.

### `Procfile` (Process Definition)
Defines web process and release commands for the application.

### `runtime.txt` (Python Version)
Specifies Python 3.11.9 for consistent runtime.

## Troubleshooting Railway Deployment

### Common Issues

1. **Build Fails with OpenGL Dependencies**
   
   **Solution**: The updated Dockerfile fixes this by replacing `libgl1-mesa-glx` with `libgl1` and `libglx-mesa0`.
   
   If still failing, try these steps:
   - Redeploy the service
   - Check Railway build logs for specific errors
   - Consider using Nixpacks deployment instead of Dockerfile

2. **Out of Memory During Build**
   
   **Symptoms**: Build fails with "Killed" or memory errors
   
   **Solutions**:
   - Use the pre-built Docker image approach (see Advanced section)
   - Optimize Dockerfile by removing unnecessary packages
   - Use smaller base image

3. **App Crashes on Startup**
   
   **Check these**:
   - Ensure all environment variables are set
   - Check Railway logs for Python errors
   - Verify YOLO model download completed
   - Test health endpoint: `/health`

4. **WebSocket Connection Issues**
   
   **Solutions**:
   - Ensure Railway domain is used (not localhost)
   - Check browser console for CORS errors
   - Verify Socket.IO client version compatibility
   - Test with different browsers (Chrome recommended)

5. **Model Loading Timeout**
   
   **Solutions**:
   - Increase `healthcheckTimeout` in railway.json to 600 seconds
   - Use smaller model (`yolov8n.pt` instead of larger variants)
   - Check Railway deployment logs for download progress

### Debug Railway Deployment

1. **Check Deployment Logs**
   ```bash
   # In Railway dashboard
   # Go to Deployments > Latest > View Logs
   ```

2. **Test Health Endpoint**
   ```bash
   curl https://your-app.up.railway.app/health
   ```

3. **Monitor Resource Usage**
   - Railway dashboard shows CPU, Memory, and Network usage
   - Monitor to stay within free tier limits

## Advanced Deployment Options

### Option 1: Pre-built Docker Image

For faster deployments, you can use a pre-built image:

1. Build locally:
   ```bash
   docker build -t your-username/realtime-object-detection .
   docker push your-username/realtime-object-detection
   ```

2. In Railway, create service from Docker image:
   - New Project > Deploy Docker Image
   - Use: `your-username/realtime-object-detection:latest`

### Option 2: GitHub Container Registry

1. Build and push to GHCR:
   ```bash
   docker build -t ghcr.io/aravinditte/realtime-object-recognition .
   docker push ghcr.io/aravinditte/realtime-object-recognition
   ```

2. Deploy from GHCR in Railway:
   - Use: `ghcr.io/aravinditte/realtime-object-recognition:latest`

### Option 3: Nixpacks Deployment

If Dockerfile fails, Railway can use Nixpacks:

1. Remove or rename `Dockerfile` temporarily
2. Railway will auto-detect Python and use `nixpacks.toml`
3. This approach installs system dependencies via Nix

## Performance Optimization for Railway

### Memory Optimization
```python
# In your Railway environment variables:
INFER_WIDTH=320          # Smaller input resolution
INFER_HEIGHT=240         # Reduces memory usage
FRAME_QUEUE_MAX=1        # Minimal frame buffering
CONFIDENCE_THRESHOLD=0.5 # Higher threshold = fewer detections
```

### CPU Optimization
```python
# Use frame skipping in the web interface
# Set frame skip to 2-3 for better performance
# This processes every 2nd or 3rd frame
```

### Network Optimization
```python
# Reduce image quality for WebSocket streaming
# In app.py, modify encode_image_to_base64:
_, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 70])
```

## Cost Management

### Staying Within Free Limits

1. **Monitor Usage**
   - Check Railway dashboard regularly
   - Set up usage alerts if available
   - Monitor CPU and memory consumption

2. **Optimize for Efficiency**
   - Use sleep/wake patterns (apps auto-sleep on Railway)
   - Implement request batching
   - Cache YOLO model properly

3. **Scale Down When Not Used**
   - Railway automatically sleeps inactive apps
   - Consider manual scaling during low usage periods

### Usage Estimates

**Typical usage on Railway free tier**:
- Light usage (few hours/day): ~$1-2/month
- Moderate usage (daily use): ~$3-4/month
- Heavy usage (continuous): May exceed $5/month limit

## Alternative Free Hosting Options

If Railway free tier is insufficient:

1. **Render** - 750 hours/month free tier
2. **Fly.io** - Generous free tier with better resource limits
3. **Google Cloud Run** - 2 million requests/month free
4. **Vercel** - For frontend-only deployment
5. **Netlify** - Static site hosting with functions

## Railway CLI Deployment

For advanced users, you can use Railway CLI:

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login and Deploy**
   ```bash
   railway login
   railway link  # Link to existing project or create new
   railway up    # Deploy current directory
   ```

3. **Set Environment Variables**
   ```bash
   railway variables set FLASK_ENV=production
   railway variables set MODEL_NAME=yolov8n.pt
   # ... add other variables
   ```

4. **Monitor Deployment**
   ```bash
   railway logs     # View application logs
   railway status   # Check deployment status
   ```

## Success Checklist

After deployment, verify these work:

- [ ] App builds successfully without errors
- [ ] Health endpoint responds: `https://your-app.up.railway.app/health`
- [ ] Main page loads with UI
- [ ] WebSocket connection establishes (check browser console)
- [ ] Image upload and processing works
- [ ] Live camera detection works (if camera available)
- [ ] No CORS errors in browser console
- [ ] App stays within Railway resource limits

## Getting Help

If you encounter issues:

1. **Check Railway Status**: [status.railway.app](https://status.railway.app)
2. **Railway Discord**: Join Railway community for support
3. **GitHub Issues**: Create issue in this repository
4. **Railway Documentation**: [docs.railway.app](https://docs.railway.app)

## Next Steps

After successful deployment:

1. **Custom Domain**: Add your own domain in Railway settings
2. **HTTPS**: Railway provides automatic HTTPS
3. **Analytics**: Monitor usage and performance
4. **Scaling**: Upgrade to Railway Pro if needed
5. **CI/CD**: Set up automatic deployments on git push

---

**Happy Deploying!** 🚀

Your real-time object detection system should now be running live on Railway. Share your deployed app URL and start detecting objects in real-time!
