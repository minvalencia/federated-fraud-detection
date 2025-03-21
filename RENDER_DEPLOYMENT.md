# Render Deployment Guide - Fraud Detection API

## Prerequisites

1. **Render Account**
   - Sign up at [render.com](https://render.com)
   - Connect your GitHub account
   - Set up billing information

2. **Project Requirements**
   - Git repository with your code
   - `requirements.txt` file
   - `render.yaml` configuration
   - Working Dockerfile (optional)

## Step-by-Step Deployment

### 1. Prepare Your Repository

1. **Add render.yaml**
```yaml
services:
  - type: web
    name: fraud-detection-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.8.0
      - key: DEBUG
        value: false
      - key: API_TOKEN
        sync: false
    disk:
      name: model-storage
      mountPath: /app/models
      sizeGB: 10
```

2. **Update requirements.txt**
```txt
fastapi>=0.68.0
uvicorn>=0.15.0
python-multipart>=0.0.5
torch>=1.10.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.2
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
prometheus-client>=0.11.0
```

### 2. Deploy on Render

1. **Create New Web Service**
   - Log into Render Dashboard
   - Click "New +"
   - Select "Web Service"
   - Choose your repository

2. **Configure Service**
   - Name: `fraud-detection-api`
   - Environment: `Python 3`
   - Region: Choose nearest
   - Branch: `main`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`

3. **Set Environment Variables**
   ```
   DEBUG=false
   API_TOKEN=your_secure_token
   MODEL_PATH=/app/models
   ```

### 3. Configure Persistent Disk

1. **Add Disk**
   - Go to service settings
   - Navigate to "Disks"
   - Click "Add Disk"
   - Set mount path: `/app/models`
   - Size: 10 GB

### 4. Set Up Custom Domain (Optional)

1. **Add Domain**
   - Go to service settings
   - Navigate to "Custom Domains"
   - Click "Add Custom Domain"
   - Follow DNS configuration instructions

### 5. Enable Auto-Deploy

1. **Configure GitHub Integration**
   - Go to service settings
   - Navigate to "Deploy"
   - Enable "Auto-Deploy"
   - Select branches to auto-deploy

## Monitoring and Management

### 1. Logs
- Access logs from Dashboard
- Filter by:
  - Severity
  - Timestamp
  - Service

### 2. Metrics
- View in Dashboard:
  - CPU usage
  - Memory usage
  - Response times
  - Request count

### 3. Scaling
- Adjust resources:
  - Memory
  - CPU
  - Instances

## Security Configuration

### 1. Environment Variables
```bash
# Required
API_TOKEN=your_secure_token
DEBUG=false

# Optional
LOG_LEVEL=INFO
MAX_WORKERS=4
```

### 2. HTTPS/SSL
- Automatic SSL certificate
- Force HTTPS
- Custom SSL certificates (if needed)

### 3. Network Rules
- IP whitelist
- Rate limiting
- CORS configuration

## Maintenance

### 1. Updates
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt

# Commit and push
git commit -am "Update dependencies"
git push origin main
```

### 2. Backups
- Regular model backups
- Database backups (if used)
- Configuration backups

### 3. Monitoring
- Set up alerts
- Monitor metrics
- Check logs regularly

## Troubleshooting

### 1. Common Issues
- Build failures
- Start command errors
- Memory issues
- Disk space

### 2. Solutions
- Check logs
- Verify environment variables
- Validate dependencies
- Test locally first

## Best Practices

### 1. Development
- Use development branches
- Test before deploying
- Keep dependencies updated

### 2. Security
- Rotate API tokens
- Monitor access logs
- Regular security audits

### 3. Performance
- Optimize model size
- Cache when possible
- Monitor resource usage

## Quick Reference

### Important URLs
```
Dashboard: https://dashboard.render.com
Logs: https://dashboard.render.com/logs
Metrics: https://dashboard.render.com/metrics
```

### Common Commands
```bash
# View logs
render logs

# Scale service
render scale fraud-detection-api

# Restart service
render restart fraud-detection-api
```

### Health Checks
```bash
# API health
curl https://your-api.onrender.com/health

# Model status
curl https://your-api.onrender.com/model/status
```