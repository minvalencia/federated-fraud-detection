# Docker Guide - Fraud Detection API

## Prerequisites

1. **Install Docker**
   - Download from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
   - Install Docker Desktop
   - Verify installation:
     ```bash
     docker --version
     docker-compose --version
     ```

2. **Project Structure**
   ```
   project/
   ├── src/
   │   ├── api/
   │   │   ├── main.py
   │   │   └── ml_model.py
   │   ├── models/
   │   │   └── fraud_detector.py
   │   └── utils/
   │       └── data_preprocessor.py
   ├── data/
   ├── Dockerfile
   ├── docker-compose.yml
   └── requirements.txt
   ```

## Docker Commands

### 1. Building and Starting the API

```bash
# Build and start containers in detached mode
docker-compose up -d

# Build without starting
docker-compose build

# Build with no cache (clean build)
docker-compose build --no-cache

# Build and start specific service
docker-compose up -d fraud-api
```

### 2. Managing Containers

```bash
# Stop all containers
docker-compose down

# Stop specific container
docker-compose stop fraud-api

# Restart containers
docker-compose restart

# View running containers
docker ps

# View all containers (including stopped)
docker ps -a
```

### 3. Viewing Logs

```bash
# View logs of all containers
docker-compose logs

# View logs of specific container
docker-compose logs fraud-api

# Follow logs in real-time
docker-compose logs -f

# View last 100 lines
docker-compose logs --tail=100
```

### 4. Cleaning Up

```bash
# Remove all containers and networks
docker-compose down

# Remove all containers, networks, and volumes
docker-compose down -v

# Remove all unused containers, networks, images
docker system prune

# Remove all unused volumes
docker volume prune
```

### 5. Debugging

```bash
# Enter container shell
docker-compose exec fraud-api bash

# View container details
docker inspect fraud-api

# View resource usage
docker stats
```

## Environment Variables

### Default Configuration
```yaml
# docker-compose.yml
environment:
  - PORT=8000
  - HOST=0.0.0.0
  - DEBUG=False
  - API_TOKEN=your_default_token
```

### Custom Configuration
Create a `.env` file:
```env
PORT=8080
DEBUG=True
API_TOKEN=your_custom_token
```

## Common Use Cases

### 1. Starting Fresh
```bash
# Stop and remove everything
docker-compose down -v

# Remove all images
docker rmi $(docker images -q)

# Build and start fresh
docker-compose up -d --build
```

### 2. Updating the API
```bash
# Pull latest code changes
git pull

# Rebuild and restart
docker-compose up -d --build
```

### 3. Scaling Services
```bash
# Scale to multiple instances
docker-compose up -d --scale fraud-api=3
```

### 4. Production Deployment
```bash
# Use production configuration
docker-compose -f docker-compose.prod.yml up -d

# With specific environment file
docker-compose --env-file .env.prod up -d
```

## Troubleshooting

### 1. Container Won't Start
```bash
# Check logs
docker-compose logs fraud-api

# Verify configuration
docker-compose config

# Check for port conflicts
netstat -ano | findstr :8000
```

### 2. Performance Issues
```bash
# Monitor resources
docker stats

# Check container health
docker inspect fraud-api | grep Health

# View container processes
docker top fraud-api
```

### 3. Network Issues
```bash
# List networks
docker network ls

# Inspect network
docker network inspect 1project_default

# Test network connectivity
docker-compose exec fraud-api ping database
```

## Best Practices

1. **Resource Management**
   ```yaml
   # In docker-compose.yml
   services:
     fraud-api:
       deploy:
         resources:
           limits:
             cpus: '0.50'
             memory: 512M
           reservations:
             cpus: '0.25'
             memory: 256M
   ```

2. **Health Checks**
   ```yaml
   # In docker-compose.yml
   services:
     fraud-api:
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
         interval: 30s
         timeout: 10s
         retries: 3
   ```

3. **Logging Configuration**
   ```yaml
   # In docker-compose.yml
   services:
     fraud-api:
       logging:
         driver: "json-file"
         options:
           max-size: "10m"
           max-file: "3"
   ```

## Quick Reference

### Start Development Environment
```bash
# Start everything
docker-compose up -d

# View logs
docker-compose logs -f

# Stop everything
docker-compose down
```

### Manage Data
```bash
# Create data directory
mkdir -p data/uploads

# Copy files to container
docker cp ./data/sample.csv fraud-api:/app/data/

# Backup data
docker cp fraud-api:/app/data/. ./backup/
```

### Update API
```bash
# Pull changes
git pull

# Rebuild and restart
docker-compose up -d --build

# Verify update
docker-compose logs fraud-api
```