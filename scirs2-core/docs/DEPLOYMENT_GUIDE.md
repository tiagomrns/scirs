# Deployment Guide for scirs2-core

## Overview

This guide provides comprehensive instructions for deploying applications built with scirs2-core in production environments. It covers deployment strategies, performance tuning, monitoring, and operational best practices.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Configuration Management](#configuration-management)
4. [Performance Tuning](#performance-tuning)
5. [Container Deployment](#container-deployment)
6. [Cloud Deployment](#cloud-deployment)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [High Availability](#high-availability)
9. [Backup and Recovery](#backup-and-recovery)
10. [Deployment Checklist](#deployment-checklist)

## System Requirements

### Minimum Requirements

- **CPU**: x86_64 or ARM64 processor with SSE4.2 support
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for installation
- **OS**: Linux (kernel 4.19+), macOS 10.15+, Windows 10+

### Recommended Requirements

- **CPU**: Multi-core processor with AVX2/AVX-512 support
- **RAM**: 32GB+ for large-scale computations
- **Storage**: SSD with 100GB+ free space
- **GPU**: CUDA 11.0+ compatible GPU (optional)

### Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    pkg-config

# macOS
brew install cmake openblas lapack

# Windows (using vcpkg)
vcpkg install openblas lapack
```

## Installation Methods

### Binary Installation

```bash
# Download pre-built binaries
curl -L https://github.com/scirs2/releases/download/v0.1.0-alpha.5/scirs2-linux-x64.tar.gz | tar xz
sudo mv scirs2 /usr/local/bin/

# Verify installation
scirs2 --version
```

### From Source

```bash
# Clone repository
git clone https://github.com/scirs2/scirs2-core.git
cd scirs2-core

# Build with optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release --all-features

# Install
cargo install --path . --all-features
```

### Package Managers

```bash
# Cargo
cargo install scirs2-core --all-features

# Conda (coming soon)
conda install -c scirs2 scirs2-core

# Docker
docker pull scirs2/scirs2-core:latest
```

## Configuration Management

### Environment Variables

```bash
# Core settings
export SCIRS2_LOG_LEVEL=info
export SCIRS2_THREAD_POOL_SIZE=16
export SCIRS2_MEMORY_LIMIT=32GB

# GPU settings
export SCIRS2_GPU_ENABLED=true
export SCIRS2_GPU_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=0,1

# Performance settings
export SCIRS2_SIMD_ENABLED=true
export SCIRS2_PARALLEL_THRESHOLD=1000
```

### Configuration File

Create `scirs2.toml`:

```toml
[core]
log_level = "info"
thread_pool_size = 16
memory_limit = "32GB"

[performance]
simd_enabled = true
parallel_threshold = 1000
chunk_size = 8192

[gpu]
enabled = true
device_ids = [0, 1]
memory_fraction = 0.8

[cache]
enabled = true
max_size = "4GB"
ttl_seconds = 3600

[monitoring]
metrics_enabled = true
metrics_port = 9090
trace_enabled = false
```

### Loading Configuration

```rust
use scirs2_core::config::{Config, ConfigBuilder};

let config = ConfigBuilder::new()
    .from_env()
    .from_file("/etc/scirs2/config.toml")
    .build()?;

// Apply configuration
scirs2_core::initialize(config)?;
```

## Performance Tuning

### CPU Optimization

```rust
use scirs2_core::parallel::ThreadPoolBuilder;

// Configure thread pool
ThreadPoolBuilder::new()
    .num_threads(num_cpus::get())
    .stack_size(8 * 1024 * 1024) // 8MB stack
    .name_prefix("scirs2-worker")
    .build_global()?;
```

### Memory Optimization

```rust
use scirs2_core::memory_efficient::{ChunkedArray, MemoryConfig};

// Configure memory settings
let mem_config = MemoryConfig::new()
    .with_chunk_size(8192)
    .with_prefetch_distance(4)
    .with_cache_size("2GB")
    .with_mmap_threshold("100MB");

// Use memory-efficient arrays
let array = ChunkedArray::from_file_with_config("data.bin", mem_config)?;
```

### SIMD Optimization

```bash
# Build with native CPU features
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma" cargo build --release

# Runtime detection
export SCIRS2_SIMD_LEVEL=avx512  # auto, sse4, avx2, avx512
```

### GPU Optimization

```rust
use scirs2_core::gpu::{GpuConfig, GpuRuntime};

let gpu_config = GpuConfig::new()
    .with_device_id(0)
    .with_memory_fraction(0.8)
    .with_stream_priority(0);

let runtime = GpuRuntime::new(gpu_config)?;
```

## Container Deployment

### Dockerfile

```dockerfile
# Multi-stage build for minimal image size
FROM rust:1.70 as builder

WORKDIR /app
COPY . .

# Build with optimizations
RUN RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --all-features

# Runtime stage
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    libopenblas-base \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/scirs2 /usr/local/bin/

# Non-root user
RUN useradd -m -u 1000 scirs2
USER scirs2

ENTRYPOINT ["scirs2"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  scirs2:
    image: scirs2/scirs2-core:latest
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          cpus: '2'
          memory: 8G
    environment:
      - SCIRS2_LOG_LEVEL=info
      - SCIRS2_THREAD_POOL_SIZE=8
    volumes:
      - ./data:/data
      - ./config:/config
    ports:
      - "9090:9090"  # Metrics
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: scirs2-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: scirs2
  template:
    metadata:
      labels:
        app: scirs2
    spec:
      containers:
      - name: scirs2
        image: scirs2/scirs2-core:latest
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1  # GPU support
        env:
        - name: SCIRS2_CONFIG
          value: "/config/scirs2.toml"
        volumeMounts:
        - name: config
          mountPath: /config
        - name: data
          mountPath: /data
      volumes:
      - name: config
        configMap:
          name: scirs2-config
      - name: data
        persistentVolumeClaim:
          claimName: scirs2-data-pvc
```

## Cloud Deployment

### AWS EC2

```bash
# Launch instance with GPU support
aws ec2 run-instances \
    --image-id ami-0abcdef1234567890 \
    --instance-type p3.2xlarge \
    --key-name my-key \
    --security-groups scirs2-sg \
    --user-data file://deploy-script.sh

# Deploy script (deploy-script.sh)
#!/bin/bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo docker pull scirs2/scirs2-core:latest
```

### Google Cloud Platform

```bash
# Create instance with GPU
gcloud compute instances create scirs2-instance \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --maintenance-policy=TERMINATE

# Deploy container
gcloud run deploy scirs2-service \
    --image=scirs2/scirs2-core:latest \
    --platform=managed \
    --region=us-central1 \
    --memory=16Gi \
    --cpu=4
```

### Azure

```bash
# Create container instance
az container create \
    --resource-group scirs2-rg \
    --name scirs2-container \
    --image scirs2/scirs2-core:latest \
    --cpu 4 \
    --memory 16 \
    --gpu-count 1 \
    --gpu-sku K80
```

## Monitoring and Observability

### Prometheus Integration

```rust
use scirs2_core::monitoring::{MetricsServer, PrometheusExporter};

// Start metrics server
let metrics_server = MetricsServer::new()
    .with_port(9090)
    .with_exporter(PrometheusExporter::new())
    .start()?;
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "scirs2 Performance Metrics",
    "panels": [
      {
        "title": "Array Operations/sec",
        "targets": [{
          "expr": "rate(scirs2_array_ops_total[5m])"
        }]
      },
      {
        "title": "Memory Usage",
        "targets": [{
          "expr": "scirs2_memory_bytes_used"
        }]
      },
      {
        "title": "GPU Utilization",
        "targets": [{
          "expr": "scirs2_gpu_utilization_percent"
        }]
      }
    ]
  }
}
```

### Logging Configuration

```rust
use scirs2_core::logging::{Logger, LogConfig};

let log_config = LogConfig::new()
    .with_level("info")
    .with_format("json")
    .with_output("/var/log/scirs2/app.log")
    .with_rotation("daily");

Logger::init(log_config)?;
```

### OpenTelemetry Tracing

```rust
use scirs2_core::tracing::{TracingConfig, OtelExporter};

let tracing_config = TracingConfig::new()
    .with_service_name("scirs2-app")
    .with_endpoint("http://jaeger:14268/api/traces")
    .with_sample_rate(0.1);

scirs2_core::tracing::init(tracing_config)?;
```

## High Availability

### Load Balancing

```nginx
upstream scirs2_backend {
    least_conn;
    server scirs2-1:8080 weight=10 max_fails=3 fail_timeout=30s;
    server scirs2-2:8080 weight=10 max_fails=3 fail_timeout=30s;
    server scirs2-3:8080 weight=10 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    location / {
        proxy_pass http://scirs2_backend;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Health Checks

```rust
use scirs2_core::health::{HealthCheck, HealthStatus};

// Implement health endpoint
async fn health_check() -> HealthStatus {
    HealthCheck::new()
        .check_memory()
        .check_gpu()
        .check_dependencies()
        .execute()
        .await
}
```

### Failover Strategy

```yaml
# Kubernetes readiness probe
readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5

# Liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
```

## Backup and Recovery

### Data Backup

```bash
#!/bin/bash
# Backup script
BACKUP_DIR="/backup/scirs2/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup data files
rsync -av --progress /data/scirs2/ "$BACKUP_DIR/data/"

# Backup configuration
cp -r /etc/scirs2/ "$BACKUP_DIR/config/"

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"

# Upload to S3
aws s3 cp "$BACKUP_DIR.tar.gz" s3://backup-bucket/scirs2/
```

### Disaster Recovery

```rust
use scirs2_core::recovery::{RecoveryManager, BackupConfig};

let backup_config = BackupConfig::new()
    .with_s3_bucket("backup-bucket")
    .with_retention_days(30)
    .with_encryption(true);

let recovery = RecoveryManager::new(backup_config);

// Automatic backup
recovery.schedule_backup("0 2 * * *")?; // Daily at 2 AM

// Recovery
recovery.restore_from_latest()?;
```

## Deployment Checklist

### Pre-Deployment

- [ ] System requirements verified
- [ ] Dependencies installed and tested
- [ ] Configuration files prepared
- [ ] Security settings reviewed
- [ ] Backup strategy implemented

### Deployment

- [ ] Application deployed to target environment
- [ ] Configuration loaded correctly
- [ ] Health checks passing
- [ ] Monitoring enabled
- [ ] Logging configured

### Post-Deployment

- [ ] Performance metrics within acceptable range
- [ ] No errors in logs
- [ ] Alerts configured
- [ ] Documentation updated
- [ ] Team trained on operations

### Production Readiness

- [ ] Load testing completed
- [ ] Disaster recovery tested
- [ ] Security audit passed
- [ ] SLA requirements met
- [ ] Operational runbook created

## Troubleshooting Common Issues

### Memory Issues

```bash
# Check memory usage
free -h
ps aux | grep scirs2

# Increase memory limits
export SCIRS2_MEMORY_LIMIT=64GB
```

### Performance Issues

```bash
# Profile CPU usage
perf record -g ./scirs2
perf report

# Check thread utilization
htop -H
```

### GPU Issues

```bash
# Check GPU availability
nvidia-smi

# Reset GPU
nvidia-smi --gpu-reset

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## Additional Resources

- [Performance Tuning Guide](./PERFORMANCE_TUNING.md)
- [Security Best Practices](./SECURITY_GUIDE.md)
- [API Reference](https://docs.scirs2.org)
- [Community Support](https://github.com/scirs2/scirs2-core/discussions)

---

*Last Updated: 2025-06-22 | Version: 0.1.0-alpha.5*