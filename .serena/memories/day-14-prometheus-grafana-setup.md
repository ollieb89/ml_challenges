# Day 14: Prometheus Metrics & Grafana Dashboard

## Overview
Implemented real-time monitoring for the Pose Analysis and GPU Optimization pipeline using Prometheus and Grafana.

## Key Components

### 1. Centralized Metrics Management
- **File**: `projects/shared_utils/src/shared_utils/metrics.py`
- **Class**: `MetricsManager` (Singleton)
- **Metrics Endpoint**: Port `8000` (listening on `0.0.0.0` for Docker accessibility).
- **Core Metrics**:
  - `pose_fps`: Gauge, labeled by `stream_id`.
  - `pose_latency_ms`: Summary, labeled by `stream_id`.
  - `vram_usage_mb`: Gauge, labeled by `device_id`.
  - `vram_fragmentation_ratio`: Gauge, labeled by `device_id`.

### 2. Instrumentation
- **Pose Analyzer**: `ConcurrentStreamProcessor` updates FPS and latency in the inference loop.
- **GPU Optimizer**: `FragmentationSolver` updates VRAM usage and fragmentation ratio during monitoring.

### 3. Infrastructure (Docker)
- **Compose File**: `docker-compose.monitoring.yml`
- **Prometheus**:
  - Config: `config/prometheus.yml`
  - Scrapes `host.docker.internal:8000`.
  - Uses `extra_hosts: host.docker.internal:host-gateway` for Linux host access.
- **Grafana**:
  - Port: `7000` (mapped to internal `3000`).
  - **Auto-Provisioning**:
    - Datasource: `config/grafana/provisioning/datasources/prometheus.yml` (UID: `prometheus_ds`).
    - Dashboard: `config/grafana/provisioning/dashboards/provider.yml` pointing to `config/grafana_dashboard.json`.

## Resolution of "No Data" Issues
1. **Networking**: On Linux, Docker containers cannot access `localhost` of the host. `extra_hosts` was added to map `host.docker.internal` to the host gateway.
2. **Binding**: The Python metrics server must bind to `0.0.0.0` instead of `127.0.0.1` to accept connections from the container network.
3. **Provisioning**: Grafana was set up with automatic provisioning to ensure the Prometheus datasource and custom dashboard are available immediately upon container start.

## Verification
Verified using `curl -s http://localhost:8000/metrics` and checking Prometheus target status via API.
