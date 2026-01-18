# GPU Optimizer API - Memory Tracing Guide

Complete guide for using the MemoryTracer through the FastAPI interface.

## Starting the API Server

```bash
cd projects/gpu_optimizer
python -m gpu_optimizer.api.main
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Memory Tracing

**POST** `/api/trace/memory`

Trace memory usage with layer-level attribution and fragmentation analysis.

**Request Body:**
```json
{
  "model_name": "my_model",
  "input_shape": [8, 3, 224, 224],
  "batch_size": 8,
  "max_events": 10000,
  "enable_fragmentation": true
}
```

**Response:**
```json
{
  "model_name": "my_model",
  "total_events": 1523,
  "total_layers": 12,
  "peak_allocated_mb": 245.67,
  "peak_reserved_mb": 512.00,
  "avg_fragmentation": 0.15,
  "tracing_time_ms": 45.23,
  "layer_stats": {
    "conv1": {
      "layer_name": "conv1",
      "peak_allocated": 12582912,
      "peak_reserved": 16777216,
      "total_allocations": 5,
      "total_deallocations": 3,
      "avg_fragmentation": 0.12,
      "duration_ms": 2.34
    }
  }
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/trace/memory" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "resnet50",
    "input_shape": [4, 3, 224, 224],
    "batch_size": 4,
    "max_events": 5000
  }'
```

### 2. Export Flame Graph

**POST** `/api/trace/export`

Export a trace as flame graph (speedscope) or timeline format.

**Request Body:**
```json
{
  "trace_id": "trace_1705536000000",
  "format": "speedscope"
}
```

**Response:**
```json
{
  "trace_id": "trace_1705536000000",
  "format": "speedscope",
  "download_url": "/api/trace/download/trace_1705536000000_flamegraph.json",
  "file_size_bytes": 45678
}
```

**Supported Formats:**
- `speedscope` - Flame graph format for https://www.speedscope.app/
- `timeline` - Raw timeline JSON with all events

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/trace/export" \
  -H "Content-Type: application/json" \
  -d '{
    "trace_id": "trace_1705536000000",
    "format": "speedscope"
  }'
```

### 3. List Active Traces

**GET** `/api/trace/list`

Get a list of all active traces stored in memory.

**Response:**
```json
{
  "active_traces": [
    "trace_1705536000000",
    "trace_1705536100000"
  ],
  "count": 2
}
```

**cURL Example:**
```bash
curl "http://localhost:8000/api/trace/list"
```

### 4. Delete Trace

**DELETE** `/api/trace/{trace_id}`

Delete a trace from memory to free up resources.

**Response:**
```json
{
  "message": "Trace trace_1705536000000 deleted successfully",
  "remaining_traces": 1
}
```

**cURL Example:**
```bash
curl -X DELETE "http://localhost:8000/api/trace/trace_1705536000000"
```

## Python Client Example

```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

# 1. Trace memory usage
trace_request = {
    "model_name": "my_transformer",
    "input_shape": [4, 128, 512],
    "batch_size": 4,
    "max_events": 10000,
    "enable_fragmentation": True
}

response = requests.post(f"{BASE_URL}/api/trace/memory", json=trace_request)
trace_data = response.json()

print(f"Peak Memory: {trace_data['peak_allocated_mb']:.2f} MB")
print(f"Fragmentation: {trace_data['avg_fragmentation']:.1%}")
print(f"Total Layers: {trace_data['total_layers']}")

# 2. List active traces
response = requests.get(f"{BASE_URL}/api/trace/list")
traces = response.json()
print(f"Active traces: {traces['active_traces']}")

# 3. Export as flame graph
if traces['active_traces']:
    trace_id = traces['active_traces'][0]
    
    export_request = {
        "trace_id": trace_id,
        "format": "speedscope"
    }
    
    response = requests.post(f"{BASE_URL}/api/trace/export", json=export_request)
    export_data = response.json()
    
    print(f"Exported to: {export_data['download_url']}")
    print(f"File size: {export_data['file_size_bytes']} bytes")

# 4. Clean up - delete trace
response = requests.delete(f"{BASE_URL}/api/trace/{trace_id}")
print(response.json()['message'])
```

## JavaScript/TypeScript Client Example

```typescript
const BASE_URL = 'http://localhost:8000';

// 1. Trace memory usage
async function traceMemory() {
  const response = await fetch(`${BASE_URL}/api/trace/memory`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_name: 'my_model',
      input_shape: [8, 3, 224, 224],
      batch_size: 8,
      max_events: 10000,
      enable_fragmentation: true
    })
  });
  
  const data = await response.json();
  console.log(`Peak Memory: ${data.peak_allocated_mb.toFixed(2)} MB`);
  console.log(`Layers: ${data.total_layers}`);
  
  return data;
}

// 2. Export flame graph
async function exportFlameGraph(traceId: string) {
  const response = await fetch(`${BASE_URL}/api/trace/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      trace_id: traceId,
      format: 'speedscope'
    })
  });
  
  const data = await response.json();
  console.log(`Download URL: ${data.download_url}`);
  
  return data;
}

// 3. List and clean up traces
async function cleanupTraces() {
  const listResponse = await fetch(`${BASE_URL}/api/trace/list`);
  const { active_traces } = await listResponse.json();
  
  for (const traceId of active_traces) {
    await fetch(`${BASE_URL}/api/trace/${traceId}`, { method: 'DELETE' });
    console.log(`Deleted trace: ${traceId}`);
  }
}
```

## Complete Workflow Example

### 1. Start the API Server

```bash
cd projects/gpu_optimizer
uvicorn gpu_optimizer.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Check API Health

```bash
curl http://localhost:8000/api/health
```

### 3. Trace Your Model

```python
import requests

response = requests.post("http://localhost:8000/api/trace/memory", json={
    "model_name": "resnet50",
    "input_shape": [16, 3, 224, 224],
    "batch_size": 16,
    "max_events": 20000,
    "enable_fragmentation": True
})

trace_result = response.json()
print(f"Peak Memory: {trace_result['peak_allocated_mb']:.2f} MB")
print(f"Fragmentation: {trace_result['avg_fragmentation']:.1%}")

# Print top 5 memory-intensive layers
layer_stats = trace_result['layer_stats']
sorted_layers = sorted(
    layer_stats.items(),
    key=lambda x: x[1]['peak_allocated'],
    reverse=True
)[:5]

print("\nTop 5 Memory-Intensive Layers:")
for name, stats in sorted_layers:
    peak_mb = stats['peak_allocated'] / (1024**2)
    print(f"  {name}: {peak_mb:.2f} MB")
```

### 4. Export and Visualize

```python
# Get trace ID
traces = requests.get("http://localhost:8000/api/trace/list").json()
trace_id = traces['active_traces'][0]

# Export as flame graph
export_response = requests.post("http://localhost:8000/api/trace/export", json={
    "trace_id": trace_id,
    "format": "speedscope"
})

export_data = export_response.json()
print(f"Flame graph exported: {export_data['download_url']}")
```

### 5. View in Speedscope

1. Download the exported JSON file
2. Visit https://www.speedscope.app/
3. Drag and drop the JSON file
4. Explore memory usage over time

## API Documentation

Interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

## Error Handling

The API returns standard HTTP status codes:

- `200` - Success
- `404` - Resource not found (e.g., trace_id doesn't exist)
- `422` - Validation error (invalid request body)
- `500` - Internal server error
- `503` - Service unavailable (CUDA not available)

**Example Error Response:**
```json
{
  "detail": "Trace trace_invalid not found"
}
```

## Performance Considerations

### Memory Usage

Each active trace stores events in memory. The memory usage per trace is:

```
memory_per_trace = max_events * 120 bytes
```

For example:
- `max_events=10000` → ~1.2 MB per trace
- `max_events=100000` → ~12 MB per trace

### Overhead

Memory tracing has <1% overhead on model execution. For best performance:

1. Use appropriate `max_events` based on your needs
2. Delete traces after exporting to free memory
3. Disable fragmentation tracking if not needed

### Concurrent Requests

The API can handle multiple concurrent tracing requests. Each request creates a new tracer instance.

## Integration with Existing Tools

### With PyTorch Profiler

```python
import torch.profiler

# Use both profilers together
with torch.profiler.profile() as torch_prof:
    response = requests.post("http://localhost:8000/api/trace/memory", json={
        "model_name": "my_model",
        "input_shape": [4, 10],
        "batch_size": 4
    })

# Compare results
print(torch_prof.key_averages())
print(response.json())
```

### With TensorBoard

Export timeline data and visualize alongside TensorBoard metrics:

```python
# Export timeline
export_response = requests.post("http://localhost:8000/api/trace/export", json={
    "trace_id": trace_id,
    "format": "timeline"
})

# Load and process timeline data
import json
with open('timeline.json') as f:
    timeline = json.load(f)
    
# Add to TensorBoard or custom visualization
```

## Troubleshooting

### CUDA Not Available

```json
{
  "detail": "CUDA not available"
}
```

**Solution**: Ensure you're running on a GPU-enabled system with CUDA installed.

### Memory Tracer Not Available

```json
{
  "detail": "Memory tracer not available"
}
```

**Solution**: Restart the API server. The tracer is initialized on startup.

### Trace Not Found

```json
{
  "detail": "Trace trace_123 not found"
}
```

**Solution**: Check active traces with `/api/trace/list` before exporting or deleting.

## Best Practices

1. **Clean up traces**: Delete traces after exporting to free memory
2. **Use appropriate max_events**: Balance detail vs memory usage
3. **Monitor API health**: Check `/api/health` regularly
4. **Export important traces**: Save flame graphs for later analysis
5. **Use batch operations**: Trace multiple models in sequence

## Production Deployment

For production deployment:

```bash
# Install production server
pip install gunicorn

# Run with multiple workers
gunicorn gpu_optimizer.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## Security Considerations

- Add authentication middleware for production
- Limit max_events to prevent memory exhaustion
- Implement rate limiting for tracing endpoints
- Secure file downloads with proper access controls
- Use HTTPS in production

## Next Steps

- Explore the interactive API docs at `/api/docs`
- Run the test suite: `pytest tests/test_api_integration.py`
- Check out example notebooks in `examples/`
- Read the full MemoryTracer documentation in `MEMORY_TRACER_README.md`
