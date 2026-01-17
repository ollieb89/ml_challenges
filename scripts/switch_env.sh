```bash
#!/bin/bash
# scripts/switch_env.sh

ENV_NAME=${1:-cuda}  # Default to cuda

if [ "$ENV_NAME" = "cuda" ]; then
    echo "🚀 Switching to CUDA environment..."
    pixi run --environment cuda python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
elif [ "$ENV_NAME" = "cpu" ]; then
    echo "🚀 Switching to CPU environment..."
    pixi run --environment cpu python -c "import torch; print(f'PyTorch CPU: {torch.__version__}')"
else
    echo "❌ Unknown environment: $ENV_NAME"
    exit 1
fi
```
