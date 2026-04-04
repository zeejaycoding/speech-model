#!/bin/bash
# Pre-download Whisper model during build to avoid runtime download timeout

echo "Pre-downloading Whisper base model..."
python -c "import whisper; whisper.load_model('base', device='cpu')" || echo "Model pre-download failed, will retry at runtime"
echo "Model pre-download complete"
