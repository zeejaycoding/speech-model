"""
High-accuracy speech-to-text module using OpenAI Whisper
"""
import whisper
import json
import sys
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_FALLBACK_ORDER = ['tiny', 'base', 'small', 'medium', 'large']
PRODUCTION_MODEL_CAP = ['tiny', 'base', 'small']

class SpeechRecognizer:
    
    def __init__(self, model_size='medium'):
        self.model_size = model_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        preferred_size = self.model_size

        if os.getenv('RAILWAY_ENVIRONMENT') and os.getenv('ALLOW_HEAVY_WHISPER_MODEL', 'false').lower() != 'true':
            if preferred_size not in PRODUCTION_MODEL_CAP:
                logger.warning(
                    f"Whisper model '{preferred_size}' is too heavy for Railway; capping to 'base'"
                )
                preferred_size = 'base'

        candidate_sizes = [preferred_size]

        if preferred_size in MODEL_FALLBACK_ORDER:
            preferred_index = MODEL_FALLBACK_ORDER.index(preferred_size)
            candidate_sizes.extend(reversed(MODEL_FALLBACK_ORDER[:preferred_index]))
        else:
            candidate_sizes.extend(['base', 'tiny'])

        seen = set()
        ordered_candidates = []
        for candidate in candidate_sizes:
            if candidate not in seen:
                ordered_candidates.append(candidate)
                seen.add(candidate)

        last_error = None
        for candidate_size in ordered_candidates:
            try:
                logger.info(f"Loading Whisper {candidate_size} model...")
                self.model = whisper.load_model(candidate_size, device='cpu')
                self.model_size = candidate_size
                if candidate_size != preferred_size:
                    logger.warning(
                        f"Requested Whisper model '{preferred_size}' could not be loaded; using fallback '{candidate_size}' instead"
                    )
                logger.info("Model loaded successfully")
                return
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to load Whisper {candidate_size} model: {e}")

        logger.error(f"Failed to load any Whisper model variant: {last_error}")
        raise last_error
    
    def transcribe(self, audio_path, language='en'):
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            logger.info(f"Transcribing {audio_path}...")
            logger.info(f"Audio path exists: {Path(audio_path).exists()}, Size: {Path(audio_path).stat().st_size} bytes")
            
            result = self.model.transcribe(
                audio_path,
                language=language,
                fp16=False,  # Use float32 for stability
                task='transcribe',
                temperature=0,  # Use greedy decoding (deterministic results)
                best_of=1  # Single pass (not multiple sampling attempts)
            )
            logger.info(f"✓ Transcription completed successfully")
            
            # Extract text and calculate confidence
            text = result.get('text', '').strip().lower()
            
            # Calculate average confidence from segments
            segments = result.get('segments', [])
            if segments:
                avg_confidence = sum(s.get('confidence', 0.9) for s in segments) / len(segments)
            else:
                avg_confidence = 0.85
            
            response = {
                'text': text,
                'confidence': round(avg_confidence, 3),
                'language': result.get('language', 'en'),
                'segments': [
                    {
                        'text': seg.get('text', '').strip().lower(),
                        'start': seg.get('start', 0),
                        'end': seg.get('end', 0),
                        'confidence': seg.get('confidence', 0.9)
                    }
                    for seg in segments
                ]
            }
            
            logger.info(f"Transcription complete: '{text}' (confidence: {avg_confidence})")
            return response
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

def main():
    """CLI interface for speech-to-text"""
    if len(sys.argv) < 2:
        print(json.dumps({
            'error': 'Usage: python speech_to_text.py <audio_file> [model_size]'
        }))
        sys.exit(1)
    
    audio_path = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else 'medium'
    
    try:
        recognizer = SpeechRecognizer(model_size=model_size)
        result = recognizer.transcribe(audio_path)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        sys.exit(1)

if __name__ == '__main__':
    main()