"""
Main assessment service orchestrating the speech assessment pipeline
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from speech_to_text import SpeechRecognizer
from pronunciation_scoring import PronunciationScorer
from utils import get_audio_duration, get_audio_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechAssessmentService:
    """Handles complete speech assessment workflow"""
    
    def __init__(self, whisper_model='medium'):
        """
        Initialize assessment service
        
        Args:
            whisper_model: Whisper model size to use
        """
        self.speech_recognizer = SpeechRecognizer(model_size=whisper_model)
        self.scorer = PronunciationScorer(
            phoneme_weight=0.95,
            word_weight=0.00,
            prosody_weight=0.05,
        )
        self.upload_folder = 'uploads'
        self._ensure_upload_folder()
    
    def _ensure_upload_folder(self):
        """Ensure upload folder exists"""
        Path(self.upload_folder).mkdir(exist_ok=True)
    
    def assess_pronunciation(self, audio_path, expected_word, child_id=None,
                             session_id=None, child_date_of_birth=None):
        """
        Complete assessment: transcribe audio and score pronunciation.
        
        Args:
            audio_path:            path to recorded audio file
            expected_word:         word the child was supposed to say
            child_id:              ID of the child (optional, for database)
            session_id:            ID of the assessment session (optional)
            child_date_of_birth:   ISO date string 'YYYY-MM-DD' (for adaptive
                                   thresholds; pass from DB child record)
            
        Returns:
            dict with complete assessment results
        """
        logger.info(f"Starting assessment for word '{expected_word}'")
        
        try:
            # Step 1: Speech-to-text
            logger.info(f"Step 1: Transcribing audio...")
            transcription = self.speech_recognizer.transcribe(audio_path)
            recognized_text   = transcription['text']
            speech_confidence = transcription['confidence']
            
            logger.info(f"Transcription result: '{recognized_text}'")
            
            # Step 2: Score pronunciation (includes prosody + non-verbal path)
            logger.info(f"Step 2: Scoring pronunciation...")
            scoring_result = self.scorer.score(
                expected_word,
                recognized_text,
                audio_path=audio_path,
                speech_confidence=speech_confidence,
                child_date_of_birth=child_date_of_birth,
            )
            
            # Step 3: Get audio features
            logger.info(f"Step 3: Extracting audio features...")
            audio_duration = get_audio_duration(audio_path)
            audio_features = get_audio_features(audio_path)
            
            # Step 4: Build result
            result = {
                'timestamp':        datetime.utcnow().isoformat(),
                'expected_word':    expected_word,
                'recognized_text':  recognized_text,
                'speech_confidence': speech_confidence,
                'scores': {
                    'phoneme_score': scoring_result['phoneme_score'],
                    'word_score':    scoring_result['word_score'],
                    'prosody_score': scoring_result.get('prosody_score', 0.0),
                    'final_score':   scoring_result['final_score'],
                    'confidence':    scoring_result['confidence'],
                },
                'status':                scoring_result['details']['status'],
                'motivational_feedback': scoring_result['details'].get('motivational_feedback', ''),
                'audio': {
                    'duration_seconds': audio_duration,
                    'features':         audio_features,
                    'file_path':        audio_path,
                },
                'details': scoring_result['details'],
            }
            
            logger.info(f"Assessment complete. Final score: {result['scores']['final_score']}")
            return result
            
        except Exception as e:
            logger.error(f"Assessment failed: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            return {
                'error':        str(e),
                'error_type':   type(e).__name__,
                'expected_word': expected_word,
                'timestamp':    datetime.utcnow().isoformat()
            }
    
    def batch_process_session(self, child_id, session_id, word_audio_pairs):
        """
        Process multiple words in a single session
        
        Args:
            child_id: ID of the child
            session_id: ID of the session (optional)
            word_audio_pairs: list of tuples [(word1, audio_path1), (word2, audio_path2), ...]
            
        Returns:
            list of assessment results for each word
        """
        results = []
        
        for expected_word, audio_path in word_audio_pairs:
            result = self.assess_pronunciation(
                audio_path, 
                expected_word, 
                child_id=child_id,
                session_id=session_id
            )
            results.append(result)
        
        return results
