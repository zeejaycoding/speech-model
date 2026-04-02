"""
Pronunciation and word accuracy scoring module
Compares expected word vs recognized speech with high accuracy
"""
import sys
import json
import logging
from datetime import date
from utils import (
    get_phonemes, calculate_edit_distance, get_audio_duration,
    score_phonemes_child_aware, score_prosody, score_nonverbal_attempt,
)
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ARPAbet vowel bases (stress digit stripped before lookup)
_VOWEL_BASES = {
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY',
    'EH', 'ER', 'EY', 'IH', 'IY',
    'OW', 'OY', 'UH', 'UW',
}

def _is_vowel(phoneme):
    """Return True if the ARPAbet phoneme is a vowel (stress digit optional)."""
    return phoneme.rstrip('012') in _VOWEL_BASES


# ---------------------------------------------------------------------------
# Adaptive status thresholds
# ---------------------------------------------------------------------------
# Younger children and autistic speakers need lower bars to avoid demotivation.
# age_thresholds maps child age (years) to (excellent, good, fair, poor) cutoffs.
# Children >= 8 use the adult/standard thresholds.
_AGE_THRESHOLDS = {
    # age : (EXCELLENT, GOOD, FAIR, POOR)
    3:  (50, 35, 20, 10),   # very forgiving — celebrate any attempt
    4:  (55, 40, 25, 12),
    5:  (65, 50, 35, 18),
    6:  (75, 60, 45, 22),
    7:  (80, 65, 48, 24),
    8:  (85, 70, 50, 25),   # standard / adult baseline
}

def _get_thresholds(child_age_years=None):
    if child_age_years is None:
        return _AGE_THRESHOLDS[8]
    age = max(3, min(8, int(child_age_years)))
    return _AGE_THRESHOLDS[age]

def _age_from_dob(date_of_birth):
    """Return age in years from ISO date string 'YYYY-MM-DD', or None."""
    try:
        dob = date.fromisoformat(str(date_of_birth)[:10])
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except Exception:
        return None

def _motivational_feedback(status, expected_ph, recognized_ph):
    """
    Return an encouraging, child-friendly feedback string based on the
    assessment outcome.  Detects Final Consonant Deletion specifically
    to give targeted (rather than generic) guidance.
    """
    # Detect if child got the start right but dropped the ending
    n = len(recognized_ph)
    m = len(expected_ph)
    fcd = (n > 0 and n < m and recognized_ph == expected_ph[:n])

    if status == 'EXCELLENT':
        return "Amazing job! You said that word perfectly! Keep it up!"
    elif status == 'GOOD':
        return "Great effort! You're so close — just a tiny bit more practice and you'll nail it!"
    elif status == 'FAIR':
        if fcd:
            return "Well done for starting the word! Try to finish the whole word — you can do it!"
        return "Good try! You got some sounds right. Let's practise together!"
    elif status == 'POOR':
        if fcd:
            return "You started really well! Can you try to say the ending sound too?"
        return "Nice try! Let's listen to the word together and try again — you're doing great!"
    else:  # INCORRECT
        return "That's okay — every attempt helps you learn! Let's try again together!"


class PronunciationScorer:
    """Scores pronunciation accuracy"""
    
    def __init__(self, phoneme_weight=0.60, word_weight=0.20, prosody_weight=0.20):
        """
        Initialize scorer.

        Weights must sum to 1.0.
        prosody_weight (0.20 default) captures pitch variation, rhythm, and
        volume — key autism-relevant speech dimensions not covered by phoneme
        accuracy alone.
        """
        self.phoneme_weight = phoneme_weight
        self.word_weight    = word_weight
        self.prosody_weight = prosody_weight
        
        if abs((phoneme_weight + word_weight + prosody_weight) - 1.0) > 0.01:
            raise ValueError("phoneme_weight + word_weight + prosody_weight must sum to ~1.0")
    
    def score(self, expected_word, recognized_text, audio_path=None,
              speech_confidence=1.0, child_date_of_birth=None):
        """
        Score pronunciation accuracy.

        Args:
            expected_word:         word the child was supposed to say
            recognized_text:       ASR output
            audio_path:            path to WAV file (needed for prosody)
            speech_confidence:     ASR confidence 0-1 (low → may trigger non-verbal path)
            child_date_of_birth:   ISO date string 'YYYY-MM-DD' (for adaptive thresholds)

        Returns:
            dict with phoneme_score, word_score, prosody_score, final_score, details
        """
        import string
        
        expected_word   = expected_word.strip().lower()
        recognized_text = recognized_text.strip().lower()
        child_age       = _age_from_dob(child_date_of_birth)
        thresholds      = _get_thresholds(child_age)

        logger.info(f"Scoring: Expected='{expected_word}' Recognized='{recognized_text}'")
        
        # ── Prosody (always, if audio available) ──────────────────────────
        prosody_result = None
        prosody_score  = 0.0
        if audio_path:
            prosody_result = score_prosody(audio_path)
            prosody_score  = prosody_result['prosody_score']
            logger.info(f"Prosody score: {prosody_score:.2f} "
                        f"(pitch={prosody_result['pitch_variation_score']:.1f}, "
                        f"rhythm={prosody_result['rhythm_score']:.1f}, "
                        f"vol={prosody_result['volume_score']:.1f})")

        # ── Handle empty inputs ────────────────────────────────────────────
        if not expected_word:
            logger.warning("Expected word is empty")
            return self._empty_response()
        
        # ── Non-verbal / low-confidence path ──────────────────────────────
        # If ASR produced nothing OR confidence is very low, score on audio
        # energy alone (gives credit for attempt, not accuracy).
        LOW_CONFIDENCE_THRESHOLD = 0.40
        if (not recognized_text or speech_confidence < LOW_CONFIDENCE_THRESHOLD) and audio_path:
            nv = score_nonverbal_attempt(audio_path)
            logger.info(f"Non-verbal path: vocalization={nv['vocalization_detected']}, "
                        f"attempt_score={nv['attempt_score']}")
            final_score = nv['attempt_score']   # 0-30, credit for trying
            return {
                'phoneme_score': 0.0,
                'word_score':    0.0,
                'prosody_score': round(prosody_score, 2),
                'final_score':   round(final_score, 2),
                'confidence':    round(speech_confidence, 3),
                'details': {
                    'expected':               expected_word,
                    'recognized':             recognized_text,
                    'status':                 self._get_status(final_score, thresholds),
                    'status_path':            'NON_VERBAL',
                    'vocalization_detected':  nv['vocalization_detected'],
                    'prosody':                prosody_result,
                    'child_age_years':        child_age,
                }
            }

        if not recognized_text:
            return {
                'phoneme_score': 0.0,
                'word_score':    0.0,
                'prosody_score': round(prosody_score, 2),
                'final_score':   0.0,
                'confidence':    0.0,
                'details': {
                    'expected':        expected_word,
                    'recognized':      recognized_text,
                    'status':          'NO_SPEECH_DETECTED',
                    'child_age_years': child_age,
                }
            }
        
        # ── Clean & extract scoring token ──────────────────────────────────
        recognized_text_cleaned = ''.join(
            c for c in recognized_text if c.isalpha() or c.isspace()
        ).strip()
        
        if ' ' in recognized_text_cleaned:
            words = recognized_text_cleaned.split()
            logger.info(f"Multi-word result: {words}. Using first word: '{words[0]}'")
            recognized_text_for_scoring = words[0]
        else:
            recognized_text_for_scoring = recognized_text_cleaned
        
        # ── Phoneme scoring ───────────────────────────────────────────────
        expected_phonemes    = get_phonemes(expected_word)
        recognized_phonemes  = get_phonemes(recognized_text_for_scoring)

        logger.info(f"Expected phonemes:   {expected_phonemes}")
        logger.info(f"Recognized phonemes: {recognized_phonemes}")

        phoneme_score = self._score_phonemes(expected_phonemes, recognized_phonemes, child_age)
        word_score    = self._score_words(expected_word, recognized_text_for_scoring)

        # ── Effort bonus ──────────────────────────────────────────────────
        # Reward the child for making a clear, confident attempt.
        # Capped at 10 pts so it cannot push a wrong answer into GOOD/EXCELLENT.
        effort_bonus = 0.0
        if audio_path:
            try:
                from utils import get_audio_features as _gaf
                _af = _gaf(audio_path)
                if _af and speech_confidence >= 0.7 and _af.get('rms_energy', 0) > 0.01:
                    effort_bonus = min(10.0, speech_confidence * 10.0)
                    logger.info(f"Effort bonus: {effort_bonus:.1f}")
            except Exception:
                pass

        # ── Combine (phoneme + word + prosody) + effort ───────────────────
        final_score = min(100.0, (
            phoneme_score * self.phoneme_weight +
            word_score    * self.word_weight    +
            prosody_score * self.prosody_weight +
            effort_bonus
        ))

        confidence = self._calculate_confidence(
            expected_word, recognized_text_for_scoring, final_score
        )

        status = self._get_status(final_score, thresholds)
        feedback = _motivational_feedback(status, expected_phonemes, recognized_phonemes)

        response = {
            'phoneme_score': round(phoneme_score, 2),
            'word_score':    round(word_score, 2),
            'prosody_score': round(prosody_score, 2),
            'final_score':   round(final_score, 2),
            'confidence':    round(confidence, 3),
            'details': {
                'expected':              expected_word,
                'recognized':            recognized_text,
                'recognized_processed':  recognized_text_for_scoring,
                'expected_phonemes':     expected_phonemes,
                'recognized_phonemes':   recognized_phonemes,
                'phoneme_distance':      calculate_edit_distance(
                                             expected_phonemes, recognized_phonemes),
                'word_distance':         calculate_edit_distance(
                                             list(expected_word),
                                             list(recognized_text_for_scoring)),
                'prosody':               prosody_result,
                'effort_bonus':          round(effort_bonus, 2),
                'child_age_years':       child_age,
                'status':                status,
                'motivational_feedback': feedback,
            }
        }

        logger.info(
            f"Scores - Phoneme: {phoneme_score:.2f}, Word: {word_score:.2f}, "
            f"Prosody: {prosody_score:.2f}, Effort: {effort_bonus:.1f}, Final: {final_score:.2f}"
        )
        return response
    
    def _score_phonemes(self, expected_ph, recognized_ph, child_age=None):
        """
        Score based on phoneme sequence accuracy.

        Uses a child-aware edit distance that gives half-penalty for
        developmentally expected substitutions (e.g. 'w' for 'r') and
        applies Final Consonant Deletion forgiveness.
        Also adds a consonant-skeleton component so vowel imprecision
        (common in young/autistic children) does not dominate.
        """
        if not expected_ph:
            return 0.0
        if not recognized_ph:
            return 0.0

        # Child-aware full-sequence score (with FCD forgiveness)
        full_score = score_phonemes_child_aware(expected_ph, recognized_ph, child_age)

        # Consonant-skeleton score (strip vowels)
        exp_cons = [ph for ph in expected_ph  if not _is_vowel(ph)]
        rec_cons = [ph for ph in recognized_ph if not _is_vowel(ph)]
        if exp_cons and rec_cons:
            cons_score = score_phonemes_child_aware(exp_cons, rec_cons, child_age)
        else:
            cons_score = full_score

        # Full sequence dominant; consonant skeleton provides partial floor
        return full_score * 0.75 + cons_score * 0.25
    
    def _score_words(self, expected, recognized):
        """Score based on character-level word accuracy"""
        if not expected:
            return 0.0
        if not recognized:
            return 0.0
        
        # Levenshtein distance for character-level accuracy
        distance = calculate_edit_distance(list(expected), list(recognized))
        max_len = max(len(expected), len(recognized))
        word_accuracy = max(0, (1 - distance / max_len)) * 100
        
        # Bonus: exact match
        if expected == recognized:
            return 100.0
        
        return word_accuracy
    
    def _calculate_confidence(self, expected, recognized, final_score):
        """
        Calculate confidence metric
        Based on: score, similarity, and recognized text being recognizable
        """
        # Base confidence from score
        score_confidence = final_score / 100.0
        
        # Similarity ratio (how much of recognized matches expected)
        matcher = SequenceMatcher(None, expected, recognized)
        similarity_ratio = matcher.ratio()
        
        # Combined confidence
        confidence = (score_confidence * 0.7) + (similarity_ratio * 0.3)
        return min(1.0, max(0.0, confidence))
    
    def _get_status(self, final_score, thresholds=None):
        """Get assessment status using adaptive thresholds."""
        exc, good, fair, poor = thresholds or _get_thresholds()
        if final_score >= exc:
            return 'EXCELLENT'
        elif final_score >= good:
            return 'GOOD'
        elif final_score >= fair:
            return 'FAIR'
        elif final_score >= poor:
            return 'POOR'
        else:
            return 'INCORRECT'
    
    def _empty_response(self):
        """Return empty/error response"""
        return {
            'phoneme_score': 0.0,
            'word_score': 0.0,
            'final_score': 0.0,
            'confidence': 0.0,
            'details': {'status': 'ERROR'}
        }

def main():
    """CLI interface for pronunciation scoring"""
    if len(sys.argv) < 3:
        print(json.dumps({
            'error': 'Usage: python pronunciation_scoring.py <expected_word> <recognized_text>'
        }))
        sys.exit(1)
    
    expected = sys.argv[1]
    recognized = sys.argv[2]
    
    try:
        scorer = PronunciationScorer()
        result = scorer.score(expected, recognized)
        print(json.dumps(result))
    except Exception as e:
        logger.error(f"Error: {e}")
        print(json.dumps({'error': str(e)}))
        sys.exit(1)

if __name__ == '__main__':
    main()