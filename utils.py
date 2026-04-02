import pronouncing
import numpy as np
import librosa
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_DIGRAPH_TO_PHONEME = {
    'sh': 'SH', 'ch': 'CH', 'th': 'DH', 'ph': 'F',
    'wh': 'W',  'ng': 'NG', 'ck': 'K',  'qu': 'K',
}
_LETTER_TO_PHONEME = {
    'b': 'B',   'c': 'K',   'd': 'D',   'f': 'F',   'g': 'G',
    'h': 'HH',  'j': 'JH',  'k': 'K',   'l': 'L',   'm': 'M',
    'n': 'N',   'p': 'P',   'r': 'R',   's': 'S',   't': 'T',
    'v': 'V',   'w': 'W',   'x': 'K',   'y': 'Y',   'z': 'Z',
    'a': 'AH0', 'e': 'EH1', 'i': 'IH1', 'o': 'AO1', 'u': 'UH1',
}

def approximate_phonemes_from_letters(word):
    phonemes = []
    i = 0
    while i < len(word):
        two = word[i:i+2]
        if two in _DIGRAPH_TO_PHONEME:
            phonemes.append(_DIGRAPH_TO_PHONEME[two])
            i += 2
        elif word[i] in _LETTER_TO_PHONEME:
            phonemes.append(_LETTER_TO_PHONEME[word[i]])
            i += 1
        else:
            i += 1
    return phonemes


_CHILD_SUBSTITUTIONS = {
    # Liquid simplification: r/l → w/j  (very common < age 6)
    'W':  {'R', 'L'},   # "wabbit" for "rabbit", "wove" for "love"
    'R':  {'W'},
    'Y':  {'L'},        # "yeg" for "leg"
    # Stopping: fricatives → stops
    'T':  {'S', 'Z', 'TH', 'DH', 'SH'},  # "tun" for "sun"
    'D':  {'DH', 'Z', 'V'},
    'B':  {'V'},        # "berry" for "very"
    'P':  {'F'},        # "pish" for "fish"
    'G':  {'K'},        # voiced ↔ voiceless
    'K':  {'G'},
    # Fronting: velars → alveolars
    'T':  {'K', 'S', 'Z', 'TH', 'DH', 'SH'},
    'D':  {'G', 'DH', 'Z', 'V'},
    # Cluster reduction: consonant clusters simplified
    'N':  {'ND', 'NT'},
    'S':  {'ST', 'SP', 'SK'},
    # Velar nasal
    'N':  {'NG'},
}

def is_child_substitution(produced, target):
    produced_base = produced.rstrip('012')
    target_base   = target.rstrip('012')
    if produced_base == target_base:
        return True
    expected_targets = _CHILD_SUBSTITUTIONS.get(produced_base, set())
    return target_base in expected_targets


def score_phonemes_child_aware(expected_ph, recognized_ph, child_age_years=None):
    m, n = len(expected_ph), len(recognized_ph)
    if m == 0:
        return 0.0
    if n == 0:
        return 0.0

    # Weighted DP where substitutions that are expected child errors cost 0.5
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if expected_ph[i-1] == recognized_ph[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                sub_cost = 0.5 if is_child_substitution(recognized_ph[j-1], expected_ph[i-1]) else 1.0
                dp[i][j] = min(
                    dp[i-1][j]   + 1.0,      # deletion
                    dp[i][j-1]   + 1.0,      # insertion
                    dp[i-1][j-1] + sub_cost, # substitution
                )

    distance = dp[m][n]

    # ── Final Consonant Deletion forgiveness ──────────────────────────────
    # If the recognized sequence is a prefix of the expected sequence
    # (child said the start correctly but dropped the end), reduce penalty.
    is_prefix = (n < m and recognized_ph == expected_ph[:n])
    if is_prefix:
        fcd_factor = 0.5 if (child_age_years is not None and child_age_years < 5) else 0.6
        distance = max(0.0, distance * fcd_factor)

    max_len = max(m, n)
    return max(0.0, (1.0 - distance / max_len)) * 100.0


def get_phonemes(word):
    import string
    # Clean the word: lowercase, remove punctuation, strip
    word = word.strip().lower()
    # Remove punctuation and special characters
    word = ''.join(c for c in word if c.isalpha() or c.isspace())
    word = word.strip()
    
    try:
        if not word:
            logger.warning(f"Word is empty after cleaning")
            return []
        
        phones = pronouncing.phones_for_word(word)
        if not phones:
            # Fall back to letter-level approximation so scoring still works
            # for ASR artifacts (common with child speech)
            approx = approximate_phonemes_from_letters(word)
            if approx:
                logger.warning(
                    f"No CMU phonemes for '{word}'; using letter approximation: {approx}"
                )
            else:
                logger.warning(f"No phonemes found for word: {word}")
            return approx
        
        # Use the first (most common) pronunciation
        phoneme_string = phones[0]
        phonemes = phoneme_string.split()
        return phonemes
    except Exception as e:
        logger.error(f"Error getting phonemes for '{word}': {e}")
        return []

def calculate_edit_distance(seq1, seq2):
    m, n = len(seq1), len(seq2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )
    
    return dp[m][n]

def get_audio_duration(audio_path):
    try:
        y, sr = librosa.load(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        return round(duration, 2)
    except Exception as e:
        logger.error(f"Error getting audio duration: {e}")
        return None

def get_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path)
        
        # Basic features
        duration = librosa.get_duration(y=y, sr=sr)
        rms_energy = np.sqrt(np.mean(y ** 2))
        
        # Zero crossing rate (indicates presence of consonants)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        return {
            'duration': round(duration, 2),
            'rms_energy': round(float(rms_energy), 4),
            'zero_crossing_rate': round(float(zcr), 4),
            'mfcc_mean': [round(float(x), 4) for x in mfcc_mean]
        }
    except Exception as e:
        logger.error(f"Error extracting audio features: {e}")
        return None


def score_prosody(audio_path):
    try:
        y, sr = librosa.load(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)

        # ── Speech presence check ──────────────────────────────────────────
        rms_energy = float(np.sqrt(np.mean(y ** 2)))
        has_speech = rms_energy > 0.005   # empirical floor; silence ≈ 0.001

        if not has_speech or duration < 0.3:
            return {
                'pitch_variation_score': 0.0,
                'rhythm_score': 0.0,
                'volume_score': 0.0,
                'prosody_score': 0.0,
                'has_speech': False,
                'details': {'rms_energy': round(rms_energy, 5), 'duration': round(duration, 2)}
            }

        # ── Pitch variation (F0) ───────────────────────────────────────────
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

        if len(voiced_f0) > 1:
            # Coefficient of variation of F0 — captures pitch expressiveness
            f0_cv = float(np.std(voiced_f0) / (np.mean(voiced_f0) + 1e-9))
            # Typical natural speech CV ≈ 0.15-0.35; flat ≈ <0.05
            # Map 0→0 pts, 0.15→60 pts, 0.30→100 pts (clamp at 100)
            pitch_variation_score = min(100.0, (f0_cv / 0.30) * 100.0)
        else:
            f0_cv = 0.0
            pitch_variation_score = 0.0   # completely monotone / no voicing

        # ── Rhythm (onset regularity) ──────────────────────────────────────
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        if len(onset_frames) > 2:
            ioi = np.diff(onset_frames)           # inter-onset intervals
            rhythm_cv = float(np.std(ioi) / (np.mean(ioi) + 1e-9))
            # Lower CV = more regular rhythm → higher score
            rhythm_score = max(0.0, min(100.0, (1.0 - rhythm_cv) * 100.0))
        else:
            rhythm_cv = None
            rhythm_score = 50.0   # insufficient data → neutral

        # ── Volume stability (RMS over frames) ────────────────────────────
        rms_frames = librosa.feature.rms(y=y)[0]
        if len(rms_frames) > 1:
            vol_cv = float(np.std(rms_frames) / (np.mean(rms_frames) + 1e-9))
            # Very stable ≈ 0.3, highly erratic ≈ >1.5
            volume_score = max(0.0, min(100.0, (1.0 - vol_cv / 1.5) * 100.0))
        else:
            vol_cv = None
            volume_score = 50.0

        # ── Composite prosody score ────────────────────────────────────────
        # Pitch variation is the strongest autism marker (weight 0.5)
        prosody_score = (
            pitch_variation_score * 0.50 +
            rhythm_score          * 0.30 +
            volume_score          * 0.20
        )

        return {
            'pitch_variation_score': round(pitch_variation_score, 2),
            'rhythm_score':          round(rhythm_score, 2),
            'volume_score':          round(volume_score, 2),
            'prosody_score':         round(prosody_score, 2),
            'has_speech':            True,
            'details': {
                'f0_cv':       round(f0_cv, 4),
                'rhythm_cv':   round(rhythm_cv, 4) if rhythm_cv is not None else None,
                'vol_cv':      round(vol_cv, 4)   if vol_cv   is not None else None,
                'rms_energy':  round(rms_energy, 5),
                'duration':    round(duration, 2),
                'voiced_frames': len(voiced_f0),
            }
        }

    except Exception as e:
        logger.error(f"Error scoring prosody: {e}")
        return {
            'pitch_variation_score': 0.0,
            'rhythm_score': 0.0,
            'volume_score': 0.0,
            'prosody_score': 0.0,
            'has_speech': False,
            'details': {'error': str(e)}
        }


def score_nonverbal_attempt(audio_path):
    try:
        y, sr = librosa.load(audio_path)
        rms = float(np.sqrt(np.mean(y ** 2)))
        duration = float(librosa.get_duration(y=y, sr=sr))

        if rms < 0.005 or duration < 0.2:
            return {'attempt_score': 0.0, 'vocalization_detected': False, 'rms': round(rms, 5)}

        # Scale energy: 0.005 → 5 pts, 0.02+ → 30 pts
        attempt_score = min(30.0, max(5.0, (rms / 0.02) * 30.0))
        return {
            'attempt_score': round(attempt_score, 2),
            'vocalization_detected': True,
            'rms': round(rms, 5)
        }
    except Exception as e:
        logger.error(f"Error scoring nonverbal attempt: {e}")
        return {'attempt_score': 0.0, 'vocalization_detected': False, 'rms': 0.0}

def clean_text(text):
    text = text.strip().lower()
    # Remove extra spaces
    text = ' '.join(text.split())
    return text

def similarity_ratio(str1, str2):
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, str1, str2)
    return matcher.ratio()

def phoneme_similarity(word1, word2):
    ph1 = get_phonemes(word1)
    ph2 = get_phonemes(word2)
    
    if not ph1 or not ph2:
        return 0.0
    
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, ph1, ph2)
    return matcher.ratio()