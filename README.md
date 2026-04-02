# Speech Assessment System for Autism Detection

## Overview
Complete backend system for assessing child pronunciation and tracking speech development. Uses high-accuracy speech-to-text (Whisper) with phoneme-based pronunciation scoring. **Database powered by Supabase PostgreSQL.**

## Quick Start

### Prerequisites
- Python 3.8+
- Supabase account (already have it on dashboard ✓)

### 1. Get Connection String

From your **Supabase Dashboard**:
- **Project Settings** → **Database** → Copy connection string
- It looks like: `postgresql://postgres:password@host:5432/postgres`

### 2. Update .env

```
DATABASE_URL=postgresql://postgres:your-password@your-host:5432/postgres
```

### 3. Install & Run

```bash
pip install -r requirements.txt
python api.py
```

Server runs on `http://localhost:5000`

## Project Structure

```
speech-model/
├── speech_to_text.py          # Whisper-based speech recognition
├── pronunciation_scoring.py   # Phoneme & word accuracy scoring
├── utils.py                   # Audio processing utilities
├── assessment_service.py      # Main orchestration service
├── db_service.py              # Database operations service
├── models.py                  # Database models (SQLAlchemy)
├── api.py                     # Flask REST API
├── config.py                  # Configuration management
├── requirements.txt           # Dependencies
├── SUPABASE_SETUP.md          # Supabase setup guide
└── README.md                  # This file
```

## Features

✓ **Accurate Speech-to-Text**: Whisper model (no assumptions)  
✓ **Pronunciation Scoring**: Phoneme + word-level (60/40 split)  
✓ **Progress Tracking**: Track improvement over time  
✓ **PostgreSQL Database**: Supabase-powered  
✓ **REST API**: Production-ready Flask API  
✓ **Batch Processing**: Assess multiple words per session  
✓ **Data Export**: Export child data for analytics  

## API Endpoints

### Health Check
```
GET /api/health
```

### Assessment
```
POST /api/assess
- audio_file: Audio file (WAV, MP3, OGG, M4A)
- expected_word: Target word
- child_id: Child ID (optional)
- session_id: Session ID (optional)
```

Response:
```json
{
  "expected_word": "apple",
  "recognized_text": "apple",
  "scores": {
    "phoneme_score": 98.5,
    "word_score": 100.0,
    "final_score": 99.1,
    "confidence": 0.98
  },
  "status": "EXCELLENT"
}
```

### Child Management
```
POST /api/children          # Create child
GET /api/children           # List all children
GET /api/children/<id>      # Get child details
```

### Sessions
```
POST /api/sessions                      # Create session
GET /api/sessions/<id>                  # Get session details
GET /api/children/<id>/sessions         # Get child's sessions
```

### Progress Tracking
```
GET /api/progress/<child_id>            # Overall progress
GET /api/progress/<child_id>/weekly     # Weekly metrics
GET /api/children/<id>/stats            # Comprehensive stats
GET /api/children/<id>/export           # Export all data
```

### Batch Assessment
```
POST /api/batch-assess
- child_id: Child ID
- session_id: Session ID
- words: List of words
- audio_files: List of audio files
```

## Scoring System

### Score Components
1. **Phoneme Score (60%)**: Phonetic accuracy
2. **Word Score (40%)**: Character-level accuracy
3. **Final Score**: Weighted combination (0-100)

### Status
- **EXCELLENT**: Score ≥ 85
- **GOOD**: Score ≥ 70
- **FAIR**: Score ≥ 50
- **POOR**: Score ≥ 25
- **INCORRECT**: Score < 25

## Configuration

Edit `.env`:
```
DEBUG=True
WHISPER_MODEL=medium
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-key
DATABASE_URL=postgresql://...
```

**Whisper Models:**
- `tiny`: Fastest, lower accuracy
- `base`: Balanced
- `small`: Better accuracy
- `medium`: High accuracy (default)
- `large`: Highest accuracy, slowest

## Database (Supabase)

Tables created automatically:
- **children** - Child profiles
- **assessment_sessions** - Assessment sessions
- **speech_attempts** - Pronunciation attempts
- **progress_metrics** - Progress tracking

[See SUPABASE_SETUP.md for details](SUPABASE_SETUP.md)

## Usage Examples

### Python API
```python
from assessment_service import SpeechAssessmentService
from db_service import DatabaseService
from models import init_db
from sqlalchemy.orm import sessionmaker

# Initialize
engine = init_db()
Session = sessionmaker(bind=engine)
db = Session()

# Create service
service = SpeechAssessmentService(db)
db_service = DatabaseService(db)

# Assess pronunciation
result = service.assess_pronunciation(
    audio_path='recording.wav',
    expected_word='apple'
)
print(f"Score: {result['scores']['final_score']}")

# Track progress
progress = db_service.get_child_progress(child_id=1)
print(f"Average: {progress['average_score']}")
```

### cURL
```bash
# Create child
curl -X POST http://localhost:5000/api/children \
  -H "Content-Type: application/json" \
  -d '{"name": "John", "age": 6}'

# Create session
curl -X POST http://localhost:5000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"child_id": 1}'

# Assess
curl -X POST http://localhost:5000/api/assess \
  -F "audio_file=@test.wav" \
  -F "expected_word=apple" \
  -F "child_id=1" \
  -F "session_id=1"
```

## Performance

- First model load: ~2-3 seconds
- Transcription: ~1-2 seconds per audio
- Audio formats: WAV, MP3, OGG, M4A
- Max file size: 25MB
- Database: PostgreSQL (production-grade)

## Troubleshooting

### Database Connection Issues
- Check Supabase connection string in `.env`
- Verify password doesn't contain special characters
- See [SUPABASE_SETUP.md](SUPABASE_SETUP.md)

### Whisper Not Loading
- Ensure internet connection (first download)
- Check disk space (model ~1.5GB)
- Try smaller model: `WHISPER_MODEL=small`

### Audio Files Not Found
- Check file path is correct
- File must exist before transcription
- Supported: WAV, MP3, OGG, M4A

## Future Enhancements

- [ ] Real-time streaming assessment
- [ ] Acoustic signal analysis
- [ ] Stress/intonation detection
- [ ] Machine learning-based confidence
- [ ] Speech disorder detection
- [ ] Flutter/React Native mobile app

## License

Educational - FYP Autism Detection Project

## Support

- Supabase: [supabase.com/docs](https://supabase.com/docs)
- Whisper: [github.com/openai/whisper](https://github.com/openai/whisper)
- Issues: Check GitHub issues or project docs
