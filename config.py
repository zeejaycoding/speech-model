import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'tiny')
    
    # Supabase Credentials
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    
    # Note: No DATABASE_URL needed - using REST API instead
    # This works on IPv4 networks without direct PostgreSQL connection
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    
    DEBUG = os.getenv('DEBUG', False)
    UPLOAD_FOLDER = 'uploads'
    MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25MB
    ALLOWED_AUDIO_FORMATS = {'wav', 'mp3', 'ogg', 'm4a'}
    
    # Scoring thresholds
    EXCELLENT_SCORE = 85
    GOOD_SCORE = 70
    FAIR_SCORE = 50
    
    # Phoneme similarity weight (vs word accuracy)
    PHONEME_WEIGHT = 0.6
    WORD_WEIGHT = 0.4

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
