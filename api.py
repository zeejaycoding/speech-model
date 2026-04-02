"""
Flask REST API for speech assessment with Supabase backend
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import logging
from datetime import datetime
from pathlib import Path

from config import Config, config
from db_service import DatabaseService
from assessment_service import SpeechAssessmentService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Ensure upload folder exists
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Initialize Supabase REST API (no direct PostgreSQL connection needed!)
try:
    logger.info("✓ Initializing Supabase REST API...")
    logger.info("✓ IPv4 compatible - no direct database connection needed!")
except Exception as e:
    logger.error(f"Failed to initialize: {e}")
    raise

# Global services - initialized at startup to avoid timeout
logger.info("🔄 Loading speech assessment service at startup...")
assessment_service = SpeechAssessmentService(
    whisper_model=app.config['WHISPER_MODEL']
)
logger.info("✓ Speech assessment service loaded")

db_service = DatabaseService()
logger.info("✓ Database service initialized")

def get_assessment_service():
    """Get assessment service (already initialized at startup)"""
    return assessment_service

def get_db_service():
    """Get database service (already initialized at startup)"""
    return db_service

@app.before_request
def before_request():
    """Services already initialized at startup"""
    pass

def allowed_file(filename):
    """Check if file has allowed audio format"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_AUDIO_FORMATS']

# ============ Health & Status ============

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'whisper_model': app.config['WHISPER_MODEL'],
        'database': 'Supabase REST API (IPv4 compatible)'
    }), 200

# ============ Assessment Endpoints ============

@app.route('/api/assess', methods=['POST'])
def assess_pronunciation():
    """
    Assess a child's pronunciation
    
    Request (multipart/form-data):
        - audio_file: audio file (WAV, MP3, etc)
        - expected_word: word child was supposed to say
        - child_id: UUID of the child
        - user_id: (optional) parent's user ID
        - session_id: (optional) UUID of assessment session
        
    Returns:
        Assessment results with scores and saved status
    """
    try:
        # Validate request
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio_file']
        expected_word = request.form.get('expected_word', '').strip()
        child_id = request.form.get('child_id', '').strip()
        user_id = request.form.get('user_id', '').strip()
        session_id = request.form.get('session_id', '').strip() or None
        
        if not expected_word:
            return jsonify({'error': 'Expected word not provided'}), 400
        
        if not child_id:
            return jsonify({'error': 'Child ID is required'}), 400
        
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        if not allowed_file(audio_file.filename):
            return jsonify({'error': f'File type not allowed. Allowed: {app.config["ALLOWED_AUDIO_FORMATS"]}'}), 400
        
        # Save audio file
        filename = secure_filename(audio_file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        
        logger.info(f"Audio file saved: {filepath}")
        
        # Step 1: Run assessment (speech-to-text + scoring)
        logger.info(f"Assessing pronunciation for child {child_id}...")
        service = get_assessment_service()

        # Fetch child DOB for adaptive thresholds (non-fatal if unavailable)
        child_dob = None
        try:
            child_record = get_db_service().get_child(child_id)
            if child_record:
                child_dob = child_record.get('date_of_birth')
        except Exception:
            pass

        assessment_result = service.assess_pronunciation(
            filepath,
            expected_word,
            child_date_of_birth=child_dob,
        )
        
        # Step 2: Save to database if successful
        if 'error' not in assessment_result:
            try:
                db_service = get_db_service()
                attempt_data = {
                    'expected_word': assessment_result['expected_word'],
                    'recognized_text': assessment_result['recognized_text'],
                    'audio_file_path': filepath,
                    'scores': assessment_result['scores'],
                    'speech_confidence': assessment_result['speech_confidence'],
                    'audio': assessment_result['audio']
                }
                
                saved_attempt = db_service.save_attempt(child_id, attempt_data)
                assessment_result['saved'] = True
                assessment_result['attempt_id'] = saved_attempt['id'] if saved_attempt else None
                logger.info(f"✓ Assessment saved for child {child_id}")
                
            except Exception as e:
                logger.error(f"Warning: Assessment completed but failed to save to database: {e}")
                assessment_result['saved'] = False
                assessment_result['save_error'] = str(e)
        
        return jsonify(assessment_result), 200
        
    except Exception as e:
        logger.error(f"Error in assess_pronunciation: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'error_type': type(e).__name__}), 500

@app.route('/api/batch-assess', methods=['POST'])
def batch_assess():
    """
    Assess multiple words in a session
    
    Request (multipart/form-data):
        - child_id: UUID of the child
        - user_id: (optional) parent's user ID
        - session_id: (optional) UUID of assessment session
        - words: list of expected words
        - audio_files: list of audio files
        
    Returns:
        List of assessment results
    """
    try:
        child_id = request.form.get('child_id', '').strip()
        user_id = request.form.get('user_id', '').strip()
        session_id = request.form.get('session_id', '').strip() or None
        words = request.form.getlist('words')
        
        if not child_id:
            return jsonify({'error': 'child_id is required'}), 400
        
        if 'audio_files' not in request.files:
            return jsonify({'error': 'No audio files provided'}), 400
        
        audio_files = request.files.getlist('audio_files')
        
        if len(words) != len(audio_files):
            return jsonify({'error': 'Number of words must match number of audio files'}), 400
        
        # Save files and prepare pairs
        word_audio_pairs = []
        for word, audio_file in zip(words, audio_files):
            if not allowed_file(audio_file.filename):
                continue
            
            filename = secure_filename(audio_file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = timestamp + filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio_file.save(filepath)
            
            word_audio_pairs.append((word, filepath))
        
        # Run batch assessment
        service = get_assessment_service()
        assessment_results = service.batch_process_session(child_id, session_id, word_audio_pairs)
        
        # Save all results to database
        saved_count = 0
        db_service = get_db_service()
        
        for result in assessment_results:
            try:
                if 'error' not in result:
                    attempt_data = {
                        'expected_word': result['expected_word'],
                        'recognized_text': result['recognized_text'],
                        'audio_file_path': result['audio']['file_path'],
                        'scores': result['scores'],
                        'speech_confidence': result['speech_confidence'],
                        'audio': result['audio']
                    }
                    saved_attempt = db_service.save_attempt(child_id, attempt_data)
                    result['saved'] = True
                    result['attempt_id'] = saved_attempt['id'] if saved_attempt else None
                    saved_count += 1
            except Exception as e:
                logger.error(f"Failed to save attempt: {e}")
                result['saved'] = False
                result['save_error'] = str(e)
        
        return jsonify({
            'session_id': session_id,
            'child_id': child_id,
            'total_assessments': len(assessment_results),
            'saved_count': saved_count,
            'assessments': assessment_results,
            'session_average': sum(r.get('scores', {}).get('final_score', 0) for r in assessment_results) / len(assessment_results) if assessment_results else 0
        }), 200
        
    except Exception as e:
        logger.error(f"Error in batch_assess: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'error_type': type(e).__name__}), 500

# ============ Child Management ============

@app.route('/api/users/<int:user_id>/children', methods=['GET'])
def list_user_children(user_id):
    """Get all children for a user"""
    try:
        db_service = get_db_service()
        children = db_service.get_user_children(user_id)
        
        return jsonify({
            'user_id': user_id,
            'count': len(children),
            'children': children
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing children: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/children/<child_id>', methods=['GET'])
def get_child(child_id):
    """Get child details with latest scores"""
    try:
        db_service = get_db_service()
        child = db_service.get_child(child_id)
        
        if not child:
            return jsonify({'error': 'Child not found'}), 404
        
        return jsonify(child), 200
        
    except Exception as e:
        logger.error(f"Error getting child: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/children/<child_id>/scores', methods=['GET'])
def get_child_scores(child_id):
    """Get all scores for a child"""
    try:
        db_service = get_db_service()
        scores = db_service.get_child_scores(child_id)
        
        return jsonify({
            'child_id': child_id,
            'count': len(scores),
            'scores': scores
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting child scores: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/children/<child_id>/latest-score', methods=['GET'])
def get_child_latest_score(child_id):
    """Get the latest score for a child"""
    try:
        db_service = get_db_service()
        score = db_service.get_child_latest_score(child_id)
        
        if not score:
            return jsonify({'error': 'No scores available'}), 404
        
        return jsonify(score), 200
        
    except Exception as e:
        logger.error(f"Error getting latest score: {e}")
        return jsonify({'error': str(e)}), 500

# ============ Progress Tracking ============

@app.route('/api/children/<child_id>/progress', methods=['GET'])
def get_progress(child_id):
    """Get child's overall progress metrics"""
    try:
        db_service = get_db_service()
        progress = db_service.get_child_progress(child_id)
        
        if not progress:
            return jsonify({'error': 'No assessment data available'}), 404
        
        return jsonify(progress), 200
        
    except Exception as e:
        logger.error(f"Error getting progress: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/children/<child_id>/progress-metrics', methods=['POST'])
def calculate_progress(child_id):
    """Calculate and save progress metrics for a child"""
    try:
        data = request.json
        db_service = get_db_service()
        period = data.get('period', 'weekly')  # weekly, monthly, daily
        
        metrics = db_service.calculate_progress_metrics(child_id, period=period)
        
        if not metrics:
            return jsonify({'error': 'No assessment data to calculate metrics'}), 404
        
        return jsonify({
            'child_id': child_id,
            'period': period,
            'metrics': metrics
        }), 201
        
    except Exception as e:
        logger.error(f"Error calculating progress metrics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/children/<child_id>/summary', methods=['GET'])
def get_child_summary(child_id):
    """Get comprehensive summary for a child"""
    try:
        db_service = get_db_service()
        summary = db_service.get_child_summary(child_id)
        
        if not summary:
            return jsonify({'error': 'Child not found'}), 404
        
        return jsonify(summary), 200
        
    except Exception as e:
        logger.error(f"Error getting child summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/children/<child_id>/export', methods=['GET'])
def export_child_scores(child_id):
    """Export all scores for a child"""
    try:
        db_service = get_db_service()
        data = db_service.export_child_scores(child_id)
        
        if not data:
            return jsonify({'error': 'Child not found'}), 404
        
        return jsonify(data), 200
        
    except Exception as e:
        logger.error(f"Error exporting scores: {e}")
        return jsonify({'error': str(e)}), 500

# ============ Error Handlers ============

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# ============ Startup Initialization ============

def initialize_services():
    """Load services at startup (Whisper model + Supabase client)"""
    logger.info("=" * 60)
    logger.info("INITIALIZING SERVICES AT STARTUP...")
    logger.info("=" * 60)
    
    try:
        logger.info("1. Loading Whisper model (this may take 1-2 minutes)...")
        service = get_assessment_service()
        logger.info(f"✓ Whisper {app.config['WHISPER_MODEL']} model loaded successfully!")
    except Exception as e:
        logger.error(f"✗ Failed to load Whisper model: {e}")
        raise
    
    try:
        logger.info("2. Initializing Supabase database service...")
        db = get_db_service()
        logger.info("✓ Supabase REST API client initialized successfully!")
    except Exception as e:
        logger.error(f"✗ Failed to initialize database service: {e}")
        raise
    
    logger.info("=" * 60)
    logger.info("✓ ALL SERVICES READY - API IS OPERATIONAL")
    logger.info("=" * 60)

if __name__ == '__main__':
    # Initialize all services before starting the server
    initialize_services()
    
    logger.info("Starting Flask API server...")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG'],
        use_reloader=False  # Disable auto-reloader to prevent crashes during audio processing
    )
