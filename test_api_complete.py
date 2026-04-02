#!/usr/bin/env python
"""
Complete API testing - tests end-to-end flow with existing audio:
1. Load audio files from uploads folder
2. Send to API endpoints
3. Verify scores are calculated
4. Verify data is saved to database
5. Check child stats are updated
"""

import requests
import json
import sys
import time
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:5000"
UPLOAD_FOLDER = Path("uploads")

# Test UUIDs - will be prompted for real values
CHILD_ID = None  # Will be set by user input
USER_ID = None

def load_existing_audios():
    """Load existing audio files from uploads folder"""
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Loading Existing Audio Files")
    logger.info("="*60)
    
    if not UPLOAD_FOLDER.exists():
        logger.error(f"✗ Upload folder not found: {UPLOAD_FOLDER}")
        return {}
    
    audio_files = list(UPLOAD_FOLDER.glob("*.wav")) + list(UPLOAD_FOLDER.glob("*.mp3"))
    
    if not audio_files:
        logger.error(f"✗ No audio files found in {UPLOAD_FOLDER}")
        return {}
    
    audios = {}
    for audio_path in audio_files:
        audios[audio_path.name] = {
            "path": audio_path,
            "label": f"Audio: {audio_path.name}"
        }
        logger.info(f"✓ Found audio: {audio_path.name} ({audio_path.stat().st_size / 1024:.1f} KB)")
    
    logger.info(f"\nTotal audio files: {len(audios)}")
    return audios

def test_health_check():
    """Test API is running"""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Health Check")
    logger.info("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=10)
        if response.status_code == 200:
            logger.info(f"✓ API is healthy")
            data = response.json()
            logger.info(f"  - Status: {data.get('status')}")
            logger.info(f"  - Whisper model: {data.get('whisper_model')}")
            logger.info(f"  - Database: {data.get('database')}")
            return True
        else:
            logger.error(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"✗ Could not connect to API: {e}")
        logger.error("Make sure to run: python api.py")
        return False

def test_single_assessment(audio_path, expected_word):
    """Test single word assessment"""
    logger.info(f"\n  Testing: '{expected_word}' with audio: {audio_path.name}")
    
    try:
        with open(audio_path, 'rb') as f:
            files = {
                'audio_file': (audio_path.name, f, 'audio/wav')
            }
            data = {
                'expected_word': expected_word,
                'child_id': CHILD_ID,
            }
            if USER_ID:
                data['user_id'] = USER_ID
            
            logger.info(f"    Sending to API (timeout: 300s for first run)...")
            response = requests.post(
                f"{API_BASE_URL}/api/assess",
                files=files,
                data=data,
              #  timeout=300  # 5 minutes for first run when Whisper loads
            )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"    ✓ Assessment successful!")
            logger.info(f"      - Recognized: '{result.get('recognized_text', 'N/A')}'")
            logger.info(f"      - Phoneme score: {result['scores']['phoneme_score']:.1f}")
            logger.info(f"      - Word score: {result['scores']['word_score']:.1f}")
            logger.info(f"      - Final score: {result['scores']['final_score']:.1f}")
            logger.info(f"      - Saved to DB: {result.get('saved', False)}")
            if result.get('attempt_id'):
                logger.info(f"      - Attempt ID: {result['attempt_id']}")
            return result
        else:
            logger.error(f"    ✗ Failed with status {response.status_code}")
            try:
                logger.error(f"      Response: {response.json()}")
            except:
                logger.error(f"      Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        logger.error(f"    ✗ Timeout (API taking too long)")
        logger.error(f"      This can happen on first run when Whisper model loads")
        logger.error(f"      First request can take 2-3 minutes. Try running again!")
        return None
    except Exception as e:
        logger.error(f"    ✗ Error: {e}")
        return None

def run_assessment_tests(test_audios):
    """Run assessment tests with existing audio files"""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Assessing Pronunciations")
    logger.info("="*60)
    
    results = {}
    all_scores = []
    
    for i, (filename, audio_info) in enumerate(test_audios.items(), 1):
        logger.info(f"\n[{i}/{len(test_audios)}] Processing: {filename}")
        
        # Ask user for expected word for each audio
        expected_word = input(f"  Expected word for '{filename}'? (default: 'hello'): ").strip() or "hello"
        
        result = test_single_assessment(audio_info["path"], expected_word)
        if result and "error" not in result:
            result["audio_label"] = audio_info["label"]
            result["expected_word_input"] = expected_word
            results[filename] = result
            all_scores.append(result['scores']['final_score'])
        
        time.sleep(0.5)  # Rate limit
    
    # Summary
    logger.info("\n" + "-"*60)
    logger.info("ASSESSMENT SUMMARY:")
    logger.info("-"*60)
    
    logger.info(f"\nSuccessfully assessed: {len(results)} / {len(test_audios)}")
    
    if all_scores:
        logger.info(f"Scores: Average: {np.mean(all_scores):.1f}, Min: {min(all_scores):.1f}, Max: {max(all_scores):.1f}")
        for filename, result in results.items():
            score = result['scores']['final_score']
            recognized = result['recognized_text']
            logger.info(f"  - {filename}: '{recognized}' → Score: {score:.1f}")
    else:
        logger.warning("No assessments succeeded")
    
    return results

def verify_database_saved(results):
    """Verify data was saved to database by checking child record"""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Verifying Database Saves")
    logger.info("="*60)
    
    try:
        logger.info(f"\nFetching child record: {CHILD_ID}")
        response = requests.get(f"{API_BASE_URL}/api/children/{CHILD_ID}", timeout=10)
        
        if response.status_code == 200:
            child = response.json()
            logger.info(f"✓ Retrieved child record: {child.get('name', 'Unknown')}")
            logger.info(f"  - Total attempts: {child.get('total_attempts', 0)}")
            logger.info(f"  - Average score: {child.get('average_score', 0)}")
            logger.info(f"  - Last score: {child.get('last_score', 0)}")
            logger.info(f"  - Last assessed: {child.get('last_assessed_date', 'N/A')}")
            
            saved = child.get('total_attempts', 0)
            expected = len(results)
            
            if saved >= expected:
                logger.info(f"✓ Database saved {saved} attempts")
                return True
            else:
                logger.warning(f"⚠ Expected {expected} saves but got {saved}")
                return False
        elif response.status_code == 404:
            logger.error(f"✗ Child not found: {CHILD_ID}")
            logger.error("   Make sure you entered a valid child UUID from your database")
            return False
        else:
            logger.error(f"✗ HTTP {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        return False

def verify_scores_endpoint(results):
    """Verify we can retrieve saved scores"""
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Retrieving Saved Scores")
    logger.info("="*60)
    
    try:
        logger.info(f"\nFetching scores for child: {CHILD_ID}")
        response = requests.get(f"{API_BASE_URL}/api/children/{CHILD_ID}/scores", timeout=10)
        
        if response.status_code == 200:
            scores_data = response.json()
            score_count = scores_data.get('count', 0)
            
            logger.info(f"✓ Retrieved {score_count} saved scores")
            
            if score_count > 0:
                scores = scores_data.get('scores', [])
                logger.info(f"\nLast 3 attempts:")
                for i, score in enumerate(scores[-3:], 1):
                    logger.info(f"  {i}. Expected: '{score.get('expected_word')}' → "
                               f"Recognized: '{score.get('recognized_text')}' "
                               f"(Score: {score.get('final_score', 0):.1f})")
                return True
            return score_count > 0
        elif response.status_code == 404:
            logger.warning(f"⚠ Child not found or no scores: {response.status_code}")
            return False
        else:
            logger.warning(f"⚠ HTTP {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        return False

def get_progress_metrics():
    """Get progress metrics for the child"""
    logger.info("\n" + "="*60)
    logger.info("STEP 6: Progress Metrics")
    logger.info("="*60)
    
    try:
        logger.info(f"\nCalculating progress for child: {CHILD_ID}")
        response = requests.get(f"{API_BASE_URL}/api/children/{CHILD_ID}/progress", timeout=10)
        
        if response.status_code == 200:
            progress = response.json()
            logger.info(f"✓ Progress metrics calculated")
            logger.info(f"  - Total attempts: {progress.get('total_attempts', 0)}")
            logger.info(f"  - Average score: {progress.get('average_score', 0):.1f}")
            logger.info(f"  - Best score: {progress.get('best_score', 0):.1f}")
            logger.info(f"  - Worst score: {progress.get('worst_score', 0):.1f}")
            logger.info(f"  - Improvement trend: {progress.get('improvement_trend', 0):.1f}")
            return True
        elif response.status_code == 404:
            logger.warning(f"⚠ No progress data found")
            return False
        else:
            logger.warning(f"⚠ HTTP {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        return False

def main():
    """Run complete test suite"""
    global CHILD_ID, USER_ID
    
    logger.info("\n")
    logger.info("╔" + "="*58 + "╗")
    logger.info("║" + " "*58 + "║")
    logger.info("║" + "  SPEECH ASSESSMENT API - TEST WITH EXISTING AUDIO".center(58) + "║")
    logger.info("║" + " "*58 + "║")
    logger.info("╚" + "="*58 + "╝")
    
    # Get real child ID from user
    logger.info("\n" + "="*60)
    logger.info("SETUP: Get Child ID from Supabase")
    logger.info("="*60)
    logger.info("\nYou need a real child UUID from your Supabase database:")
    logger.info("  1. Go to: https://app.supabase.com")
    logger.info("  2. Select your project")
    logger.info("  3. Go to 'children' table")
    logger.info("  4. Copy a UUID from the 'id' column\n")
    
    CHILD_ID = input("Enter child UUID: ").strip()
    if not CHILD_ID:
        logger.error("✗ Child UUID is required!")
        sys.exit(1)
    
    USER_ID_input = input("Enter parent/user UUID (optional, press Enter to skip): ").strip()
    if USER_ID_input:
        USER_ID = USER_ID_input
    
    logger.info(f"✓ Using child_id: {CHILD_ID}")
    if USER_ID:
        logger.info(f"✓ Using user_id: {USER_ID}")
    
    # Step 1: Load existing audio
    test_audios = load_existing_audios()
    if not test_audios:
        logger.error("\n✗ No audio files to test")
        sys.exit(1)
    
    # Step 2: Check API is running
    if not test_health_check():
        logger.error("\n✗ Cannot proceed - API is not running")
        logger.error("Start the API in another terminal with: python api.py")
        sys.exit(1)
    
    # Step 3: Run assessments
    results = run_assessment_tests(test_audios)
    
    if len(results) == 0:
        logger.warning("\n⚠ No assessments succeeded - check errors above")
        sys.exit(1)
    
    # Step 4-6: Verify database and retrieve data
    time.sleep(1)  # Wait for database writes
    
    db_saved = verify_database_saved(results)
    scores_retrieved = verify_scores_endpoint(results)
    metrics_retrieved = get_progress_metrics()
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("FINAL TEST SUMMARY")
    logger.info("="*60)
    
    all_passed = db_saved and scores_retrieved and metrics_retrieved
    
    if all_passed:
        logger.info("\n✓✓✓ ALL TESTS PASSED! ✓✓✓")
        logger.info("  ✓ Audio assessments processed")
        logger.info("  ✓ Scores saved to database")
        logger.info("  ✓ Child stats updated")
        logger.info("  ✓ Data retrievable via API")
        logger.info("  ✓ Progress metrics calculated")
        logger.info("\n✓ SYSTEM IS WORKING CORRECTLY!")
    else:
        logger.warning("\n⚠ SOME TESTS FAILED:")
        logger.warning(f"  - Database saved: {'✓' if db_saved else '✗'}")
        logger.warning(f"  - Scores retrieved: {'✓' if scores_retrieved else '✗'}")
        logger.warning(f"  - Metrics retrieved: {'✓' if metrics_retrieved else '✗'}")
    
    logger.info("\n" + "="*60)
    logger.info("✓ Uploaded audio files saved to: uploads/")
    logger.info("="*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n[Interrupted by user]")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\u2717 Unexpected error: {e}")
        sys.exit(1)
