"""
Supabase Database Service Layer (REST API Version)
Handles database operations via Supabase REST API instead of direct PostgreSQL connection
This works on IPv4 networks without needing direct database access
"""
import logging
from supabase import create_client, Client
from datetime import datetime, timedelta
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for database operations using Supabase REST API"""
    
    def __init__(self):
        """Initialize service without opening the Supabase client immediately."""
        self.supabase: Client | None = None
        self._init_error: Exception | None = None

    def _get_supabase(self) -> Client:
        """Create the Supabase client on first use."""
        if self.supabase is not None:
            return self.supabase

        if self._init_error is not None:
            raise RuntimeError(f"Supabase client unavailable: {self._init_error}")

        try:
            if not Config.SUPABASE_URL or not Config.SUPABASE_KEY:
                raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

            self.supabase = create_client(
                Config.SUPABASE_URL,
                Config.SUPABASE_KEY,
            )
            logger.info("✓ Supabase REST API client initialized successfully")
            return self.supabase
        except Exception as e:
            self._init_error = e
            logger.error(f"✗ Error initializing Supabase client: {e}")
            raise RuntimeError(f"Supabase client unavailable: {e}") from e
    
    # ============ Child Operations ============
    
    def get_child(self, child_id):
        """Get child by ID"""
        try:
            supabase = self._get_supabase()
            response = supabase.table('children').select('*').eq('id', child_id).execute()
            if response.data:
                return response.data[0]
            logger.warning(f"⚠ Child not found: {child_id}")
            return None
        except Exception as e:
            logger.error(f"✗ Error getting child: {e}")
            return None
    
    def get_user_children(self, user_id):
        """Get all children for a user"""
        try:
            supabase = self._get_supabase()
            response = supabase.table('children').select('*').eq('user_id', user_id).execute()
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"✗ Error getting user children: {e}")
            return []
    
    def update_child(self, child_id, **kwargs):
        """Update child profile"""
        try:
            kwargs['updated_at'] = datetime.utcnow().isoformat()
            supabase = self._get_supabase()
            response = supabase.table('children').update(kwargs).eq('id', child_id).execute()
            logger.info(f"✓ Updated child: {child_id}")
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"✗ Error updating child: {e}")
            raise
    
    
    # ============ Speech Attempt Operations ============
    
    def get_child_scores(self, child_id):
        """Get all scores/attempts for a child, sorted by timestamp"""
        try:
            supabase = self._get_supabase()
            response = supabase.table('speech_attempts').select('*').eq('child_id', child_id).order('attempt_timestamp', desc=False).execute()
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"✗ Error getting child scores: {e}")
            return []
    
    def get_child_latest_score(self, child_id):
        """Get the most recent score for a child"""
        try:
            supabase = self._get_supabase()
            response = supabase.table('speech_attempts').select('*').eq('child_id', child_id).order('attempt_timestamp', desc=True).limit(1).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"✗ Error getting latest score: {e}")
            return None
    
    def save_attempt(self, child_id, attempt_data):
        """Save speech attempt to database and update child stats"""
        try:
            supabase = self._get_supabase()
            attempt = {
                'child_id': child_id,
                'session_id': None,  # Optional: can be added later if sessions are used
                'expected_word': attempt_data['expected_word'],
                'recognized_text': attempt_data['recognized_text'],
                'audio_file_path': attempt_data.get('audio_file_path'),
                'phoneme_score': attempt_data['scores']['phoneme_score'],
                'word_score': attempt_data['scores']['word_score'],
                'final_score': attempt_data['scores']['final_score'],
                'speech_confidence': attempt_data['speech_confidence'],
                'duration_seconds': attempt_data.get('audio', {}).get('duration_seconds'),
                'attempt_timestamp': datetime.utcnow().isoformat()
            }
            
            response = supabase.table('speech_attempts').insert(attempt).execute()
            logger.info(f"✓ Saved attempt {response.data[0]['id']} for child {child_id}")
            
            # Update child summary stats
            self._update_child_stats(child_id)
            
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"✗ Error saving attempt: {e}")
            raise
    
    def _update_child_stats(self, child_id):
        """Update child's summary statistics after new attempt (if supported)"""
        try:
            # Get all attempts for this child
            attempts = self.get_child_scores(child_id)
            
            if not attempts:
                return
            
            scores = [a['final_score'] for a in attempts]
            total_attempts = len(attempts)
            average_score = sum(scores) / total_attempts
            last_score = scores[-1]

            # Summary stats are derived from speech_attempts; the children table
            # does not have summary columns, so we just log them here.
            logger.info(f"✓ Child {child_id} stats: avg={average_score:.2f}, total={total_attempts}, last={last_score:.2f}")
                
        except Exception as e:
            logger.debug(f"⚠ Error calculating child stats (non-critical): {e}")
    
    # ============ Progress Tracking ============
    
    def get_child_progress(self, child_id):
        """Get overall progress metrics for a child"""
        try:
            attempts = self.get_child_scores(child_id)
            
            if not attempts:
                return None
            
            scores = [a['final_score'] for a in attempts]
            total_attempts = len(attempts)
            average_score = sum(scores) / total_attempts
            
            # Trend analysis (first half vs second half)
            mid_point = total_attempts // 2
            if mid_point > 0:
                first_half_avg = sum(scores[:mid_point]) / mid_point
                second_half_avg = sum(scores[mid_point:]) / (total_attempts - mid_point)
                improvement = second_half_avg - first_half_avg
            else:
                improvement = 0.0
            
            child = self.get_child(child_id)
            
            return {
                'child_id': child_id,
                'child_name': child['name'] if child else 'Unknown',
                'total_attempts': total_attempts,
                'average_score': round(average_score, 2),
                'improvement_trend': round(improvement, 2),
                'latest_score': round(scores[-1], 2),
                'best_score': round(max(scores), 2),
                'worst_score': round(min(scores), 2),
                'latest_attempt': attempts[-1]['attempt_timestamp']
            }
        except Exception as e:
            logger.error(f"✗ Error getting progress: {e}")
            return None
    
    def calculate_progress_metrics(self, child_id, period='weekly'):
        """Calculate and save progress metrics for a child"""
        try:
            attempts = self.get_child_scores(child_id)
            
            if not attempts:
                return None
            
            # Group by time period
            if period == 'weekly':
                grouped = self._group_by_week(attempts)
            elif period == 'monthly':
                grouped = self._group_by_month(attempts)
            else:
                grouped = self._group_by_day(attempts)
            
            # Calculate metrics for each period
            metrics_list = []
            for period_key, period_attempts in sorted(grouped.items())[-1:]:  # Get latest period only
                scores = [a['final_score'] for a in period_attempts]
                
                metric = {
                    'child_id': child_id,
                    'metric_date': datetime.utcnow().isoformat(),
                    'period': period,
                    'total_attempts': len(period_attempts),
                    'average_score': round(sum(scores) / len(scores), 2),
                    'best_score': round(max(scores), 2),
                    'worst_score': round(min(scores), 2),
                    'improvement': 0.0,
                    'most_difficult_phoneme': None,
                    'most_accurate_phoneme': None,
                    'created_at': datetime.utcnow().isoformat()
                }
                
                # Calculate improvement from previous period if exists
                if len(grouped) > 1:
                    prev_scores = list(grouped.items())[-2][1]
                    prev_avg = sum([a['final_score'] for a in prev_scores]) / len(prev_scores)
                    metric['improvement'] = round(metric['average_score'] - prev_avg, 2)
                
                metrics_list.append(metric)
            
            if metrics_list:
                supabase = self._get_supabase()
                response = supabase.table('progress_metrics').insert(metrics_list).execute()
                logger.info(f"✓ Saved progress metrics for child {child_id}")
                return response.data if response.data else None
            
            return None
        except Exception as e:
            logger.error(f"✗ Error calculating progress metrics: {e}")
            return None
    
    def _group_by_week(self, attempts):
        """Group attempts by week"""
        grouped = {}
        for attempt in attempts:
            attempt_date = datetime.fromisoformat(attempt['attempt_timestamp'])
            week_key = attempt_date.isocalendar()[1]
            if week_key not in grouped:
                grouped[week_key] = []
            grouped[week_key].append(attempt)
        return grouped
    
    def _group_by_month(self, attempts):
        """Group attempts by month"""
        grouped = {}
        for attempt in attempts:
            attempt_date = datetime.fromisoformat(attempt['attempt_timestamp'])
            month_key = attempt_date.strftime('%Y-%m')
            if month_key not in grouped:
                grouped[month_key] = []
            grouped[month_key].append(attempt)
        return grouped
    
    def _group_by_day(self, attempts):
        """Group attempts by day"""
        grouped = {}
        for attempt in attempts:
            attempt_date = datetime.fromisoformat(attempt['attempt_timestamp'])
            day_key = attempt_date.strftime('%Y-%m-%d')
            if day_key not in grouped:
                grouped[day_key] = []
            grouped[day_key].append(attempt)
        return grouped
    
# ============ Utility Methods ============
    
    def get_child_summary(self, child_id):
        """Get comprehensive summary for a child"""
        try:
            child = self.get_child(child_id)
            if not child:
                return None
            
            scores = self.get_child_scores(child_id)
            progress = self.get_child_progress(child_id)
            
            return {
                'child': child,
                'total_attempts': len(scores),
                'progress': progress,
                'latest_scores': scores[-5:] if scores else []  # Last 5 attempts
            }
        except Exception as e:
            logger.error(f"✗ Error getting child summary: {e}")
            return None
    
    def export_child_scores(self, child_id):
        """Export all scores for a child (for analytics/reports)"""
        try:
            child = self.get_child(child_id)
            if not child:
                return None
            
            scores = self.get_child_scores(child_id)
            
            return {
                'child': child,
                'total_attempts': len(scores),
                'scores': scores,
                'export_timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"✗ Error exporting scores: {e}")
            return None
