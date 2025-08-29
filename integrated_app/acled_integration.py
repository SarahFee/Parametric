"""
ACLED API Integration Module
Fetches real-time conflict and security data directly from ACLED API
"""

import os
import json
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class ACLEDClient:
    """Client for ACLED (Armed Conflict Location & Event Data Project) API"""
    
    def __init__(self):
        self.base_url = "https://api.acleddata.com"
        self.access_token = os.getenv('ACLED_ACCESS_TOKEN')
        self.refresh_token = os.getenv('ACLED_REFRESH_TOKEN')
        self.cache_dir = "hdx_cache"
        self.cache_file = os.path.join(self.cache_dir, "acled_conflict_data.json")
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers for ACLED API"""
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid (7 days)"""
        if not os.path.exists(self.cache_file):
            return False
        
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
                cache_time = datetime.fromisoformat(cache_data.get('cache_timestamp', ''))
                return datetime.now() - cache_time < timedelta(days=7)
        except Exception:
            return False
    
    def _save_to_cache(self, data: Dict[str, Any]) -> None:
        """Save data to cache with timestamp"""
        cache_data = {
            **data,
            'cache_timestamp': datetime.now().isoformat(),
            'data_source': 'ACLED API',
            'api_endpoint': 'https://api.acleddata.com/acled/read'
        }
        
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"ACLED data cached to {self.cache_file}")
        except Exception as e:
            logger.error(f"Error saving ACLED cache: {e}")
    
    def _load_from_cache(self) -> Optional[Dict[str, Any]]:
        """Load data from cache"""
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading ACLED cache: {e}")
            return None
    
    def fetch_sudan_conflict_data(self, year: int = 2023) -> Dict[str, Any]:
        """
        Fetch conflict data for Sudan from ACLED API
        
        Args:
            year: Year to fetch data for (default: 2023)
            
        Returns:
            Dict containing conflict events and emergency parameters
        """
        # Check cache first
        if self._is_cache_valid():
            logger.info("Using cached ACLED data")
            return self._load_from_cache()
        
        if not self.access_token:
            logger.warning("ACLED access token not found, using fallback data")
            return self._get_fallback_data()
        
        try:
            # ACLED API parameters for Sudan conflict data
            params = {
                'key': self.access_token,
                'email': 'your_email@example.com',  # User needs to provide correct email
                'country': 'Sudan',
                'year': year,
                'event_type': 'Violence against civilians|Battles|Explosions/Remote violence',
                'limit': 1000
            }
            
            logger.info(f"Fetching ACLED conflict data for Sudan {year}")
            response = requests.get(
                "https://api.acleddata.com/acled/read",
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                conflict_events = data.get('data', [])
                
                # Process the conflict data into emergency parameters
                processed_data = self._process_conflict_data(conflict_events)
                
                # Cache the processed data
                self._save_to_cache(processed_data)
                
                logger.info(f"Successfully fetched {len(conflict_events)} ACLED conflict events")
                return processed_data
                
            else:
                logger.error(f"ACLED API error: {response.status_code} - {response.text}")
                return self._get_fallback_data()
                
        except Exception as e:
            logger.error(f"Error fetching ACLED data: {e}")
            return self._get_fallback_data()
    
    def _process_conflict_data(self, events: List[Dict]) -> Dict[str, Any]:
        """
        Process raw ACLED conflict events into emergency parameters
        
        Args:
            events: List of ACLED conflict events
            
        Returns:
            Dict with emergency parameters and processed events
        """
        if not events:
            return self._get_fallback_data()
        
        # Analyze conflict patterns
        monthly_events = {}
        event_types = {}
        total_fatalities = 0
        high_severity_events = 0
        
        for event in events:
            # Count events by month
            event_date = event.get('event_date', '')
            if event_date:
                try:
                    month = datetime.strptime(event_date, '%Y-%m-%d').strftime('%b')
                    monthly_events[month] = monthly_events.get(month, 0) + 1
                except ValueError:
                    continue
            
            # Count event types
            event_type = event.get('event_type', 'Unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
            
            # Sum fatalities
            fatalities = event.get('fatalities', 0)
            if isinstance(fatalities, (int, float)):
                total_fatalities += fatalities
                if fatalities >= 10:  # High severity threshold
                    high_severity_events += 1
        
        # Calculate emergency probability based on conflict intensity
        total_events = len(events)
        avg_monthly_events = total_events / 12 if total_events > 0 else 0
        
        # Base emergency probability from conflict frequency (normalized)
        base_prob = min(avg_monthly_events / 50, 0.5)  # Cap at 50%
        
        # Adjust for fatality severity
        fatality_multiplier = 1 + (high_severity_events / total_events) if total_events > 0 else 1
        emergency_probability = min(base_prob * fatality_multiplier, 0.6)  # Cap at 60%
        
        # Generate monthly risk factors based on actual event distribution
        monthly_risk_factors = {}
        max_monthly = max(monthly_events.values()) if monthly_events else 1
        
        for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
            events_in_month = monthly_events.get(month, 0)
            risk_factor = (events_in_month / max_monthly) * 1.5 + 0.5  # Scale 0.5-2.0
            monthly_risk_factors[month] = round(risk_factor, 2)
        
        return {
            'emergency_probability': round(emergency_probability, 3),
            'monthly_risk_factors': monthly_risk_factors,
            'emergency_probability_modifiers': {
                'NGO': 1.4,      # Higher risk for NGOs
                'UN Agency': 0.7, # Lower risk for UN agencies
                'Hybrid': 1.1    # Moderate risk for hybrid orgs
            },
            'emergency_impact': {
                'NGO': 0.15,     # Higher impact on NGOs
                'UN Agency': 0.08,
                'Hybrid': 0.12
            },
            'conflict_statistics': {
                'total_events': total_events,
                'total_fatalities': total_fatalities,
                'high_severity_events': high_severity_events,
                'event_types': event_types,
                'monthly_distribution': monthly_events
            },
            'raw_acled_events': events[:50],  # Store first 50 events for map visualization
            'data_source': 'ACLED API',
            'data_source_timestamp': datetime.now().isoformat()
        }
    
    def _get_fallback_data(self) -> Dict[str, Any]:
        """Return fallback emergency parameters when ACLED API is unavailable"""
        return {
            'emergency_probability': 0.25,  # Higher baseline for Sudan conflict
            'monthly_risk_factors': {
                'Jan': 0.9, 'Feb': 0.8, 'Mar': 1.1, 'Apr': 1.3,
                'May': 1.5, 'Jun': 1.6, 'Jul': 1.4, 'Aug': 1.2,
                'Sep': 1.0, 'Oct': 0.9, 'Nov': 0.8, 'Dec': 0.7
            },
            'emergency_probability_modifiers': {
                'NGO': 1.4, 'UN Agency': 0.7, 'Hybrid': 1.1
            },
            'emergency_impact': {
                'NGO': 0.15, 'UN Agency': 0.08, 'Hybrid': 0.12
            },
            'conflict_statistics': {
                'total_events': 0,
                'total_fatalities': 0,
                'high_severity_events': 0,
                'event_types': {},
                'monthly_distribution': {}
            },
            'raw_acled_events': [],
            'data_source': 'ACLED Fallback Parameters',
            'data_source_timestamp': datetime.now().isoformat(),
            'fallback_reason': 'ACLED API unavailable or authentication failed'
        }

def get_acled_emergency_parameters() -> Dict[str, Any]:
    """
    Main function to get emergency parameters from ACLED data
    
    Returns:
        Dict containing emergency parameters derived from ACLED conflict data
    """
    client = ACLEDClient()
    return client.fetch_sudan_conflict_data()

if __name__ == "__main__":
    # Test the ACLED integration
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ACLED integration...")
    data = get_acled_emergency_parameters()
    
    print(f"Emergency Probability: {data.get('emergency_probability', 'N/A')}")
    print(f"Data Source: {data.get('data_source', 'N/A')}")
    print(f"Total Events: {data.get('conflict_statistics', {}).get('total_events', 'N/A')}")