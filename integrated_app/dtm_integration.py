"""
DTM (Displacement Tracking Matrix) API Integration

This module fetches displacement and mobility data from IOM's DTM API V3
to provide emergency parameters based on internal displacement patterns.
"""

import os
import json
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# DTM API Configuration
DTM_BASE_URL = "https://dtmapi.iom.int/api"
DTM_PRIMARY_KEY = os.getenv('DTM_PRIMARY_KEY')
DTM_SECONDARY_KEY = os.getenv('DTM_SECONDARY_KEY')

# Cache settings
CACHE_DIR = "hdx_cache"  # Use same cache directory as HDX
CACHE_DURATION_DAYS = 7

def ensure_cache_directory():
    """Ensure the cache directory exists"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        logger.info(f"Created cache directory: {CACHE_DIR}")

def get_cached_data(cache_file: str) -> Optional[Dict]:
    """Get cached data if it exists and is not expired"""
    cache_path = os.path.join(CACHE_DIR, cache_file)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cached_data = json.load(f)
        
        # Check if cache is expired
        cache_time = datetime.fromisoformat(cached_data.get('cache_timestamp', '2000-01-01'))
        if datetime.now() - cache_time > timedelta(days=CACHE_DURATION_DAYS):
            logger.info(f"Cache expired for {cache_file}")
            return None
        
        logger.info(f"Using cached DTM data from {cache_file}")
        return cached_data
    except Exception as e:
        logger.error(f"Error reading cache file {cache_file}: {e}")
        return None

def save_cached_data(data: Dict, cache_file: str):
    """Save data to cache with timestamp"""
    ensure_cache_directory()
    cache_path = os.path.join(CACHE_DIR, cache_file)
    
    try:
        data['cache_timestamp'] = datetime.now().isoformat()
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Cached DTM data to {cache_file}")
    except Exception as e:
        logger.error(f"Error saving cache file {cache_file}: {e}")

def get_dtm_headers() -> Dict[str, str]:
    """Get headers for DTM API requests"""
    return {
        'Content-Type': 'application/json',
        'User-Agent': 'BusinessContinuityInsurance/1.0',
        'X-API-Key': DTM_PRIMARY_KEY  # Using primary key as main auth
    }

def fetch_sudan_displacement_data() -> Optional[Dict]:
    """
    Fetch displacement data for Sudan from DTM API
    Returns aggregated displacement statistics and trends
    """
    if not DTM_PRIMARY_KEY or not DTM_SECONDARY_KEY:
        logger.error("DTM API keys not configured")
        return None
    
    # Check cache first
    cache_file = "dtm_sudan_displacement.json"
    cached_data = get_cached_data(cache_file)
    if cached_data:
        return cached_data
    
    logger.info("Fetching Sudan displacement data from DTM API...")
    
    try:
        # First, get country list to verify Sudan is available
        countries_url = f"{DTM_BASE_URL}/Common/GetAllCountryList"
        headers = get_dtm_headers()
        
        logger.info(f"Fetching countries list from: {countries_url}")
        countries_response = requests.get(countries_url, headers=headers, timeout=10)
        
        if countries_response.status_code != 200:
            logger.error(f"Failed to fetch countries list: {countries_response.status_code}")
            return None
        
        countries_data = countries_response.json()
        logger.info(f"Retrieved {len(countries_data)} countries from DTM")
        
        # Find Sudan in the list
        sudan_found = False
        sudan_code = None
        for country in countries_data:
            # Handle both dict and string responses
            if isinstance(country, dict):
                country_name = country.get('CountryName', '').lower()
            else:
                country_name = str(country).lower()
            
            if 'sudan' in country_name:
                sudan_found = True
                sudan_code = country.get('Admin0Pcode', 'SDN') if isinstance(country, dict) else 'SDN'
                logger.info(f"Found Sudan in DTM data: {country}")
                break
        
        if not sudan_found:
            logger.warning("Sudan not found in DTM countries list")
            # Use fallback Sudan code
            sudan_code = 'SDN'
        
        # Fetch Sudan displacement data at Admin0 level
        admin0_url = f"{DTM_BASE_URL}/idpAdmin0Data/GetAdmin0Datav2"
        params = {
            'CountryName': 'Sudan',
            'Admin0Pcode': sudan_code
        }
        
        logger.info(f"Fetching Sudan Admin0 data with params: {params}")
        admin0_response = requests.get(admin0_url, headers=headers, params=params, timeout=15)
        
        if admin0_response.status_code != 200:
            logger.error(f"Failed to fetch Sudan Admin0 data: {admin0_response.status_code}")
            logger.error(f"Response text: {admin0_response.text[:500]}")
            return None
        
        admin0_data = admin0_response.json()
        logger.info(f"Retrieved Admin0 data for Sudan: {len(admin0_data)} records")
        
        # Process the displacement data
        processed_data = process_dtm_displacement_data(admin0_data)
        
        # Save to cache
        save_cached_data(processed_data, cache_file)
        
        return processed_data
        
    except requests.exceptions.Timeout:
        logger.error("DTM API request timed out")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to DTM API")
        return None
    except Exception as e:
        logger.error(f"Error fetching DTM data: {e}", exc_info=True)
        return None

def process_dtm_displacement_data(raw_data: List[Dict]) -> Dict:
    """
    Process raw DTM displacement data into emergency parameters
    """
    if not raw_data:
        logger.warning("No DTM data to process")
        return create_fallback_dtm_parameters()
    
    try:
        # Analyze displacement trends and patterns
        total_idps = 0
        total_returnees = 0
        assessment_dates = []
        displacement_trends = []
        
        for record in raw_data:
            # Extract IDP figures
            idp_individuals = record.get('IdpIndividuals', 0) or 0
            returnee_individuals = record.get('ReturneeIndividuals', 0) or 0
            
            total_idps += idp_individuals
            total_returnees += returnee_individuals
            
            # Track assessment dates for temporal analysis
            assessment_date = record.get('AssessmentDate')
            if assessment_date:
                assessment_dates.append(assessment_date)
                displacement_trends.append({
                    'date': assessment_date,
                    'idps': idp_individuals,
                    'returnees': returnee_individuals
                })
        
        # Calculate emergency probability based on displacement scale
        # Higher displacement = higher emergency probability
        if total_idps > 0:
            # Normalize based on Sudan's population (~45M)
            displacement_rate = min(total_idps / 45000000, 0.2)  # Cap at 20%
            base_emergency_prob = 0.1 + (displacement_rate * 2)  # Range: 0.1-0.5
        else:
            base_emergency_prob = 0.15  # Default moderate level
        
        # Calculate monthly variation based on temporal patterns
        monthly_factors = calculate_displacement_seasonality(displacement_trends)
        
        # Emergency impact varies by organization type
        emergency_impact = {
            'NGO': 0.12,      # Higher impact on NGOs due to resource constraints
            'UN Agency': 0.08, # Lower impact due to better resources
            'Hybrid': 0.10    # Medium impact
        }
        
        # Emergency probability modifiers based on displacement context
        emergency_modifiers = {
            'NGO': 1.3,       # NGOs more affected by displacement crises
            'UN Agency': 0.8, # UN agencies more prepared
            'Hybrid': 1.1     # Moderate impact
        }
        
        processed_data = {
            'emergency_probability': base_emergency_prob,
            'monthly_risk_factors': monthly_factors,
            'emergency_probability_modifiers': emergency_modifiers,
            'emergency_impact': emergency_impact,
            'displacement_statistics': {
                'total_idps': total_idps,
                'total_returnees': total_returnees,
                'assessment_count': len(raw_data),
                'latest_assessment': max(assessment_dates) if assessment_dates else None
            },
            'data_source': 'DTM API V3 - Displacement Data',
            'data_source_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'raw_dtm_data': raw_data[:10]  # Store first 10 records for transparency
        }
        
        logger.info(f"Processed DTM data: {total_idps:,} IDPs, {total_returnees:,} returnees")
        logger.info(f"Emergency probability: {base_emergency_prob:.3f}")
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error processing DTM data: {e}", exc_info=True)
        return create_fallback_dtm_parameters()

def calculate_displacement_seasonality(trends: List[Dict]) -> Dict[str, float]:
    """
    Calculate monthly displacement patterns
    Returns seasonal factors for emergency probability
    """
    if not trends:
        # Default seasonal pattern based on typical humanitarian cycles
        return {
            'Jan': 0.9,  'Feb': 0.8,  'Mar': 1.0,  'Apr': 1.2,
            'May': 1.3,  'Jun': 1.4,  'Jul': 1.3,  'Aug': 1.2,
            'Sep': 1.1,  'Oct': 1.0,  'Nov': 0.9,  'Dec': 0.8
        }
    
    try:
        # Analyze trends by month if we have temporal data
        monthly_displacement = {month: [] for month in 
                              ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']}
        
        for trend in trends:
            try:
                # Parse date and extract month
                date_str = trend['date']
                if isinstance(date_str, str):
                    date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    month_name = date_obj.strftime('%b')
                    displacement_level = trend.get('idps', 0)
                    monthly_displacement[month_name].append(displacement_level)
            except Exception as e:
                continue
        
        # Calculate average displacement by month and normalize
        monthly_averages = {}
        for month, values in monthly_displacement.items():
            if values:
                monthly_averages[month] = sum(values) / len(values)
            else:
                monthly_averages[month] = 0
        
        # Normalize to create risk factors (0.5 to 1.5 range)
        if max(monthly_averages.values()) > 0:
            max_displacement = max(monthly_averages.values())
            return {month: 0.5 + (avg / max_displacement) 
                   for month, avg in monthly_averages.items()}
        else:
            # Fallback to default pattern
            return {
                'Jan': 0.9,  'Feb': 0.8,  'Mar': 1.0,  'Apr': 1.2,
                'May': 1.3,  'Jun': 1.4,  'Jul': 1.3,  'Aug': 1.2,
                'Sep': 1.1,  'Oct': 1.0,  'Nov': 0.9,  'Dec': 0.8
            }
    except Exception as e:
        logger.error(f"Error calculating seasonality: {e}")
        return {
            'Jan': 1.0,  'Feb': 1.0,  'Mar': 1.0,  'Apr': 1.0,
            'May': 1.0,  'Jun': 1.0,  'Jul': 1.0,  'Aug': 1.0,
            'Sep': 1.0,  'Oct': 1.0,  'Nov': 1.0,  'Dec': 1.0
        }

def create_fallback_dtm_parameters() -> Dict:
    """
    Create fallback DTM parameters when API is unavailable
    """
    logger.info("Using fallback DTM displacement parameters")
    
    return {
        'emergency_probability': 0.18,  # Moderate-high for Sudan context
        'monthly_risk_factors': {
            'Jan': 0.8,  'Feb': 0.7,  'Mar': 0.9,  'Apr': 1.1,
            'May': 1.4,  'Jun': 1.5,  'Jul': 1.4,  'Aug': 1.3,
            'Sep': 1.2,  'Oct': 1.0,  'Nov': 0.9,  'Dec': 0.8
        },
        'emergency_probability_modifiers': {
            'NGO': 1.3,
            'UN Agency': 0.8,
            'Hybrid': 1.1
        },
        'emergency_impact': {
            'NGO': 0.12,
            'UN Agency': 0.08,
            'Hybrid': 0.10
        },
        'displacement_statistics': {
            'total_idps': 0,
            'total_returnees': 0,
            'assessment_count': 0,
            'latest_assessment': None
        },
        'data_source': 'DTM Fallback Parameters',
        'data_source_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'fallback_reason': 'DTM API unavailable or no data'
    }

def get_dtm_emergency_parameters() -> Dict:
    """
    Main function to get DTM emergency parameters
    """
    logger.info("Fetching DTM emergency parameters...")
    
    try:
        dtm_data = fetch_sudan_displacement_data()
        if dtm_data:
            logger.info("Successfully retrieved DTM displacement parameters")
            return dtm_data
        else:
            logger.warning("DTM API failed, using fallback parameters")
            return create_fallback_dtm_parameters()
    except Exception as e:
        logger.error(f"Error in DTM parameter generation: {e}", exc_info=True)
        return create_fallback_dtm_parameters()

# Test function
def test_dtm_api():
    """Test function to verify DTM API connectivity"""
    print("Testing DTM API connectivity...")
    
    if not DTM_PRIMARY_KEY:
        print("❌ DTM_PRIMARY_KEY not found in environment")
        return False
    
    if not DTM_SECONDARY_KEY:
        print("❌ DTM_SECONDARY_KEY not found in environment")
        return False
    
    print("✅ API keys found")
    
    try:
        params = get_dtm_emergency_parameters()
        print(f"✅ DTM parameters retrieved successfully")
        print(f"   Emergency probability: {params.get('emergency_probability', 'N/A')}")
        print(f"   Data source: {params.get('data_source', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Error testing DTM API: {e}")
        return False

if __name__ == "__main__":
    test_dtm_api()