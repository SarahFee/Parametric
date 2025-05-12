"""
IATI API Module for Business Continuity Insurance Model

This module fetches organization financial data from the IATI API
or uses simulated data when the API is not available.
"""
import requests
import json
import logging
import os
import datetime
import random
import numpy as np
import pandas as pd

# Set up logger
logger = logging.getLogger(__name__)

# IATI API endpoints for our three organization types
API_ENDPOINTS = {
    "NGO": "https://d-portal.org/q.json?aid=GB-CHC-285908-219753",
    "UN Agency": "https://d-portal.org/q.json?aid=XM-DAC-41121-2022-EHGL-SDN",
    "Hybrid": "https://d-portal.org/q.json?aid=XM-DAC-47066-DP.2412"
}

# Organization name mapping
ORG_NAMES = {
    "NGO": "World-Vision",
    "UN Agency": "UNHCR",
    "Hybrid": "IOM"
}

# Define constants
CACHE_DIR = "iati_cache"
CACHE_FILE = os.path.join(CACHE_DIR, "iati_parameters.json")
CACHE_EXPIRY_DAYS = 7  # Consider making this shorter for testing

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

def _is_cache_valid(filepath):
    """Checks if the cache file exists and is not expired."""
    if not os.path.exists(filepath):
        return False
    try:
        mod_time = os.path.getmtime(filepath)
        expiry_time = mod_time + CACHE_EXPIRY_DAYS * 24 * 60 * 60
        return datetime.datetime.now().timestamp() < expiry_time
    except Exception as e:
        logger.warning(f"Could not check cache validity for {filepath}: {e}")
        return False

def _read_cache(filepath):
    """Reads data from a JSON cache file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading cache file {filepath}: {e}")
        return None

def _write_cache(filepath, data):
    """Writes data to a JSON cache file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)  # Added indent for readability
    except Exception as e:
        logger.error(f"Error writing cache file {filepath}: {e}")

def fetch_iati_data(endpoint_url):
    """
    Fetch data from IATI API endpoint with comprehensive error handling
    
    Args:
        endpoint_url (str): URL of the IATI API endpoint
    
    Returns:
        dict or None: Parsed JSON response or None if fetch fails
    """
    try:
        # Add timeout and headers to improve request reliability
        headers = {
            'User-Agent': 'Business Continuity Insurance Simulator/1.0',
            'Accept': 'application/json'
        }
        
        # Make request with timeout and custom headers
        response = requests.get(
            endpoint_url, 
            headers=headers, 
            timeout=10  # 10-second timeout
        )
        
        # Validate response
        response.raise_for_status()
        
        # Parse JSON
        data = response.json()
        
        # Additional validation
        if not data or 'xson' not in data:
            logger.warning(f"Unexpected data structure from {endpoint_url}")
            return None
        
        logger.info(f"Successfully fetched data from {endpoint_url}")
        return data
    
    except requests.exceptions.Timeout:
        logger.error(f"Timeout error fetching data from {endpoint_url}")
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error fetching data from {endpoint_url}")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching data from {endpoint_url}: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Unexpected error fetching data from {endpoint_url}: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error from {endpoint_url}: {e}")
    
    return None

def extract_parameters_from_iati(data, org_type):
    """
    Extract financial parameters from IATI data with detailed logging
    
    Args:
        data (dict): IATI API response
        org_type (str): Type of organization (NGO, UN Agency, Hybrid)
    
    Returns:
        dict: Extracted and processed parameters
    """
    # Default parameters if no data is available
    default_params = {
        "NGO": {
            'initial_capital': 3000000,
            'total_budget': 8000000,
            'operational_rate': 0.012,
            'premium_discount': 0.9,
            'claim_trigger': 0.4,
            'security_risk_modifier': 1.0,
            'emergency_probability_modifier': 1.2,
            'emergency_impact': 0.08
        },
        "UN Agency": {
            'initial_capital': 6000000,
            'total_budget': 15000000,
            'operational_rate': 0.008,
            'premium_discount': 1.0,
            'claim_trigger': 0.55,
            'security_risk_modifier': 0.5,
            'emergency_probability_modifier': 0.7,
            'emergency_impact': 0.15
        },
        "Hybrid": {
            'initial_capital': 4500000,
            'total_budget': 10000000,
            'operational_rate': 0.01,
            'premium_discount': 0.95,
            'claim_trigger': 0.5,
            'security_risk_modifier': 0.8,
            'emergency_probability_modifier': 1.0,
            'emergency_impact': 0.10
        }
    }
    
    # If no data or invalid data, return default
    if not data or 'xson' not in data or not data['xson']:
        logger.warning(f"No IATI data found for {org_type}. Using default parameters.")
        return default_params.get(org_type, {})
    
    try:
        # Get the first activity
        activity = data['xson'][0].get('/iati-activities/iati-activity', [{}])[0]
        
        # Extract transactions
        transactions = activity.get('/transaction', [])
        
        # Extract transaction amounts and dates
        transaction_details = []
        for t in transactions:
            if '/transaction-type@code' in t and '/value' in t and '/transaction-date@iso-date' in t:
                transaction_details.append({
                    'type': t['/transaction-type@code'],
                    'value': float(t['/value']),
                    'date': t['/transaction-date@iso-date']
                })
        
        # Calculate transaction statistics
        if transaction_details:
            transaction_amounts = [t['value'] for t in transaction_details]
            total_transaction = sum(transaction_amounts)
            
            # Detailed logging of transactions
            logger.info(f"Transactions for {org_type}:")
            for t in transaction_details:
                logger.info(f"  {t['date']}: ${t['value']:,.2f} (Type: {t['type']})")
            
            # Logging transaction statistics
            logger.info(f"Total Transaction Amount: ${total_transaction:,.2f}")
            logger.info(f"Number of Transactions: {len(transaction_amounts)}")
            
            # Calculate parameters based on transaction data
            def scale_parameter(default_value, transaction_value, max_multiplier=2.0):
                """Scale parameter conservatively based on transaction data"""
                # Calculate scaling factor
                scale_factor = min(
                    max_multiplier, 
                    max(1.0, (transaction_value / (default_value * 10)) ** 0.5)
                )
                
                # Log scaling details
                logger.info(f"Scaling parameter:")
                logger.info(f"  Default Value: ${default_value:,.2f}")
                logger.info(f"  Transaction Value: ${transaction_value:,.2f}")
                logger.info(f"  Scale Factor: {scale_factor:.2f}")
                
                return default_value * scale_factor
            
            # Scaled parameters
            initial_capital = scale_parameter(
                default_params[org_type]['initial_capital'], 
                total_transaction
            )
            total_budget = scale_parameter(
                default_params[org_type]['total_budget'], 
                total_transaction
            )
            
            # Calculate operational rate based on org type
            if org_type == "UN Agency":
                # More conservative rate calculation for UN Agency
                operational_rate = min(
                    default_params[org_type]['operational_rate'] * 1.1,
                    max(0.005, (total_transaction / (total_budget * 15)))
                )
            else:
                # Standard calculation for other org types
                operational_rate = min(
                    default_params[org_type]['operational_rate'] * 1.2,
                    max(0.005, (total_transaction / (total_budget * 12)))
                )
            
            # Transaction variability affects emergency probability
            transaction_variability = min(
                (np.std(transaction_amounts) / np.mean(transaction_amounts)) if transaction_amounts else 0.5,
                1.0  # Cap variability
            )
            
            # Adjust emergency probability modifier based on org type
            if org_type == "UN Agency":
                # More conservative emergency probability calculation for UN Agency
                emergency_prob_modifier = min(
                    0.9 + (transaction_variability * 0.3),
                    default_params[org_type]['emergency_probability_modifier'] * 1.2
                )
            else:
                # Standard calculation for other org types
                emergency_prob_modifier = min(
                    1.0 + (transaction_variability * 0.5),
                    default_params[org_type]['emergency_probability_modifier'] * 1.5
                )
            
            # Detailed logging of calculated parameters
            logger.info(f"Calculated Parameters for {org_type}:")
            logger.info(f"  Initial Capital: ${initial_capital:,.2f}")
            logger.info(f"  Total Budget: ${total_budget:,.2f}")
            logger.info(f"  Operational Rate: {operational_rate:.4f}")
            logger.info(f"  Transaction Variability: {transaction_variability:.4f}")
            logger.info(f"  Emergency Prob Modifier: {emergency_prob_modifier:.4f}")
            
            # Construct final parameters dictionary
            params = {
                'initial_capital': initial_capital,
                'total_budget': total_budget,
                'operational_rate': operational_rate,
                'transaction_variability': transaction_variability,
                'emergency_probability_modifier': emergency_prob_modifier
            }
        else:
            # Fallback to default parameters
            params = {k: default_params[org_type][k] for k in [
                'initial_capital', 'total_budget', 'operational_rate'
            ]}
            logger.warning(f"No valid transactions found for {org_type}. Using default parameters.")
        
        # Merge with default organization-specific parameters
        default_org_params = default_params[org_type]
        params.update({
            'premium_discount': default_org_params['premium_discount'],
            'claim_trigger': default_org_params['claim_trigger'],
            'security_risk_modifier': default_org_params['security_risk_modifier'],
            'emergency_impact': default_org_params['emergency_impact']
        })
        
        return params
    
    except Exception as e:
        logger.error(f"Error processing IATI data for {org_type}: {str(e)}")
        return default_params.get(org_type, {})

def extract_donor_data_from_iati(data, org_type):
    """Extract donor information from IATI transaction data with detailed logging"""
    donors = {}
    
    if not data or 'xson' not in data or not data['xson']:
        logger.warning(f"No IATI data found for extracting donors from {org_type}. Using default donor parameters.")
        return donors
    
    try:
        # Get the first activity
        activity = data['xson'][0].get('/iati-activities/iati-activity', [{}])[0]
        
        # Extract transactions
        transactions = activity.get('/transaction', [])
        
        # Log the total number of transactions found
        logger.info(f"Found {len(transactions)} transactions to analyze for donor data in {org_type}")
        
        # Loop through transactions to identify donors (focus on incoming funds)
        for t in transactions:
            # Only look at incoming funds (type code 1)
            if t.get('/transaction-type@code') == '1' and '/provider-org/narrative' in t:
                # Extract donor details
                donor_narratives = t['/provider-org/narrative']
                donor_name = donor_narratives[0].get('', 'Unknown Donor') if donor_narratives else 'Unknown Donor'
                donor_ref = t.get('/provider-org@ref', 'unknown')
                donor_type = t.get('/provider-org@type', 'unknown')
                transaction_amount = float(t.get('/value', 0))
                transaction_date = t.get('/transaction-date@iso-date', '')
                
                # Log the donor transaction
                logger.info(f"  Found donor transaction: {donor_name} ({donor_ref}) - ${transaction_amount:,.2f} on {transaction_date}")
                
                # Add to our donor dictionary
                if donor_ref not in donors:
                    donors[donor_ref] = {
                        'name': donor_name,
                        'type': donor_type,
                        'transactions': []
                    }
                
                # Add this transaction
                donors[donor_ref]['transactions'].append({
                    'amount': transaction_amount,
                    'date': transaction_date
                })
        
        # Calculate donation patterns for each donor
        for donor_id, donor in donors.items():
            transactions = donor['transactions']
            logger.info(f"Analyzing donor {donor['name']} ({donor_id}) with {len(transactions)} transactions")
            
            if transactions:
                # Calculate average donation and frequency
                total_amount = sum(t['amount'] for t in transactions)
                avg_amount = total_amount / len(transactions)
                donor['avg_donation'] = avg_amount
                
                # Log the average donation
                logger.info(f"  Average donation: ${avg_amount:,.2f}")
                
                # Sort transactions by date
                transactions.sort(key=lambda x: x['date'])
                
                # Calculate average time between donations if more than one
                if len(transactions) > 1:
                    # Convert dates to datetime objects
                    dates = [datetime.datetime.strptime(t['date'], '%Y-%m-%d') for t in transactions]
                    intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                    avg_interval = sum(intervals) / len(intervals)
                    donor['avg_interval_days'] = avg_interval
                    
                    # Log the average interval
                    logger.info(f"  Average donation interval: {avg_interval:.1f} days")
                else:
                    donor['avg_interval_days'] = 90  # Default quarterly
                    logger.info(f"  Only one transaction found, setting default interval to 90 days")
                
                # Add preferred org type
                donor['preferred_org_type'] = org_type
                logger.info(f"  Setting preferred organization type to {org_type}")
        
        return donors
    
    except Exception as e:
        logger.error(f"Error extracting donor data from {org_type}: {str(e)}")
        return {}

def extract_all_organization_parameters():
    """Extract parameters for all organization types with comprehensive logging"""
    org_parameters = {}
    
    for org_type, endpoint in API_ENDPOINTS.items():
        logger.info(f"Processing {org_type} from {endpoint}")
        
        # Try to fetch from API
        data = fetch_iati_data(endpoint)
        
        # Extract parameters
        params = extract_parameters_from_iati(data, org_type)
        
        # Store in our dictionary
        org_parameters[org_type] = params
    
    return org_parameters

def get_donor_definitions_from_iati():
    """Get donor definitions from IATI data for model initialization"""
    all_donors = {}
    donor_count = 0
    
    # Log start of donor extraction
    logger.info("Starting extraction of donor definitions from IATI data")
    
    for org_type, endpoint in API_ENDPOINTS.items():
        logger.info(f"Extracting donors from {org_type} data")
        
        # Fetch data
        data = fetch_iati_data(endpoint)
        
        # Extract donors
        donors = extract_donor_data_from_iati(data, org_type)
        donor_count += len(donors)
        
        # Add each donor to all_donors with organization type preferences
        for donor_id, donor in donors.items():
            # Set default preferences
            default_preferences = {
                "NGO": 0.3,
                "UN Agency": 0.3,
                "Hybrid": 0.3
            }
            
            # Increase preference for observed target type
            preferred_org_type = donor.get('preferred_org_type', org_type)
            target_preferences = default_preferences.copy()
            target_preferences[preferred_org_type] = 0.6
            
            # Adjust other preferences
            remaining_preference = 0.4
            other_types = [t for t in target_preferences.keys() if t != preferred_org_type]
            for t in other_types:
                target_preferences[t] = remaining_preference / len(other_types)
            
            # Clean donor name (use first part if too long)
            donor_name = donor['name']
            if len(donor_name) > 40:
                donor_name = donor_name.split(' ')[0] + " " + donor_name.split(' ')[1]
            
            # Create donor definition
            donor_def = {
                "name": donor_name,
                "donor_type": donor['type'],
                "avg_donation": donor.get('avg_donation', 1000000),
                "donation_frequency": donor.get('avg_interval_days', 90) / 30,  # Convert to months
                "target_preferences": target_preferences
            }
            
            # If donor already exists, update with average values
            if donor_id in all_donors:
                existing = all_donors[donor_id]
                # Average the donation amount
                existing["avg_donation"] = (existing["avg_donation"] + donor_def["avg_donation"]) / 2
                # Take the smaller frequency (more frequent donations)
                existing["donation_frequency"] = min(existing["donation_frequency"], donor_def["donation_frequency"])
                # Update preferences to reflect both
                for org_type in existing["target_preferences"]:
                    existing["target_preferences"][org_type] = (
                        existing["target_preferences"][org_type] + donor_def["target_preferences"][org_type]
                    ) / 2
            else:
                all_donors[donor_id] = donor_def
    
    # Convert to list format for the model
    donor_list = list(all_donors.values())
    
    # Log summary of extracted donors
    logger.info(f"Extracted {len(donor_list)} unique donors from {donor_count} donor references")
    for i, donor in enumerate(donor_list):
        logger.info(f"Donor {i+1}: {donor['name']}")
        logger.info(f"  Type: {donor['donor_type']}")
        logger.info(f"  Avg Donation: ${donor['avg_donation']:,.2f}")
        logger.info(f"  Donation Frequency: {donor['donation_frequency']:.2f} months")
        logger.info(f"  Target Preferences: {donor['target_preferences']}")
    
    # If no donors were found, return default ones
    if not donor_list:
        logger.warning("No donors extracted from IATI data, using defaults")
        donor_list = [
            {"name": "USAID", "donor_type": "11", "avg_donation": 1500000, "donation_frequency": 3,
             "target_preferences": {"NGO": 0.4, "UN Agency": 0.4, "Hybrid": 0.2}},
            {"name": "ECHO", "donor_type": "40", "avg_donation": 1200000, "donation_frequency": 2.5,
             "target_preferences": {"NGO": 0.3, "UN Agency": 0.5, "Hybrid": 0.2}},
            {"name": "Private Foundation", "donor_type": "60", "avg_donation": 500000, "donation_frequency": 4,
             "target_preferences": {"NGO": 0.7, "UN Agency": 0.1, "Hybrid": 0.2}}
        ]
    
    return donor_list

def get_risk_parameters_from_iati():
    """Extract risk parameters from IATI data"""
    # Initialize with defaults based on matrix
    risk_params = {
        'emergency_probability': 0.05,  # 5% base probability
        'security_risk_factor': 0.2,    # 20% base probability
        'waiting_period': 2,            # Recommended waiting period
        'payout_cap_multiple': 3.0      # Default payout cap
    }
    
    # In a real implementation, this would analyze humanitarian-scope markers
    # and other indicators of risk from the IATI data
    
    return risk_params

def calculate_simulation_duration_from_iati():
    """Calculate appropriate simulation duration from IATI activity timelines"""
    # Default duration
    sim_duration = 12
    
    try:
        # Fetch data for all orgs
        durations = []
        
        for org_type, endpoint in API_ENDPOINTS.items():
            data = fetch_iati_data(endpoint)
            if not data or 'xson' not in data or not data['xson']:
                continue
                
            activity = data['xson'][0].get('/iati-activities/iati-activity', [{}])[0]
            
            # Look for activity dates
            if '/activity-date' in activity:
                start_date = None
                end_date = None
                
                for date_entry in activity.get('/activity-date', []):
                    if date_entry.get('@type') == '2':  # Start date
                        start_date = date_entry.get('@iso-date')
                    elif date_entry.get('@type') == '3':  # End date
                        end_date = date_entry.get('@iso-date')
                
                if start_date and end_date:
                    start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
                    end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
                    duration_months = (end.year - start.year) * 12 + (end.month - start.month)
                    durations.append(duration_months)
        
        # Calculate average duration
        if durations:
            sim_duration = int(sum(durations) / len(durations))
            # Ensure it's within a reasonable range
            sim_duration = max(6, min(36, sim_duration))
    except Exception as e:
        logger.warning(f"Could not calculate simulation duration from IATI data: {str(e)}")
    
    return sim_duration

def get_iati_parameters_for_model():
    """Get all parameters needed for the model, from IATI data if possible"""
    # Check cache first
    if _is_cache_valid(CACHE_FILE):
        cached_data = _read_cache(CACHE_FILE)
        if cached_data:
            logger.info("Using cached IATI parameters.")
            return cached_data
    
    # If no cache, extract all parameters
    logger.info("Extracting all IATI parameters...")
    
    try:
        model_params = {
            'organizations': extract_all_organization_parameters(),
            'risks': get_risk_parameters_from_iati(),
            'sim_duration': calculate_simulation_duration_from_iati(),
            'fetch_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Cache the results
        _write_cache(CACHE_FILE, model_params)
        
        return model_params
    except Exception as e:
        logger.error(f"Error getting IATI parameters: {e}")
        # Return default parameters on failure
        return {
            'organizations': {
                "NGO": {'initial_capital': 3000000, 'total_budget': 8000000, 'operational_rate': 0.012,
                       'premium_discount': 0.9, 'claim_trigger': 0.4, 'security_risk_modifier': 1.0,
                       'emergency_probability_modifier': 1.2, 'emergency_impact': 0.08},
                "UN Agency": {'initial_capital': 6000000, 'total_budget': 15000000, 'operational_rate': 0.008,
                             'premium_discount': 1.0, 'claim_trigger': 0.6, 'security_risk_modifier': 0.5,
                             'emergency_probability_modifier': 0.7, 'emergency_impact': 0.15},
                "Hybrid": {'initial_capital': 4500000, 'total_budget': 10000000, 'operational_rate': 0.01,
                          'premium_discount': 0.95, 'claim_trigger': 0.5, 'security_risk_modifier': 0.8,
                          'emergency_probability_modifier': 1.0, 'emergency_impact': 0.10}
            },
            'risks': {'emergency_probability': 0.05, 'security_risk_factor': 0.2,
                     'waiting_period': 2, 'payout_cap_multiple': 3.0},
            'sim_duration': 12,
            'fetch_time': 'DEFAULT'
        }

def get_organization_definitions_from_iati():
    """Get organization definitions from IATI data for model initialization"""
    org_params = extract_all_organization_parameters()
    
    # Format organization definitions for model initialization
    org_definitions = []
    
    for org_type, params in org_params.items():
        org_def = {
            "name": ORG_NAMES[org_type],
            "org_type": org_type,
            "initial_capital": params["initial_capital"],
            "total_budget": params["total_budget"]
        }
        org_definitions.append(org_def)
    
    return org_definitions

def fetch_iati_data_with_api_key(api_key, country_code="SDN", datasets=None):
    """
    Fetch IATI data using an API key (for integration with API-based system)
    
    Args:
        api_key (str): IATI API key
        country_code (str): ISO3 country code
        datasets (list): Optional list of specific datasets to fetch
        
    Returns:
        dict: Combined IATI parameters or None on failure
    """
    logger.info(f"Fetching IATI data with API key {api_key[:4]}... for {country_code}")
    
    # In a real implementation, this would authenticate with the API
    # and fetch the requested datasets
    
    # For now, we'll just return the regular parameters
    try:
        params = get_iati_parameters_for_model()
        if params:
            params['country_code'] = country_code
            params['api_key_used'] = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
            logger.info(f"Successfully fetched IATI data for {country_code}")
            return params
        else:
            logger.error(f"Failed to get IATI parameters for {country_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching IATI data: {e}")
        return None


# For testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logger.info("--- Testing IATI API Module ---")
    
    # Test parameter extraction
    params = get_iati_parameters_for_model()
    if params:
        print("\n--- IATI Parameters ---")
        print(f"Fetched at: {params.get('fetch_time')}")
        
        print("\nSimulation Duration:", params.get('sim_duration'))
        
        print("\nOrganization Parameters:")
        for org_type, org_params in params.get('organizations', {}).items():
            print(f"\n{org_type}:")
            for key, value in org_params.items():
                print(f"  {key}: {value}")
        
        print("\nRisk Parameters:")
        for key, value in params.get('risks', {}).items():
            print(f"  {key}: {value}")
            
        # Test donor extraction
        print("\n--- Testing Donor Extraction ---")
        donors = get_donor_definitions_from_iati()
        print(f"Extracted {len(donors)} donors")
        for i, donor in enumerate(donors):
            print(f"\nDonor {i+1}: {donor['name']}")
            print(f"  Type: {donor['donor_type']}")
            print(f"  Avg Donation: ${donor['avg_donation']:,.2f}")
            print(f"  Frequency: {donor['donation_frequency']:.2f} months")
            print(f"  Target Preferences: {donor['target_preferences']}")
            
        # Test API function
        print("\n--- Testing API Function ---")
        api_data = fetch_iati_data_with_api_key("fake_api_key_12345", country_code="SDN")
        if api_data:
            print(f"API data fetched for {api_data.get('country_code')}")
            print(f"API key used: {api_data.get('api_key_used')}")
        else:
            print("Failed to fetch API data")
    else:
        print("\n--- Failed to fetch IATI parameters ---")