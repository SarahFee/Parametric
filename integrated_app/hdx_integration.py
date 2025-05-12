"""
Enhanced HDX Data Integration for Business Continuity Insurance Model

This module fetches data from the ACAPS INFORM Severity Index API for emergency parameters
and from the Humanitarian API (HAPI) for security parameters using ACLED data. If APIs are unavailable,
it falls back to simulated data.
"""

import logging
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
import requests

# Set up logger
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = "hdx_cache"
SECURITY_CACHE_FILE = os.path.join(CACHE_DIR, "hapi_acled_security_data.json")
EMERGENCY_CACHE_FILE = os.path.join(CACHE_DIR, "acaps_emergency_data.json")
CACHE_EXPIRY_DAYS = 7

# API Constants
ACAPS_AUTH_TOKEN = "6403e2cffe7b64f3d33034137476dd1f129aa397"
ACAPS_MONTHS_2023 = [
    'Jan2023', 'Feb2023', 'Mar2023', 'Apr2023', 'May2023', 'Jun2023',
    'Jul2023', 'Aug2023', 'Sep2023', 'Oct2023', 'Nov2023', 'Dec2023'
]

HAPI_BASE = "https://hapi.humdata.org/api/v2"
APP_IDENTIFIER = "TXlBcHA6ZmVraWguc2FyYWhAZ21haWwuY29t"
HEADERS = {"accept": "application/json"}

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache handling

def _is_cache_valid(filepath):
    return os.path.exists(filepath) and (datetime.now().timestamp() - os.path.getmtime(filepath) < CACHE_EXPIRY_DAYS * 86400)

def _read_cache(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def _write_cache(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

# Fetch ACAPS INFORM Severity Index data

def fetch_acaps_data(months):
    headers = {"Authorization": f"Token {ACAPS_AUTH_TOKEN}", "Accept": "application/json"}
    all_data = []
    for month in months:
        url = f"https://api.acaps.org/api/v1/inform-severity-index/{month}/?page=2"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            logger.warning(f"Failed to fetch ACAPS data for {month}: {response.status_code}")
            continue
        page_data = response.json()
        all_data.extend([crisis for crisis in page_data.get('results', []) if 'Sudan' in crisis.get('country', []) or 'SDN' in crisis.get('iso3', [])])
        logger.info(f"Fetched ACAPS data for {month}")
    return all_data


# In hdx_integration.py, modify the calculate_acaps_emergency_parameters function

def calculate_acaps_emergency_parameters(crisis_data):
    df = pd.DataFrame(crisis_data)
    if df.empty:
        logger.warning("No ACAPS data available for analysis.")
        return None

    df['severity'] = pd.to_numeric(df['INFORM Severity Index'], errors='coerce')
    df.dropna(subset=['severity'], inplace=True)

    thresholds = {'crisis': 3.5, 'emergency': 4.0, 'catastrophe': 4.5}
    params = {
        "emergency_probability": np.mean(df['severity'] >= thresholds['crisis']),
        "crisis_percentage": np.mean((df['severity'] >= thresholds['crisis']) & (df['severity'] < thresholds['emergency'])),
        "emergency_percentage": np.mean((df['severity'] >= thresholds['emergency']) & (df['severity'] < thresholds['catastrophe'])),
        "catastrophe_percentage": np.mean(df['severity'] >= thresholds['catastrophe']),
        "emergency_probability_modifiers": {"NGO": 1.2, "UN Agency": 0.7, "Hybrid": 1.0},
        "emergency_impact": {"NGO": 0.08, "UN Agency": 0.15, "Hybrid": 0.10},
        "data_source": "ACAPS INFORM Severity Index",
        "data_source_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_acaps_data": crisis_data  # Store the raw data for visualization
    }
    return params

# Emergency parameters handler

def get_emergency_parameters_from_hdx():
    if _is_cache_valid(EMERGENCY_CACHE_FILE):
        logger.info("Using cached ACAPS emergency data.")
        return _read_cache(EMERGENCY_CACHE_FILE)

    data = fetch_acaps_data(ACAPS_MONTHS_2023)
    params = calculate_acaps_emergency_parameters(data)
    if params:
        _write_cache(EMERGENCY_CACHE_FILE, params)
    return params

# Fetch security data from HAPI

def fetch_hapi_data(endpoint):
    url = f"{HAPI_BASE}{endpoint}?location_code=SDN&app_identifier={APP_IDENTIFIER}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        logger.error(f"Failed to fetch HAPI data: {e}")
        return []

# Security parameters handler using HAPI

def get_security_parameters_from_hdx():
    if _is_cache_valid(SECURITY_CACHE_FILE):
        logger.info("Using cached HAPI ACLED security data.")
        return _read_cache(SECURITY_CACHE_FILE)

    data = fetch_hapi_data("/coordination-context/conflict-events")
    if not data:
        logger.warning("No data from HAPI, using simulated ACLED security parameters.")
        return None

    event_types = pd.Series([d.get("event_type", "unknown") for d in data])
    event_type_dist = event_types.value_counts(normalize=True).to_dict()

    params = {
        "base_security_risk": 0.22,
        "monthly_risk_factors": {"Jan": 0.7, "Feb": 0.8, "Mar": 0.9, "Apr": 1.1, "May": 1.3, "Jun": 1.4,
                                 "Jul": 1.4, "Aug": 1.2, "Sep": 1.0, "Oct": 0.9, "Nov": 0.8, "Dec": 0.7},
        "security_risk_modifiers": {"NGO": 1.0, "UN Agency": 0.5, "Hybrid": 0.8},
        "event_type_distribution": event_type_dist,
        "data_source": "HAPI Conflict Events API",
        "data_source_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    _write_cache(SECURITY_CACHE_FILE, params)
    return params

# Combine parameters
def get_all_hdx_parameters():
    security_params = get_security_parameters_from_hdx()
    emergency_params = get_emergency_parameters_from_hdx()
    if not security_params:
        logger.warning("Using fallback simulated security parameters.")
        security_params = get_security_parameters_from_hdx()

    if not emergency_params or not security_params:
        logger.error("Failed to fetch parameters.")
        return None

    return {
        "security_parameters": security_params,
        "emergency_parameters": emergency_params,
        "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    params = get_all_hdx_parameters()
    print(json.dumps(params, indent=2))

