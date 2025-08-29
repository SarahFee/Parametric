"""
Enhanced HDX Integration with DTM Support

This module combines the original HDX integration with DTM displacement data
to provide comprehensive emergency parameters.
"""

import os
import json
import logging
from datetime import datetime
from hdx_integration import get_security_parameters_from_hdx, get_emergency_parameters_from_hdx

logger = logging.getLogger(__name__)

# Import DTM integration
try:
    import dtm_integration
    DTM_AVAILABLE = True
    logger.info("DTM integration module loaded successfully")
except ImportError as e:
    DTM_AVAILABLE = False
    logger.warning(f"DTM integration not available: {e}")

def combine_emergency_data_sources(hapi_params, dtm_params):
    """
    Combine HAPI/ACLED conflict data with DTM displacement data
    to create comprehensive emergency parameters
    """
    if not hapi_params and not dtm_params:
        return None
    
    if not hapi_params:
        logger.info("Using DTM-only emergency parameters")
        return dtm_params
    
    if not dtm_params:
        logger.info("Using HAPI-only emergency parameters")
        return hapi_params
    
    try:
        logger.info("Combining HAPI conflict data with DTM displacement data")
        
        # Base emergency probability: weighted average of both sources
        hapi_prob = hapi_params.get('base_security_risk', 0.22)
        dtm_prob = dtm_params.get('emergency_probability', 0.18)
        
        # Weight HAPI (conflict) slightly higher than DTM (displacement) 
        combined_prob = (hapi_prob * 0.6) + (dtm_prob * 0.4)
        
        # Combine monthly risk factors
        hapi_monthly = hapi_params.get('monthly_risk_factors', {})
        dtm_monthly = dtm_params.get('monthly_risk_factors', {})
        
        combined_monthly = {}
        for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
            hapi_factor = hapi_monthly.get(month, 1.0)
            dtm_factor = dtm_monthly.get(month, 1.0)
            combined_monthly[month] = (hapi_factor * 0.6) + (dtm_factor * 0.4)
        
        # Combine emergency probability modifiers
        hapi_modifiers = hapi_params.get('security_risk_modifiers', {})
        dtm_modifiers = dtm_params.get('emergency_probability_modifiers', {})
        
        combined_modifiers = {}
        for org_type in ['NGO', 'UN Agency', 'Hybrid']:
            hapi_mod = hapi_modifiers.get(org_type, 1.0)
            dtm_mod = dtm_modifiers.get(org_type, 1.0)
            combined_modifiers[org_type] = (hapi_mod + dtm_mod) / 2
        
        # Create combined parameters
        combined_params = {
            'emergency_probability': combined_prob,
            'monthly_risk_factors': combined_monthly,
            'emergency_probability_modifiers': combined_modifiers,
            'emergency_impact': dtm_params.get('emergency_impact', {}),
            'data_sources': {
                'hapi_conflict': hapi_params.get('data_source', 'HAPI Conflict Events API'),
                'dtm_displacement': dtm_params.get('data_source', 'DTM Displacement API')
            },
            'combination_method': 'weighted_average',
            'hapi_weight': 0.6,
            'dtm_weight': 0.4,
            'data_source_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            
            # Include raw data from both sources
            'raw_hapi_data': hapi_params,
            'raw_dtm_data': dtm_params,
            
            # Keep important data from both sources
            'displacement_statistics': dtm_params.get('displacement_statistics', {}),
            'event_type_distribution': hapi_params.get('event_type_distribution', {})
        }
        
        logger.info(f"Combined emergency parameters: probability={combined_prob:.3f}")
        logger.info(f"Data sources: HAPI (60%) + DTM (40%)")
        
        return combined_params
        
    except Exception as e:
        logger.error(f"Error combining emergency data sources: {e}", exc_info=True)
        return hapi_params

def get_enhanced_hdx_parameters():
    """
    Get HDX parameters with DTM integration:
    - Emergency: Combined HAPI/ACLED + DTM displacement data  
    - Security: ACAPS INFORM Severity Index data
    """
    try:
        # Get HAPI/ACLED data (conflict events) - now for emergency
        hapi_params = get_security_parameters_from_hdx()
        
        # Get DTM displacement data for emergency
        dtm_params = None
        if DTM_AVAILABLE:
            try:
                dtm_params = dtm_integration.get_dtm_emergency_parameters()
                logger.info("Successfully fetched DTM displacement parameters")
            except Exception as e:
                logger.error(f"Error fetching DTM parameters: {e}")
                dtm_params = None
        
        # Combine HAPI and DTM for emergency parameters
        emergency_params = combine_emergency_data_sources(hapi_params, dtm_params)
        
        # Get ACAPS data for security parameters  
        security_params = get_emergency_parameters_from_hdx()
        
        if not security_params:
            logger.warning("Using fallback simulated security parameters.")
            security_params = get_emergency_parameters_from_hdx()

        if not emergency_params or not security_params:
            logger.error("Failed to fetch parameters.")
            return None

        return {
            "security_parameters": security_params,
            "emergency_parameters": emergency_params,
            "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_sources": {
                "emergency": ["HAPI/ACLED", "DTM"] if dtm_params else ["HAPI/ACLED"],
                "security": ["ACAPS"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error in enhanced HDX parameter generation: {e}", exc_info=True)
        return None

# Test function
def test_enhanced_integration():
    """Test the enhanced HDX+DTM integration"""
    print("Testing Enhanced HDX+DTM Integration...")
    
    try:
        params = get_enhanced_hdx_parameters()
        if params:
            print("✅ Enhanced integration successful")
            emergency_data = params.get('emergency_parameters', {})
            print(f"   Emergency sources: {params.get('data_sources', {}).get('emergency', [])}")
            if 'displacement_statistics' in emergency_data:
                stats = emergency_data['displacement_statistics']
                print(f"   IDPs: {stats.get('total_idps', 'N/A'):,}")
            return True
        else:
            print("❌ Enhanced integration failed")
            return False
    except Exception as e:
        print(f"❌ Error testing enhanced integration: {e}")
        return False

if __name__ == "__main__":
    test_enhanced_integration()