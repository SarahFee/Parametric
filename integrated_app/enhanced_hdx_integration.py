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

# Import ACLED integration
try:
    import acled_integration
    ACLED_AVAILABLE = True
    logger.info("ACLED integration module loaded successfully")
except ImportError as e:
    ACLED_AVAILABLE = False
    logger.warning(f"ACLED integration not available: {e}")

def combine_emergency_data_sources(acled_params, dtm_params):
    """
    Combine ACLED conflict data with DTM displacement data
    to create comprehensive emergency parameters
    """
    if not acled_params and not dtm_params:
        return None
    
    if not acled_params:
        logger.info("Using DTM-only emergency parameters")
        return dtm_params
    
    if not dtm_params:
        logger.info("Using ACLED-only emergency parameters")
        return acled_params
    
    try:
        logger.info("Combining ACLED conflict data with DTM displacement data")
        
        # Base emergency probability: weighted average of both sources
        acled_prob = acled_params.get('emergency_probability', 0.25)
        dtm_prob = dtm_params.get('emergency_probability', 0.18)
        
        # Weight ACLED (conflict) slightly higher than DTM (displacement) 
        combined_prob = (acled_prob * 0.6) + (dtm_prob * 0.4)
        
        # Combine monthly risk factors
        acled_monthly = acled_params.get('monthly_risk_factors', {})
        dtm_monthly = dtm_params.get('monthly_risk_factors', {})
        
        combined_monthly = {}
        for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
            acled_factor = acled_monthly.get(month, 1.0)
            dtm_factor = dtm_monthly.get(month, 1.0)
            combined_monthly[month] = (acled_factor * 0.6) + (dtm_factor * 0.4)
        
        # Combine emergency probability modifiers
        acled_modifiers = acled_params.get('emergency_probability_modifiers', {})
        dtm_modifiers = dtm_params.get('emergency_probability_modifiers', {})
        
        combined_modifiers = {}
        for org_type in ['NGO', 'UN Agency', 'Hybrid']:
            acled_mod = acled_modifiers.get(org_type, 1.0)
            dtm_mod = dtm_modifiers.get(org_type, 1.0)
            combined_modifiers[org_type] = (acled_mod + dtm_mod) / 2
        
        # Create combined parameters
        combined_params = {
            'emergency_probability': combined_prob,
            'monthly_risk_factors': combined_monthly,
            'emergency_probability_modifiers': combined_modifiers,
            'emergency_impact': dtm_params.get('emergency_impact', {}),
            'data_sources': {
                'acled_conflict': acled_params.get('data_source', 'ACLED Conflict Events API'),
                'dtm_displacement': dtm_params.get('data_source', 'DTM Displacement API')
            },
            'combination_method': 'weighted_average',
            'acled_weight': 0.6,
            'dtm_weight': 0.4,
            'data_source_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            
            # Include raw data from both sources
            'raw_acled_data': acled_params,
            'raw_dtm_data': dtm_params,
            
            # Keep important data from both sources
            'displacement_statistics': dtm_params.get('displacement_statistics', {}),
            'conflict_statistics': acled_params.get('conflict_statistics', {})
        }
        
        logger.info(f"Combined emergency parameters: probability={combined_prob:.3f}")
        logger.info(f"Data sources: ACLED (60%) + DTM (40%)")
        
        return combined_params
        
    except Exception as e:
        logger.error(f"Error combining emergency data sources: {e}", exc_info=True)
        return acled_params

def get_enhanced_hdx_parameters(use_acled_emergency=True, use_acaps_security=True, use_dtm_emergency=True):
    """
    Get HDX parameters with DTM integration based on user selections:
    - Emergency: Combined ACLED + DTM displacement data (based on selections)
    - Security: ACAPS INFORM Severity Index data (based on selections)
    """
    try:
        # Get ACLED data (conflict events) - only if user selected it
        acled_params = None
        if use_acled_emergency and ACLED_AVAILABLE:
            acled_params = acled_integration.get_acled_emergency_parameters()
            if acled_params:
                logger.info("Successfully fetched ACLED parameters for emergency")
        
        # Get DTM displacement data for emergency - only if user selected it
        dtm_params = None
        if use_dtm_emergency and DTM_AVAILABLE:
            try:
                dtm_params = dtm_integration.get_dtm_emergency_parameters()
                logger.info("Successfully fetched DTM displacement parameters")
            except Exception as e:
                logger.error(f"Error fetching DTM parameters: {e}")
                dtm_params = None
        
        # Combine ACLED and DTM for emergency parameters
        emergency_params = combine_emergency_data_sources(acled_params, dtm_params)
        
        # Get ACAPS data for security parameters - only if user selected it
        security_params = None
        if use_acaps_security:
            security_params = get_emergency_parameters_from_hdx()
            if security_params:
                logger.info("Successfully fetched ACAPS parameters for security")
            else:
                logger.warning("Failed to fetch ACAPS security parameters.")
        
        # If user selected DTM but not other emergency sources, ensure emergency_params is DTM-only
        if use_dtm_emergency and not use_acled_emergency and not emergency_params:
            emergency_params = dtm_params

        # Allow DTM-only or ACAPS-only modes
        if not emergency_params and not security_params:
            logger.error("Failed to fetch any parameters.")
            return None

        return {
            "security_parameters": security_params,
            "emergency_parameters": emergency_params,
            "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_sources": {
                "emergency": ["ACLED", "DTM"] if dtm_params else ["ACLED"],
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