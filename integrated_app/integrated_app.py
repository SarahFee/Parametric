"""
Integrated Business Continuity Insurance Application

This script demonstrates the integrated model with both HDX and IATI data sources.
"""
import streamlit as st
import sys
import os
import logging
import traceback
from datetime import datetime
import transparent_viz
from dotenv import load_dotenv
load_dotenv()

# Initialize session state for storing simulation results
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'params' not in st.session_state:
    st.session_state.params = None

# Configure the page (must be the first Streamlit command)
st.set_page_config(
    page_title="Integrated Business Continuity Insurance",
    page_icon="🛡️",
    layout="wide"
)

# Collect import errors
import_errors = []

# Function to safely import modules
def safe_import(module_name):
    try:
        return __import__(module_name)
    except ImportError as e:
        import_errors.append({
            'module': module_name,
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        return None

# Additional imports
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Configure logging
log_directory = 'logs'
os.makedirs(log_directory, exist_ok=True)
app_log_file = os.path.join(log_directory, 'integrated_app.log')
print(f"--- CONFIGURING LOGGING TO: {os.path.abspath(app_log_file)} ---")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(app_log_file),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger(__name__)
logger.info("--- Integrated app started, logging configured ---")

# Ensure the directory containing these modules is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Try to import modules safely
integrated_model = safe_import('integrated_model')
hdx_integration = safe_import('hdx_integration')
iati_api = safe_import('iati_api')
sudan_map_timeline = safe_import('sudan_map_timeline')

# Prepare import status flags
MODEL_AVAILABLE = integrated_model is not None
HDX_AVAILABLE = hdx_integration is not None
MAP_AVAILABLE = sudan_map_timeline is not None
IATI_AVAILABLE = iati_api is not None

# Set the default API key
DEFAULT_HDX_API_KEY = os.getenv('HDX_API_KEY', 'TXlBcHA6ZmVraWguc2FyYWhAZ21haWwuY29t')

# Display any import errors that occurred
if import_errors:
    st.error("The following import errors were encountered:")
    for error in import_errors:
        st.markdown(f"**Module:** {error['module']}")
        st.markdown(f"**Error:** {error['error']}")
        with st.expander("Detailed Traceback"):
            st.code(error['traceback'])

# Unified parameter explanations dictionary
param_explanations = {
    'sim_duration': "Controls how many months the simulation runs for. Longer durations provide a more comprehensive view of long-term dynamics.",
    'payout_cap_multiple': "Maximum insurance payout as a multiple of the premium paid. Higher caps provide more protection but increase insurer risk.",
    'premium_rate': "Percentage of organization's total budget that must be paid as insurance premium.",
    'claim_trigger': "Balance threshold (as fraction of total budget) that triggers insurance claims. Lower thresholds mean earlier access to insurance funds.",
    'waiting_period': "Minimum time between successive claims by the same organization.",
    'emergency_probability': "Probability of an emergency event occurring in each step. Derived from ACAPS INFORM Severity Index data showing crisis levels.",
    'security_risk_factor': "Base probability of security events occurring in each step. Derived from ACLED conflict event data from Sudan.",
    'hdx_api_key': "API key for accessing the Humanitarian Data Exchange API (HAPI). Allows fetching real-time data for security and emergency parameters."
}

def show_introduction():
    """Display an introduction panel explaining the simulator"""
    st.markdown("""
    <div class="intro-panel">
        <h2 style="color: #000000 !important;">Welcome to the Integrated Business Continuity Insurance Simulator</h2>
        <p style="color: #000000 !important;">This simulator helps visualize how insurance products can protect organizations operating in high-risk environments from financial disruptions.</p>
        <p style="color: #000000 !important;">Organizations face regular operational costs, unexpected emergencies, and security evacuations that drain their resources. By subscribing to business continuity insurance, they can receive financial support during crisis periods.</p>
        <p style="color: #000000 !important;">The simulation is calibrated with real-world data from <strong style="color: #000000 !important;">two data sources</strong>:</p>
        <ul>
            <li style="color: #000000 !important;"><strong style="color: #000000 !important;">Humanitarian Data Exchange (HDX)</strong>: Security risk factors from ACLED conflict data and emergency data from ACAPS INFORM Severity Index</li>
            <li style="color: #000000 !important;"><strong style="color: #000000 !important;">International Aid Transparency Initiative (IATI)</strong>: For organization financial profiles</li>
        </ul>
        <p style="color: #000000 !important;">This integrated approach provides more realistic parameters for World Vision (NGO), UNHCR (UN Agency), and IOM (Hybrid) operations in Sudan.</p>
        <p style="color: #000000 !important;">Use the parameters in the sidebar to adjust the simulation and explore different scenarios.</p>
    </div>
    """, unsafe_allow_html=True)

def display_logs():
    """Displays the last 20 lines of the application log file."""
    log_file = app_log_file
    logger.debug(f"Attempting to display logs from: {os.path.abspath(log_file)}")
    st.markdown('<div class="parameter-title">Application Logs</div>', unsafe_allow_html=True)
    with st.expander("View Recent Application Logs"):
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    all_lines = f.readlines()
                    logs = all_lines[-20:]
                st.text_area(
                    f"Recent Logs ({len(logs)}/{len(all_lines)} lines shown)",
                    "".join(logs), height=300, key="app_log_display"
                )
            except Exception as e:
                st.error(f"Error reading log file ({log_file}): {e}")
                logger.error(f"Error reading log file ({log_file}): {e}", exc_info=True)
        else:
            st.warning(f"Application log file not found at {log_file}")
            logger.warning(f"Attempted to display logs, but file not found: {log_file}")

def get_parameters():
    """
    Gather simulation parameters from the user via the sidebar.
    """
    st.sidebar.markdown("## Simulation Parameters")
    
    # Default parameters
    default_duration = 12
    default_payout_cap = 3.0
    default_premium_rate = 2.0
    default_claim_trigger = 0.5
    default_waiting_period = 2
    default_emergency_prob = 5.0
    default_security_risk = 20.0

    # Data source toggles with separate controls
    st.sidebar.markdown("### Data Sources")
    
    # Separate checkboxes for HDX and ACAPS data
    use_hdx_security_data = st.sidebar.checkbox("Use HDX Security Data (ACLED)", value=False, 
                                  help="When checked, the model will use HDX data for security parameters")
    
    use_acaps_emergency_data = st.sidebar.checkbox("Use ACAPS Emergency Data", value=False, 
                                   help="When checked, the model will use ACAPS INFORM Severity Index for emergency parameters")
    
    # Compute a flag for any HDX-related data for backwards compatibility
    use_hdx_data = use_hdx_security_data or use_acaps_emergency_data
    
    # Add API key field when either HDX source is enabled
    hdx_api_key = DEFAULT_HDX_API_KEY  # Always use the default API key
    if use_hdx_data:
        # Show that we're using the default API key
        st.sidebar.info(f"Using HDX API Key: {hdx_api_key[:4]}...")
        
        # Option to clear cache
        clear_cache = st.sidebar.checkbox("Clear data cache (fetch fresh data)", 
                                       value=False,
                                       help="When checked, cached data will be cleared and fresh data will be fetched")
        
        if clear_cache and st.sidebar.button("Clear Cache Now"):
            try:
                cache_dir = "hdx_cache"
                cache_files = [
                    os.path.join(cache_dir, "hapi_acled_security_data.json"),
                    os.path.join(cache_dir, "acaps_emergency_data.json")
                ]
                
                for cache_file in cache_files:
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                        st.sidebar.success(f"Deleted cache file: {os.path.basename(cache_file)}")
                        logger.info(f"Deleted cache file: {cache_file}")
                
                st.sidebar.success("All cache files cleared! Fresh data will be fetched when you run the simulation.")
            except Exception as e:
                st.sidebar.error(f"Error clearing cache: {e}")
                logger.error(f"Error clearing cache: {e}", exc_info=True)
    
    use_iati_data = st.sidebar.checkbox("Use IATI Data", value=False, 
                                      help="When checked, the model will use IATI data for organization financial profiles")

    # General simulation settings
    st.sidebar.markdown("### Simulation Settings")
    sim_duration = st.sidebar.slider("Duration (Months)", 
                                  6, 36, default_duration, 
                                  help=param_explanations['sim_duration'])

    # Insurance parameters
    st.sidebar.markdown("### Insurance Parameters")
    payout_cap_multiple = st.sidebar.slider("Payout Cap (x Premium)", 
                                         1.0, 5.0, default_payout_cap, 
                                         help=param_explanations['payout_cap_multiple'])
    
    premium_rate = st.sidebar.slider("Premium Rate (%)", 
                                  0.5, 5.0, default_premium_rate, 0.1, 
                                  help=param_explanations['premium_rate'])
    
    claim_trigger = st.sidebar.slider("Claim Trigger (fraction of budget)", 
                                 0.1, 1.0, default_claim_trigger, 0.05, 
                                 help=param_explanations['claim_trigger'])
    
    waiting_period = st.sidebar.slider("Waiting Period (months)", 
                                    1, 12, default_waiting_period, 
                                    help=param_explanations['waiting_period'])

    # Risk factors - only shown if HDX data is disabled or manual override is selected
    show_risk_factors = not use_hdx_data or st.sidebar.checkbox("Manual Risk Override", value=False, 
                                                              help="Override HDX risk factors with manual values")
    
    if show_risk_factors:
        st.sidebar.markdown("### Risk Factors (Manual Override)")
        emergency_probability = st.sidebar.slider("Emergency Probability (%)", 
                                               0.0, 20.0, default_emergency_prob, 0.5, 
                                               help=param_explanations['emergency_probability']) / 100
        security_risk_factor = st.sidebar.slider("Security Risk Factor (%)", 
                                              0.0, 50.0, default_security_risk, 1.0, 
                                              help=param_explanations['security_risk_factor']) / 100
    else:
        st.sidebar.markdown("### Risk Factors (Data-Driven)")
        data_source = "HDX API" if use_hdx_data else "Default values"
        st.sidebar.info(f"Risk factors will be calculated from {data_source}.")
        emergency_probability = None
        security_risk_factor = None

    # Parameter summary
    with st.sidebar.expander("View Parameter Summary"):
        st.write(f"- Duration: {sim_duration} months")
        st.write(f"- Payout Cap: {payout_cap_multiple}x premium")
        st.write(f"- Premium Rate: {premium_rate}%")
        st.write(f"- Claim Trigger: {claim_trigger * 100}% of budget")
        st.write(f"- Waiting Period: {waiting_period} months")
        if show_risk_factors:
            st.write(f"- Manual Emergency Probability: {emergency_probability * 100 if emergency_probability is not None else 'N/A'}%")
            st.write(f"- Manual Security Risk Factor: {security_risk_factor * 100 if security_risk_factor is not None else 'N/A'}%")
        st.write(f"- Use HDX Data: {use_hdx_data}")
        if use_hdx_data:
            st.write(f"- Using HDX API: Yes (key: {hdx_api_key[:4]}...)")
        st.write(f"- Use IATI Data: {use_iati_data}")
        st.write("**Data Sources:**")
        st.write("- Security Risk: ACLED conflict data for Sudan")
        st.write("- Emergency Risk: ACAPS INFORM Severity Index for Sudan")
        st.write("- Financial Profiles: IATI transaction data")
    
    with st.sidebar.expander("Documentation & Help"):
        st.markdown("""
        ### Business Continuity Insurance Model
        
        This simulator models how insurance products can protect humanitarian organizations 
        operating in high-risk environments from financial disruptions.
        
        **Key Features:**
        - Uses real-world data from HDX and IATI
        - Models emergency and security risk events
        - Simulates insurance claims based on financial thresholds
        - Visualizes financial impacts and decision making
        
        For more information on:
        - **Risk Parameters:** See the "Risk Calculation" tab in Model Explanation
        - **Claim Triggers:** See the "Claim Triggers" tab in Model Explanation
        - **Payouts:** See the "Payout Mechanism" tab in Model Explanation
        """)

    # Log parameter selection
    logger.info("Simulation Parameters Selected:")
    logger.info(f"  Duration: {sim_duration} steps")
    logger.info(f"  Payout Cap: {payout_cap_multiple}x premium")
    logger.info(f"  Claim Trigger: {claim_trigger}")
    logger.info(f"  Waiting Period: {waiting_period} months")
    logger.info(f"  Use HDX Data: {use_hdx_data}")
    logger.info(f"  Use HDX API: {hdx_api_key is not None}")
    logger.info(f"  Use IATI Data: {use_iati_data}")
    if show_risk_factors:
        logger.info(f"  Manual Emergency Probability: {emergency_probability}")
        logger.info(f"  Manual Security Risk Factor: {security_risk_factor}")

    params = {
        'sim_duration': sim_duration,
        'payout_cap_multiple': payout_cap_multiple,
        'premium_rate': premium_rate,
        'claim_trigger': claim_trigger,
        'waiting_period': waiting_period,
        'emergency_probability': emergency_probability,
        'security_risk_factor': security_risk_factor,
        'use_hdx_data': use_hdx_data,  # Keep for backwards compatibility
        'use_hdx_security_data': use_hdx_security_data,  # New flag for security data only
        'use_acaps_emergency_data': use_acaps_emergency_data,  # New flag for emergency data only
        'use_iati_data': use_iati_data,
        'hdx_api_key': hdx_api_key if use_hdx_data else None
    }
    return params

def fetch_hdx_data(api_key, show_spinner=True):
    """
    Fetch HDX data using the provided API key.
    Returns the fetched data or None if fetching fails.
    """
    if not HDX_AVAILABLE or not api_key:
        return None
        
    try:
        with st.spinner("Fetching data from HDX...") if show_spinner else nullcontext():
            logger.info(f"Fetching HDX data with API key {api_key[:4]}...")
            
            # Use the module to call the function
            hdx_data = hdx_integration.get_all_hdx_parameters()
            
            if hdx_data:
                logger.info("Successfully fetched HDX data")
                
                # Log the data sources used
                sec_source = hdx_data.get("security_parameters", {}).get("data_source", "Unknown")
                emg_source = hdx_data.get("emergency_parameters", {}).get("data_source", "Unknown")
                logger.info(f"Data sources: Security = {sec_source}, Emergency = {emg_source}")
                
                return hdx_data
            else:
                logger.warning("Failed to fetch HDX data")
                return None
    except Exception as e:
        logger.error(f"Error fetching HDX data: {e}", exc_info=True)
        return None
    

class nullcontext:
    """A context manager that does nothing."""
    def __enter__(self): return None
    def __exit__(self, *args): pass

def run_model_and_get_results(params):
    """
    Creates model, runs simulation, extracts serializable results.
    Returns a dictionary of results, or None on failure.
    """
    st.write("Setting up simulation model...")
    logger.info("Initializing simulation model")

    if not MODEL_AVAILABLE:
        st.error("IntegratedBusinessContinuityModel is not available. Check logs.")
        logger.error("Attempted to run model, but MODEL_AVAILABLE is False.")
        return None

    try:
        # Import the model class dynamically to avoid errors if not imported
        IntegratedBusinessContinuityModel = getattr(integrated_model, 'IntegratedBusinessContinuityModel')
        
        # Try to fetch HDX data with API key if provided
        hdx_data = None
        if params['use_hdx_data'] and HDX_AVAILABLE and params.get('hdx_api_key'):
            hdx_api_key = params['hdx_api_key']
            st.info(f"Connecting to HDX API with key {hdx_api_key[:4]}...")
            
            # Use the fetch_hdx_data function defined above
            hdx_data = fetch_hdx_data(hdx_api_key)
            
            if hdx_data:
                # Extract and display data sources
                sec_source = hdx_data.get("security_parameters", {}).get("data_source", "Unknown")
                emg_source = hdx_data.get("emergency_parameters", {}).get("data_source", "Unknown")
                st.success(f"Successfully connected to HDX API!")
                st.info(f"Data sources: Security = {sec_source}, Emergency = {emg_source}")
            else:
                st.warning("Could not fetch HDX data with API key. Using fallback data.")

        # Status message for model initialization
        with st.spinner("Initializing simulation model..."):
            model = IntegratedBusinessContinuityModel(
                sim_duration=params['sim_duration'],
                waiting_period=params['waiting_period'],
                payout_cap_multiple=params['payout_cap_multiple'],
                premium_rate=params['premium_rate'],
                custom_claim_trigger=params['claim_trigger'],
                emergency_probability=params['emergency_probability'],
                security_risk_factor=params['security_risk_factor'],
                use_hdx_data=params['use_hdx_data'],
                use_iati_data=params['use_iati_data'],
                hdx_data=hdx_data,
                hdx_api_key=params['hdx_api_key']
            )

        # Initialize history lists if needed (defensive)
        if not hasattr(model, 'history'): model.history = []
        if not hasattr(model, 'profit_history'): model.profit_history = []
        if not hasattr(model, 'org_impact_history'):
            model.org_impact_history = {org.name: [] for org in model.orgs} if hasattr(model, 'orgs') else {}

        initial_capitals = {org.name: org.balance for org in model.orgs} if hasattr(model, 'orgs') else {}

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        logger.info(f"Starting simulation run for {model.sim_duration} steps.")
        for step in range(model.sim_duration):
            progress = (step + 1) / model.sim_duration
            progress_bar.progress(progress)
            status_text.text(f"Running simulation step {step + 1}/{model.sim_duration}...")
            logger.debug(f"Running model step {step}")
            model.step() # Includes internal data collection for history/profit

        # --- Post-Loop: Clear UI & Extract Data ---
        progress_bar.empty()
        status_text.empty()

        event_log_data = []
        if hasattr(model, 'event_log') and model.event_log:
            st.success(f"Simulation loop complete! Generated {len(model.event_log)} event log entries.")
            logger.info(f"Simulation loop completed successfully. {len(model.event_log)} events logged.")
            event_log_data = model.event_log.copy()
        else:
            st.warning("Simulation complete, but no event log generated.")
            logger.warning("Simulation completed without generating event log.")

        # Calculate final impact history here based on initial/final
        final_org_impact = {}
        if hasattr(model, 'orgs'):
             for org in model.orgs:
                  initial = initial_capitals.get(org.name, 0)
                  final = getattr(org, 'balance', 0)
                  final_org_impact[org.name] = final - initial

        results = {
            "event_log": event_log_data,
            "history": model.history.copy() if hasattr(model, 'history') else [],
            "profit_history": model.profit_history.copy() if hasattr(model, 'profit_history') else [],
            "sim_duration": getattr(model, 'sim_duration', 0),
            "final_org_states": [],
            "final_insurer_state": {},
            "final_donor_stats": [],
            "donor_configs": [],
            "hdx_params": model.hdx_params.copy() if hasattr(model,'hdx_params') and model.hdx_params else None,
            "hdx_data_source": getattr(model, 'hdx_data_source', 'Not specified') if hasattr(model, 'hdx_data_source') else 'Default',
            "iati_params": model.iati_params.copy() if hasattr(model, 'iati_params') and model.iati_params else None,
            "final_org_impact": final_org_impact, # Store final calculated impact
            "org_balance_history": model.org_balance_history.copy() if hasattr(model, 'org_balance_history') else {},
            "detailed_events": model.detailed_events.copy() if hasattr(model, 'detailed_events') else []
        }

        if hasattr(model, 'orgs'):
             results["final_org_states"] = [{
                "name": getattr(org, 'name', 'N/A'), 
                "org_type": getattr(org, 'org_type', 'N/A'),
                "balance": getattr(org, 'balance', 0), 
                "initial_capital": getattr(org, 'initial_capital', 0),
                "total_budget": getattr(org, 'total_budget', 0), 
                "premium_paid_total": getattr(org, 'premium_paid', 0),
                "last_claim_step": getattr(org, 'last_claim_step', -999),
                "security_risk_modifier": getattr(org, 'security_risk_modifier', 1.0),
                "emergency_probability_modifier": getattr(org, 'emergency_probability_modifier', 1.0),
                "emergency_impact": getattr(org, 'emergency_impact', 0.1),
                "claim_trigger": getattr(org, 'claim_trigger', 0.5)  # Make sure this is included
            } for org in model.orgs]

        if hasattr(model, 'insurer'):
             results["final_insurer_state"] = {
                  "name": getattr(model.insurer, 'name', 'ParaInsure'), 
                  "capital": getattr(model.insurer, 'capital', 0),
                  "initial_capital": getattr(model.insurer, 'initial_capital', 0),
                  "profits": getattr(model.insurer, 'profits', 0), 
                  "premiums_collected": getattr(model.insurer, 'premiums_collected', 0),
                  "payouts_made": getattr(model.insurer, 'payouts_made', 0)
             }

        if hasattr(model, 'donors'):
             results["final_donor_stats"] = [
                  donor.get_donation_statistics() for donor in model.donors if hasattr(donor, 'get_donation_statistics')
             ]
             results["donor_configs"] = [{
                    "name": getattr(d, 'name', 'N/A'), 
                    "donor_type": getattr(d, 'donor_type', 'N/A'),
                    "avg_donation": getattr(d, 'avg_donation', 0), 
                    "donation_frequency": getattr(d, 'donation_frequency', 0),
                    "target_preferences": getattr(d, 'target_preferences', {})
             } for d in model.donors]

        logger.info("Data extracted from model for session state.")
        return results

    except Exception as e:
        st.error(f"Simulation or data extraction failed: {e}")
        logger.error(f"Simulation/extraction failed: {e}", exc_info=True)
        return None

def display_metrics(results):
    """Display summary metrics from the results dictionary."""
    try:
        final_orgs = results.get('final_org_states', [])
        final_insurer = results.get('final_insurer_state', {})
        sim_duration = results.get('sim_duration', 'N/A')
        hdx_data_source = results.get('hdx_data_source', 'Not specified')

        if not final_orgs:
            st.warning("No final organization states found in results.")
            return

        final_balances = [org.get('balance', 0) for org in final_orgs]
        avg_balance = sum(final_balances) / len(final_balances) if final_balances else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-box">'
                        '<h3>Average Org Final Balance</h3>'
                        f'<h2>${avg_balance:,.0f}</h2></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-box">'
                        '<h3>Total Simulation Steps</h3>'
                        f'<h2>{sim_duration}</h2></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-box">'
                        '<h3>Insurer Final Profit</h3>'
                        f'<h2>${final_insurer.get("profits", 0):,.0f}</h2></div>', unsafe_allow_html=True)
        
        # Add data source information
        st.markdown(f"**Data Source**: {hdx_data_source}")
        
        # Show security and emergency parameters if available
        hdx_params = results.get('hdx_params', {})
        if hdx_params:
            security_params = hdx_params.get('security_parameters', {})
            emergency_params = hdx_params.get('emergency_parameters', {})
            
            if security_params or emergency_params:
                with st.expander("View Risk Parameters"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Security Risk Parameters**")
                        if security_params:
                            st.write(f"Base Risk: {security_params.get('base_security_risk', 0) * 100:.1f}%")
                            st.write(f"Source: {security_params.get('data_source', 'Unknown')}")
                            if 'security_risk_modifiers' in security_params:
                                st.write("Organization Risk Modifiers:")
                                for org_type, modifier in security_params['security_risk_modifiers'].items():
                                    st.write(f"- {org_type}: {modifier:.2f}x")
                    
                    with col2:
                        st.markdown("**Emergency Risk Parameters**")
                        if emergency_params:
                            st.write(f"Base Probability: {emergency_params.get('emergency_probability', 0) * 100:.1f}%")
                            st.write(f"Source: {emergency_params.get('data_source', 'Unknown')}")
                            if 'emergency_probability_modifiers' in emergency_params:
                                st.write("Organization Emergency Modifiers:")
                                for org_type, modifier in emergency_params['emergency_probability_modifiers'].items():
                                    st.write(f"- {org_type}: {modifier:.2f}x")
                            
                            # Display additional ACAPS-specific data if available
                            if 'crisis_percentage' in emergency_params:
                                st.write("\nACAPS INFORM Severity Categories:")
                                st.write(f"- Crisis: {emergency_params.get('crisis_percentage', 0) * 100:.1f}%")
                                st.write(f"- Emergency: {emergency_params.get('emergency_percentage', 0) * 100:.1f}%")
                                st.write(f"- Catastrophe: {emergency_params.get('catastrophe_percentage', 0) * 100:.1f}%")
    except Exception as e:
        st.error(f"Error displaying metrics: {e}")
        logger.error(f"Error in display_metrics: {e}", exc_info=True)

def plot_continuity_impact(results):
    """Plot individual organization balances over time from results."""
    try:
        org_balance_history = results.get('org_balance_history', {})
        if not org_balance_history:
            st.warning("No organization balance history data available to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define colors for different organization types
        colors = {
            "World-Vision": "#3498db",  # NGO - Blue
            "UNHCR": "#2ecc71",         # UN Agency - Green
            "IOM": "#9b59b6"            # Hybrid - Purple
        }
        
        # Get org_types from final_org_states
        org_types = {}
        for org in results.get('final_org_states', []):
            org_types[org['name']] = org['org_type']
        
        # Plot each organization separately with its own line
        for org_name, balances in org_balance_history.items():
            time_steps = list(range(len(balances)))
            org_type = org_types.get(org_name, "")
            label = f"{org_name} ({org_type})"
            color = colors.get(org_name, "#7f8c8d")  # Default gray if not found
            
            ax.plot(time_steps, balances, color=color, linewidth=2.5, label=label)
        
        ax.set_title("Organization Balances Over Time", fontsize=14)
        ax.set_xlabel("Months (Steps)", fontsize=12)
        ax.set_ylabel("Balance ($)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.pyplot(fig)
        plt.close(fig)  # Close figure
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error plotting organization balances: {e}")
        logger.error(f"Error in plot_continuity_impact: {e}", exc_info=True)

def plot_profit_and_impact(results):
    """Plot the insurer profit and each organization's balance over time from results."""
    try:
        profit_history = results.get('profit_history', [])
        org_balance_history = results.get('org_balance_history', {})
        
        if not profit_history and not org_balance_history:
            st.warning("No profit or organization balance history data available to plot.")
            return

        # Create tabs for different visualization views
        tabs = st.tabs(["Insurer Profit", "Organizations Comparison", "Individual Organizations"])
        
        with tabs[0]:
            # --- Plotting Insurer Profit (if available) ---
            if profit_history:
                fig, ax = plt.subplots(figsize=(12, 6))
                steps = list(range(len(profit_history)))
                ax.plot(steps, profit_history, label="Insurer Profit", linewidth=2, color='green')
                ax.set_title("Insurer Profit Over Time", fontsize=14)
                ax.set_xlabel("Months (Steps)", fontsize=12)
                ax.set_ylabel("Value ($)", fontsize=12)
                ax.axhline(0, color='gray', linestyle=':', linewidth=1)
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
        
        with tabs[1]:
            # --- Plotting All Organizations Together for Comparison ---
            if org_balance_history:
                # Create a DataFrame for easier plotting with Plotly
                balance_data = {org_name: balances for org_name, balances in org_balance_history.items()}
                steps = list(range(len(next(iter(balance_data.values())))))
                
                # Use Plotly for interactive comparison
                import plotly.graph_objects as go
                fig = go.Figure()
                
                # Color mapping for org types
                colors = {
                    "World-Vision": "#3498db",  # NGO - Blue
                    "UNHCR": "#2ecc71",         # UN Agency - Green
                    "IOM": "#9b59b6"            # Hybrid - Purple
                }
                
                # Add insurer profit line
                if profit_history:
                    fig.add_trace(
                        go.Scatter(
                            x=steps,
                            y=profit_history,
                            mode='lines+markers',
                            name='Insurer (ParaInsure)',
                            line=dict(color='#e74c3c', width=3, dash='solid')
                        )
                    )
                
                # Add each organization
                for org_name, balances in balance_data.items():
                    # Get org type for label and color
                    org_type = next((org['org_type'] for org in results.get('final_org_states', []) 
                                   if org['name'] == org_name), "")
                    label = f"{org_name} ({org_type})"
                    color = colors.get(org_name, "#7f8c8d")  # Default gray if not found
                    
                    fig.add_trace(
                        go.Scatter(
                            x=steps,
                            y=balances,
                            mode='lines+markers',
                            name=label,
                            line=dict(color=color, width=2)
                        )
                    )
                
                # Layout
                fig.update_layout(
                    title="Financial Performance Comparison",
                    xaxis_title="Months (Steps)",
                    yaxis_title="Balance ($)",
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation
                st.markdown("""
                ### Financial Performance Comparison
                
                This chart shows how each organization's balance changes over time compared to the insurer's profit.
                * **NGO (Blue)**: World-Vision
                * **UN Agency (Green)**: UNHCR
                * **Hybrid (Purple)**: IOM
                * **Insurer (Red)**: ParaInsure
                
                You can hover over the lines to see exact values and click on the legend items to hide/show specific organizations.
                """)
        
        with tabs[2]:
            # --- Individual Organization Performance ---
            if org_balance_history:
                # Select organization to view
                org_names = list(org_balance_history.keys())
                selected_org = st.selectbox("Select Organization to View", org_names)
                
                if selected_org and selected_org in org_balance_history:
                    # Get org data
                    org_data = next((org for org in results.get('final_org_states', []) 
                                    if org['name'] == selected_org), {})
                    balances = org_balance_history[selected_org]
                    steps = list(range(len(balances)))
                    
                    # Create detailed organization view
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Balance line
                    ax.plot(steps, balances, label=f"{selected_org} Balance", linewidth=2, 
                            color=colors.get(selected_org, "#3498db"))
                    
                    # Add initial balance line
                    if org_data.get('initial_capital'):
                        initial_capital = org_data.get('initial_capital')
                        ax.axhline(initial_capital, color='gray', linestyle='--', 
                                  label=f"Initial Capital (${initial_capital:,.0f})")
                    
                    # Add claim trigger threshold
                    if org_data.get('total_budget') and org_data.get('claim_trigger'):
                        threshold = org_data.get('total_budget') * org_data.get('claim_trigger')
                        ax.axhline(threshold, color='red', linestyle=':', 
                                  label=f"Claim Trigger (${threshold:,.0f})")
                    
                    # Find claim events in event log
                    claim_steps = []
                    claim_amounts = []
                    for event in results.get('detailed_events', []):
                        if (event.get('org_name') == selected_org and 
                            'claim' in event.get('type', '').lower() and
                            event.get('step') is not None and 
                            event.get('impact') is not None):
                            claim_steps.append(event.get('step'))
                            # Get balance at that step
                            if 0 <= event.get('step') < len(balances):
                                claim_amounts.append(balances[event.get('step')])
                    
                    # Mark claims on the chart
                    if claim_steps and claim_amounts:
                        ax.scatter(claim_steps, claim_amounts, color='red', s=100, zorder=5,
                                  marker='*', label='Insurance Claims')
                    
                    # Configure the plot
                    ax.set_title(f"{selected_org} ({org_data.get('org_type', '')}) Financial Performance", fontsize=14)
                    ax.set_xlabel("Months (Steps)", fontsize=12)
                    ax.set_ylabel("Balance ($)", fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Organization stats
                    st.subheader(f"{selected_org} Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Organization Type", org_data.get('org_type', 'Unknown'))
                        st.metric("Initial Capital", f"${org_data.get('initial_capital', 0):,.0f}")
                    
                    with col2:
                        st.metric("Final Balance", f"${org_data.get('balance', 0):,.0f}")
                        st.metric("Total Budget", f"${org_data.get('total_budget', 0):,.0f}")
                    
                    with col3:
                        initial = org_data.get('initial_capital', 0)
                        final = org_data.get('balance', 0)
                        change = final - initial
                        change_pct = (change / initial * 100) if initial else 0
                        
                        st.metric("Net Change", 
                                 f"${change:,.0f}", 
                                 f"{change_pct:.1f}%",
                                 delta_color="normal" if change >= 0 else "inverse")
                        st.metric("Premium Paid", f"${org_data.get('premium_paid_total', 0):,.0f}")

    except Exception as e:
        st.error(f"Error plotting organization performance: {e}")
        logger.error(f"Error in plot_organization_performance: {e}", exc_info=True)

def display_key_insights(results):
    """Display key insights from the simulation results dictionary."""
    try:
        st.subheader("Key Insights")

        final_orgs = results.get('final_org_states', [])
        final_insurer = results.get('final_insurer_state', {})
        event_log = results.get('event_log', [])
        sim_duration = results.get('sim_duration', 0)
        hdx_data_source = results.get('hdx_data_source', 'Not specified')

        if not final_orgs or not final_insurer:
             st.warning("Results data incomplete for key insights.")
             return

        total_premiums = final_insurer.get('premiums_collected', 0)
        insurer_profits = final_insurer.get('profits', 0)

        emergency_count = sum(1 for log in event_log if "emergency" in log.lower())
        security_count = sum(1 for log in event_log if "security" in log.lower())
        claim_count = sum(1 for log in event_log if "made a claim" in log.lower())

        # Adjust names if needed based on your actual org names in results['final_org_states']
        org_names = {org.get('name'): org.get('org_type') for org in final_orgs}
        ngo_claims = sum(1 for log in event_log if "made a claim" in log.lower() and any(name in log for name, type in org_names.items() if type == "NGO"))
        un_claims = sum(1 for log in event_log if "made a claim" in log.lower() and any(name in log for name, type in org_names.items() if type == "UN Agency"))
        hybrid_claims = sum(1 for log in event_log if "made a claim" in log.lower() and any(name in log for name, type in org_names.items() if type == "Hybrid"))

        # Create a white background card using Streamlit elements
        with st.container():
            # Create a custom style that adds padding and borders
            st.markdown("""
            <style>
            div.stMarkdown {
                background-color: white;
                padding: 1px 20px;
                border-radius: 5px;
                margin-bottom: 1px;
            }
            div.stMarkdown h3 {
                color: #0066cc;
            }
            div.stMarkdown p, div.stMarkdown ul, div.stMarkdown li {
                color: black;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            # Left column content
            with col1:
                # The background and styling will be applied to this panel
                st.markdown("### Insurance Performance")
                profit_margin = (insurer_profits / total_premiums * 100) if total_premiums > 0 else 0
                st.write(f"• Total Premiums Collected: ${total_premiums:,.0f}")
                st.write(f"• Claims Processed (in log): {claim_count}")
                st.write(f"• Total Payouts Made: ${final_insurer.get('payouts_made', 0):,.0f}")
                st.write(f"• Insurer Final Profit: ${insurer_profits:,.0f}")
                st.write(f"• Net Profit Margin: {profit_margin:.1f}%")
                
                st.markdown("### Claims by Organization Type (logged)")
                st.write(f"• NGO: {ngo_claims} claims")
                st.write(f"• UN Agency: {un_claims} claims")
                st.write(f"• Hybrid: {hybrid_claims} claims")
                
            # Right column content
            with col2:
                st.markdown("### Risk Events Logged")
                disruption_rate = (emergency_count + security_count) / sim_duration if sim_duration > 0 else 0
                st.write(f"• Emergency events: {emergency_count}")
                st.write(f"• Security evacuations: {security_count}")
                st.write(f"• Total disruptions: {emergency_count + security_count}")
                st.write(f"• Disruption rate: {disruption_rate:.1f} per month")
                
                st.markdown("### Data Source Impact")
                st.write(f"• HDX Data Source: {hdx_data_source}")
                st.write("• Emergency Data: ACAPS INFORM Severity Index")
                st.write("• Security Data: ACLED conflict events")
                st.write(f"• IATI Data: {'Yes' if results.get('iati_params') else 'No'}")
                
    except Exception as e:
        st.error(f"Error displaying key insights: {e}")
        logger.error(f"Error in display_key_insights: {e}", exc_info=True)

def display_event_log(results):
    """Display the event log from the results dictionary."""
    try:
        st.subheader("Simulation Event Log")
        event_log = results.get('event_log', [])
        sim_duration = results.get('sim_duration', 0)
        org_states = results.get('final_org_states', [])  # For org names filter

        if not event_log:
            st.warning("No simulation event logs available in results.")
            return

        # Initialize session state for filter values if they don't exist
        if 'event_log_type_filter' not in st.session_state:
            st.session_state.event_log_type_filter = "All Events"
        if 'event_log_org_filter' not in st.session_state:
            st.session_state.event_log_org_filter = "All Participants"
        if 'event_log_step_filter' not in st.session_state:
            max_step = sim_duration - 1
            st.session_state.event_log_step_filter = (0, max_step)

        # Add filtering options that update session state
        col1, col2, col3 = st.columns(3)
        with col1:
            # Use a unique key by adding a prefix or suffix
            log_filter = st.selectbox(
                "Filter by event type:",
                ["All Events", "Claims", "Premium", "Emergencies", "Security Events", "Donations", "Funds Received", "Operations Cost"],
                key="event_log_type_filter_select",
                index=["All Events", "Claims", "Premium", "Emergencies", "Security Events", "Donations", "Funds Received", "Operations Cost"].index(st.session_state.event_log_type_filter)
            )
            # Update session state
            st.session_state.event_log_type_filter = log_filter
            
        with col2:
            org_names = [org.get('name', 'Unknown') for org in org_states]
            org_filter_options = ["All Participants"] + org_names + ["Donors", "ParaInsure"]
            try:
                org_filter_index = org_filter_options.index(st.session_state.event_log_org_filter)
            except ValueError:
                org_filter_index = 0  # Default to "All Participants" if not found
                
            org_filter = st.selectbox(
                "Filter by participant:",
                org_filter_options,
                key="event_log_org_filter_select",
                index=org_filter_index
            )
            # Update session state
            st.session_state.event_log_org_filter = org_filter
            
        with col3:
            max_step = max(1, sim_duration - 1)
            current_filter = st.session_state.event_log_step_filter
            
            # Ensure filter is within valid range
            min_step = min(max(0, current_filter[0]), max_step)
            max_step_filter = min(max(min_step, current_filter[1]), max_step)
            
            step_filter = st.slider(
                "Filter by step (month):",
                0, max_step, (min_step, max_step_filter),
                key="event_log_step_filter_slider"
            )
            # Update session state
            st.session_state.event_log_step_filter = step_filter

        # Now use the filter values from session state
        log_filter = st.session_state.event_log_type_filter
        org_filter = st.session_state.event_log_org_filter
        step_filter = st.session_state.event_log_step_filter

        # Rest of your filtering logic remains the same
        filtered_logs = event_log.copy()

        # Filter by event type
        filter_keywords = {
            "Claims": ["claim", "payout"], "Premium": ["premium"], "Emergencies": ["emergency"],
            "Security Events": ["security", "evacuation"], "Donations": ["donated"],
            "Funds Received": ["received funds"], "Operations Cost": ["spent on operations"]
        }
        if log_filter != "All Events" and log_filter in filter_keywords:
            keywords = filter_keywords[log_filter]
            filtered_logs = [log for log in filtered_logs if any(keyword in log.lower() for keyword in keywords)]

        # Filter by participant
        if org_filter != "All Participants":
            if org_filter == "Donors":
                filtered_logs = [log for log in filtered_logs if "donor" in log.lower()]
            elif org_filter == "ParaInsure":
                filtered_logs = [log for log in filtered_logs if "ParaInsure" in log or "Insurance Provider" in log]
            else:
                filtered_logs = [log for log in filtered_logs if org_filter in log]

        # Filter by step
        min_step, max_step_filter = step_filter
        filtered_logs = [
            log for log in filtered_logs
            if log.startswith("[Step ") and log.find("]") > 0 and
            min_step <= int(log[log.find(" ") + 1: log.find("]")]) <= max_step_filter
        ]

        # Display filtered logs
        st.write(f"Showing {len(filtered_logs)} of {len(event_log)} total event log entries")
        
        # Rest of your display code remains the same
        st.markdown('<div class="log-container">', unsafe_allow_html=True)
        if not filtered_logs:
            st.markdown('<div class="log-entry">No simulation event log entries match filter criteria.</div>', unsafe_allow_html=True)
        for log in filtered_logs:
            step_marker = log[: log.find("]") + 1] if log.startswith("[Step ") else ""
            log_content = log[len(step_marker):].strip()
            style_class = "log-entry"
            # Adjust names if needed
            if "World-Vision" in log:
                style_class += " org-ngo"
            elif "UNHCR" in log:
                style_class += " org-un"
            elif "IOM" in log:
                style_class += " org-hybrid"
            if "===" in log or "--- Step" in log:
                style_class += " log-header"
            st.markdown(f'<div class="{style_class}"><span class="log-step">{step_marker}</span> {log_content}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying simulation event log: {e}")
        logger.error(f"Error in display_event_log: {e}", exc_info=True)

# In integrated_app.py, add a new function to display impact visualizations

def display_impact_visualizations(results):
    """Display visualizations showing the impact of events on organizations"""
    if not results.get('detailed_events'):
        st.info("Detailed event tracking not available - run a new simulation to see impact visualizations.")
        return
    
    st.subheader("Event Impact Analysis")
    
    # Group events by organization
    org_events = {}
    for event in results['detailed_events']:
        org_name = event.get('org_name')
        if org_name:
            if org_name not in org_events:
                org_events[org_name] = []
            org_events[org_name].append(event)
    
    # Display impact diagrams for each organization
    org_selector = st.selectbox(
        "Select Organization",
        list(org_events.keys()),
        key="impact_org_selector"
    )
    
    if org_selector in org_events:
        # Create organization data structure for visualization
        org_data = {
            'name': org_selector,
            'balance_history': results.get('org_balance_history', {}).get(org_selector, [])
        }
        
        # Get filtered events
        filtered_events = org_events[org_selector]
        
        # Create and display impact diagram
        impact_fig = transparent_viz.create_event_impact_diagram(
            org_data, 
            filtered_events
        )
        if impact_fig:
            st.plotly_chart(impact_fig, use_container_width=True)
# In integrated_app.py, add a new function to display ACAPS data insights

def display_acaps_insights(results):
    """Display insights from ACAPS severity data"""
    # Guard against None results
    if results is None:
        st.info("No simulation results available for ACAPS insights.")
        return
        
    hdx_params = results.get('hdx_params', {})
    # Guard against hdx_params being None
    if hdx_params is None:
        hdx_params = {}
        
    emergency_params = hdx_params.get('emergency_parameters', {})
    # Guard against emergency_params being None
    if emergency_params is None:
        emergency_params = {}
    
    # Safe string comparison - convert to string first
    data_source = str(emergency_params.get('data_source', ''))
    if not emergency_params or 'ACAPS' not in data_source:
        st.info("ACAPS data not available in this simulation.")
        return
    
    st.subheader("ACAPS Severity Insights")
    
    # Display crisis percentages if available
    if 'crisis_percentage' in emergency_params:
        crisis_cols = st.columns(3)
        with crisis_cols[0]:
            st.metric(
                "Crisis Level",
                f"{emergency_params.get('crisis_percentage', 0) * 100:.1f}%",
                help="Percentage of areas with severity index ≥ 3.5"
            )
        with crisis_cols[1]:
            st.metric(
                "Emergency Level",
                f"{emergency_params.get('emergency_percentage', 0) * 100:.1f}%",
                help="Percentage of areas with severity index ≥ 4.0"
            )
        with crisis_cols[2]:
            st.metric(
                "Catastrophe Level",
                f"{emergency_params.get('catastrophe_percentage', 0) * 100:.1f}%",
                help="Percentage of areas with severity index ≥ 4.5"
            )
    
    # If raw data is available, create explainer visualization
    acaps_data = hdx_params.get('raw_acaps_data')
    if acaps_data:
        try:
            thresholds = {'crisis': 3.5, 'emergency': 4.0, 'catastrophe': 4.5}
            acaps_fig = transparent_viz.create_acaps_severity_explainer(acaps_data, thresholds)
            st.plotly_chart(acaps_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating ACAPS visualization: {e}")
            logger.error(f"Error in ACAPS visualization: {e}", exc_info=True)

def main():
    """Main application entry point"""
    # Set page title and icon (already done at the top of the script)

    # Display introduction
    show_introduction()

    # Get simulation parameters
    params = get_parameters()

    # HDX API information expander
    if HDX_AVAILABLE:
        with st.expander("HDX & ACAPS API Integration Information"):
            st.markdown("""
            ### Using the HDX & ACAPS API Integration
            
            This simulator connects to APIs to fetch real-time data for risk parameters:
            
            **Benefits of using the APIs:**
            - Get up-to-date security incident data from ACLED via HDX
            - Access current ACAPS INFORM Severity Index for emergency parameters
            - More accurate risk modeling based on real conditions
            
            **How it works:**
            1. Check "Use HDX Data" in the sidebar to enable API integration
            2. The simulator uses a pre-configured API key
            3. Check "Clear data cache" if you want to fetch fresh data
            
            If the API connections fail, the simulator will automatically fall back to cached or simulated data.
            """)

    # Run simulation button with options
    run_col1, run_col2 = st.columns([2, 1])
    with run_col1:
        run_button = st.button("Run Simulation", key="run_simulation", use_container_width=True)
    with run_col2:
        show_details = st.checkbox("Show details during run", value=True, 
                                help="Displays detailed information during simulation run")

    # Check if we should run a new simulation or use cached results
    run_new_simulation = run_button
    display_saved_results = st.session_state.simulation_results is not None

    if run_new_simulation:
        try:
            # Run the model
            results = run_model_and_get_results(params)

            if results:
                # Save results and parameters to session state
                st.session_state.simulation_results = results
                st.session_state.params = params.copy()
                display_saved_results = True  # Set flag to display results
            else:
                st.error("Simulation failed. Please check the logs for more information.")

        except Exception as e:
            st.error(f"An unexpected error occurred during simulation: {e}")
            logger.error(f"Unexpected error in main simulation: {e}", exc_info=True)

    # Display saved results (either from a new run or from cache)
    if display_saved_results:
        try:
            # Get results and parameters from session state
            results = st.session_state.simulation_results
            saved_params = st.session_state.params
            
            # Display various analyses and visualizations
            st.markdown("## Simulation Results")

            # Metrics
            display_metrics(results)

            # Continuity Impact Plot
            plot_continuity_impact(results)

            # Profit and Impact Plot
            plot_profit_and_impact(results)

            # Key Insights
            display_key_insights(results)

            # Event Log
            display_event_log(results)

            st.markdown("## Model Explanation")
            try:
                # Just display the text part first without full visualization
                st.markdown("### How Claim Triggers Work")
                st.markdown(f"""
                Organizations can make insurance claims when their balance drops below a certain threshold.
                - Current claim trigger: {saved_params.get('claim_trigger', 0.5) * 100:.0f}% of budget
                - Waiting period: {saved_params.get('waiting_period', 2)} months between claims
                """)
            except Exception as e:
                st.error(f"Error in basic explanation: {e}")
                logger.error(f"Error in basic explanation: {e}", exc_info=True)

            st.markdown("## Organization Claim Thresholds")
            gauge_cols = st.columns(len(results['final_org_states']))
            for i, org in enumerate(results['final_org_states']):
                with gauge_cols[i]:
                    try:
                        # Use .get() with default values for all parameters to prevent errors
                        gauge_fig = transparent_viz.create_threshold_gauge(
                            org.get('name', f"Organization {i+1}"), 
                            org.get('org_type', "Unknown"), 
                            org.get('balance', 0), 
                            org.get('total_budget', 100000), 
                            org.get('claim_trigger', 0.5)  # Default to 0.5 if missing
                        )
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating gauge for {org.get('name', f'Organization {i+1}')}: {e}")
                        logger.error(f"Error in gauge visualization: {e}", exc_info=True)

            # ACAPS Data Insights
            display_acaps_insights(results)

            # Impact Visualizations
            display_impact_visualizations(results)

            # Application Logs
            if show_details:
                display_logs()
            else:
                st.error("Simulation failed. Please check the logs for more information.")

        except Exception as e:
            st.error(f"An unexpected error occurred during simulation: {e}")
            logger.error(f"Unexpected error in main simulation: {e}", exc_info=True)
            
    # HDX Timeline Map Section
    if MAP_AVAILABLE and HDX_AVAILABLE:
        st.markdown("---")
        st.markdown("## 📍 HDX Events Timeline Map")
        
        map_tab1, map_tab2 = st.tabs(["🗺️ Interactive Map", "ℹ️ About the Map"])
        
        with map_tab1:
            try:
                # Run the map visualization
                sudan_map_timeline.main()
            except Exception as map_error:
                st.error(f"Error loading timeline map: {map_error}")
                logger.error(f"Map error: {map_error}", exc_info=True)
        
        with map_tab2:
            st.markdown("""
            ### About the HDX Events Timeline Map
            
            **Data Source Correction**: 
            - **🔴 Emergency Events**: Now correctly sourced from HAPI/ACLED conflict data
            - **🟠 Security Events**: Now correctly sourced from ACAPS INFORM Severity Index
            
            **Features**:
            - **Timeline Animation**: 1 second = 1 month of real data from 2023
            - **Interactive Controls**: Play, pause, reset, and manual timeline scrubbing
            - **Hover Information**: Click/hover on events to see detailed descriptions
            """)
    elif not MAP_AVAILABLE:
        st.info("📍 HDX Events Timeline Map is not available due to import issues.")
    elif not HDX_AVAILABLE:
        st.info("📍 HDX Events Timeline Map requires HDX integration to be available.")

# Custom CSS for improved styling
st.markdown("""
<style>
/* Force dark text colors for better visibility */
.intro-panel h2, .intro-panel p, .intro-panel ul, .intro-panel li, .intro-panel strong {
    color: #000000 !important; /* Black text */
}

.intro-panel {
    background-color: #f8f9fa;
    border-left: 4px solid #007bff;
    padding: 15px;
    margin-bottom: 20px;
    border-radius: 4px;
    color: #000000 !important; /* Force black text on the container */
}

/* Original CSS preserved */
.metric-box {
    border: 1px solid #e0e0e0;
    border-radius: 5px;
    padding: 10px;
    text-align: center;
    background-color: #f9f9f9;
}
.plot-container {
    margin: 20px 0;
    border: 1px solid #e0e0e0;
    border-radius: 5px;
    padding: 15px;
}
.log-container {
    max-height: 500px;
    overflow-y: auto;
    border: 1px solid #e0e0e0;
    border-radius: 5px;
    padding: 10px;
}
.log-entry {
    margin-bottom: 5px;
    padding: 5px;
    border-bottom: 1px solid #f0f0f0;
}
.log-entry.log-header {
    font-weight: bold;
    background-color: #f0f0f0;
}
.log-entry.org-ngo {
    color: #3498db;
}
.log-entry.org-un {
    color: #2ecc71;
}
.log-entry.org-hybrid {
    color: #9b59b6;
}
.log-step {
    font-weight: bold;
    margin-right: 10px;
    color: #7f8c8d;
}
.parameter-title {
    font-size: 18px;
    font-weight: bold;
    margin-top: 15px;
    margin-bottom: 10px;
}

/* Additional Streamlit-specific selectors to force dark text */
.css-1fv8s86 p, .css-1fv8s86 h1, .css-1fv8s86 h2, .css-1fv8s86 h3, 
.css-1fv8s86 li, .css-1fv8s86 span, .css-1fv8s86 div {
    color: #000000 !important; 
}

/* Fix for Streamlit markdown text */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# Run the main application
if __name__ == "__main__":
    main()
                
